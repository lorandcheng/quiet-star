import torch
torch.backends.cuda.matmul.allow_tf32 = True
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline, AutoConfig
from accelerate import infer_auto_device_map, init_empty_weights, dispatch_model
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from transformers import TrainingArguments, Trainer
import os
import time
import wandb
#from huggingface_custom_callback import EarlyStoppingCallback
#import wasn't used and caused import error lol
from eval_helpers import preprocess_eval_function_gsm, preprocess_eval_function_csqa, get_preprocess_function, compute_metrics, truncate_or_pad
random_seed = 42
torch.manual_seed(random_seed)
random.seed(random_seed)
import numpy as np
np.random.seed(random_seed)


# MAIN SETUP
root_prefix = "./quietSTAR/"
wandb_cache_dir = root_prefix + "cache/quietstar/wandb_cache"
dataset_name = 'open-web-math/open-web-math'
# dataset_name = 'c4'
project_name = "quiet-star"
os.environ["WANDB_PROJECT"] = project_name + "-" + dataset_name.split("/")[-1]
os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir
n_ahead_talk_global = 4
n_passes_global = 2
n_ahead_global = 12
n_examples = 1_000
full_batch_size = 4
eval_and_logging_steps = 10
save_steps = 100

pct_test = 1
max_seq_len = 128
gradient_accumulation_steps = 4
use_meta_prompt = False
initial_start_token = "---"
embedding_scale = 1e2
wandb_enabled = True
debug_prefix = ""
cycling_sot_token = False
reward_modeling = False
rm_beta = 0.1
policy_loss_beta = 1e6


# experiment with next as the start token, and no meta_prompt
# use_meta_prompt = False
# initial_start_token = "next"
# embedding_scale = 1
# gradient_accumulation_steps = 4
# full_batch_size = 4

# experiment with cycling sot token ie multiple 1 scaled variations on the sot token. the end of thought token is still scaled, and the same ---
# cycling_sot_token = True
# use_meta_prompt = False
# gradient_accumulation_steps = 4
# full_batch_size = 4

# meta prompting setting
use_meta_prompt = True
gradient_accumulation_steps = 8
full_batch_size = 2


# reward modeling
# cycling_sot_token = False
# use_meta_prompt = False
# reward_modeling = True
# rm_beta = 0.05
# gradient_accumulation_steps = 4
# full_batch_size = 4
# policy_loss_beta = 1e4


# debug settings
# use_meta_prompt = False
# gradient_accumulation_steps = 1
# full_batch_size = 2
# max_seq_len = 16
# eval_and_logging_steps = 1 # testing the debug. Infer seems wrong???
# wandb_enabled = False
# # initial_start_token = 'next'
# cycling_sot_token = True
# debug_prefix = "DEBUG_"

# global necessary for some reason
global_gradient_accumulation_steps = gradient_accumulation_steps * n_passes_global
def model_init(params):
    original = False
    if params is None:
        params = {}
    else:
        params = params.params
    # save params to file
    n_ahead = params.get("n_ahead", n_ahead_global if not original else 1)
    n_ahead_talk = params.get("n_ahead_talk", n_ahead_talk_global if not original else 1)
    n_passes = params.get("n_passes", n_passes_global if not original else 1)
    gumbel_temperature = params.get("gumbel_temperature", 1)
    use_start_thought_token = params.get("use_start_thought_token", True)
    use_end_thought_token = params.get("use_end_thought_token", True)
    include_policy_loss = params.get("include_policy_loss", True)
    gumbel_detach = params.get("gumbel_detach", True)
    merged_talk_heads = params.get("merged_talk_heads", True)
    gradient_accumulation_steps = params.get("gradient_accumulation_steps", global_gradient_accumulation_steps)
    residual_think_head = params.get("residual_think_head", False)
    optimize_lm_head_only_at_start = params.get("optimize_lm_head_only_at_start", False)

    model_name = "mistralai/Mistral-7B-v0.1"
    print("Loading model")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map='auto',
        cache_dir=root_prefix + "cache",
        max_thoughts=n_ahead + n_ahead_talk + 1,
        merged_talk_heads=merged_talk_heads,
        merged_lm_and_talk_heads=False,
        merged_lm_and_think_heads=True,
        use_concat_talk_head=True,
        use_shallow_think=True,
        use_shallow_talk=False,
        use_complex_think_head=False,
        use_complex_talk_head=True,
        use_weighted_talk_head=True,
    )
    print("Loaded model")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    special_tokens_to_add = []
    if model.use_start_thought_token:
        special_tokens_to_add.append("<|startthought|>")
    if model.use_end_thought_token:
        special_tokens_to_add.append("<|endthought|>")
    if special_tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_to_add})
        model.resize_token_embeddings(len(tokenizer))
    model.tokenizer = tokenizer
    model.gumbel_detach = gumbel_detach
    model.include_policy_loss = include_policy_loss
    model.use_end_thought_token = use_end_thought_token
    model.use_start_thought_token = use_start_thought_token
    model.use_meta_prompt = use_meta_prompt
    model.n_ahead = n_ahead
    model.n_ahead_talk = n_ahead_talk
    model.n_passes = n_passes
    model.n_tokens_print = gradient_accumulation_steps
    model.gradient_accumulation_steps = gradient_accumulation_steps
    model.residual_think_head = residual_think_head
    model.optimize_lm_head_only_at_start = optimize_lm_head_only_at_start
    model.gumbel_temperature = gumbel_temperature
    model.wandb_enabled = wandb_enabled
    model.original_mode = original
    model.config_params = params
    model.run_start = int(time.time())
    model.kill_after = 500
    model.train()

    model.embedding_scale = embedding_scale
    model.initial_start_token = initial_start_token
    model.cycling_sot_token = cycling_sot_token
    model.reward_modeling = reward_modeling
    model.rm_beta = rm_beta
    model.policy_loss_beta = policy_loss_beta
    return model

def main():
    # Load dataset
    import datasets
    dataset = load_dataset(
        dataset_name,
        "en" if "c4" in dataset_name else "default",
        split=f"train[:{n_examples}]",
        # ignore_verifications=True,
        verification_mode=datasets.VerificationMode.NO_CHECKS,
        # num_proc=16,
        cache_dir=root_prefix + "cache/datasets/",
    )

    train_dataset = dataset.shuffle(seed=random_seed).map(get_preprocess_function(max_seq_len), batched=True, writer_batch_size=200, remove_columns=["text"])

    eval_dataset_gsm = load_dataset("gsm8k", "main", split=f"test[:{pct_test}%]", verification_mode=datasets.VerificationMode.NO_CHECKS).map(preprocess_eval_function_gsm, batched=True, writer_batch_size=200)
    eval_dataset_csqa = load_dataset("tau/commonsense_qa", "default", split=f"validation[:{pct_test}%]", verification_mode=datasets.VerificationMode.NO_CHECKS).map(preprocess_eval_function_csqa, batched=True, writer_batch_size=200)

    eval_datasets = {
        "gsm8k": eval_dataset_gsm,
        "csqa": eval_dataset_csqa,
    }

    batch_size = full_batch_size // n_passes_global
    global_gradient_accumulation_steps = full_batch_size // batch_size * gradient_accumulation_steps
    run_id = int(time.time())
    training_args = TrainingArguments(
        # use_cpu=True,
        # no_cuda=True, 
        output_dir=root_prefix + f"cache/quietstar/{run_id}",
        learning_rate=1e-6,
        optim="adamw_8bit", # "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=global_gradient_accumulation_steps,
        max_grad_norm=1.0,
        max_steps=100000,
        warmup_steps=20,
        auto_find_batch_size=True, # doesn't seem to work for my a40 Jakob
        weight_decay=0.001,
        label_names=["labels"],
        include_inputs_for_metrics=True,
        logging_steps=eval_and_logging_steps,
        eval_steps=eval_and_logging_steps,
        evaluation_strategy="steps",
        save_safetensors=False,
        save_steps=save_steps,
        run_name=f"{debug_prefix}n={n_ahead_global}_nt={n_ahead_talk_global}_np={n_passes_global}_fbz={full_batch_size}_seql={max_seq_len}_gacc={gradient_accumulation_steps}_mp={use_meta_prompt}_sot={initial_start_token}_embsc={embedding_scale}_cyc={cycling_sot_token}_rm={reward_modeling}_rmb={rm_beta}_plb={policy_loss_beta}",
        save_total_limit = 1, #running out of scratch storage, only save latest ckpt
    )
    trainer = Trainer(
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        compute_metrics=compute_metrics,
        model_init=model_init,
    )

    trainer.train()
    #enter full path of checkpoint, including checkpoint-xxxx
if __name__ == "__main__":
    main()