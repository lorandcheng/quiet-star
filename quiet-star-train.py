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
import argparse
#from huggingface_custom_callback import EarlyStoppingCallback
#import wasn't used and caused import error lol
from eval_helpers import preprocess_eval_function_gsm, preprocess_eval_function_csqa, preprocess_function, compute_metrics, truncate_or_pad
random_seed = 42
torch.manual_seed(random_seed)
random.seed(random_seed)
import numpy as np
np.random.seed(random_seed)

# # Only eval on subset of eval dataset to save time
# class TrainerEvalSampling(Trainer):
#     def __init__(self, *args, eval_sample_size=16, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.eval_sample_size = eval_sample_size
#     def get_eval_dataloader(self, eval_dataset=None):
#         '''
#         Samples the evaluation dataset and returns a subset 
#         of size self.eval_sample_size.
#         '''
#         if eval_dataset is None:
#             eval_dataset = self.eval_dataset
#         idxs = random.sample(range(len(eval_dataset)), self.eval_sample_size)
#         eval_subset = eval_dataset.select(idxs)
#         return super().get_eval_dataloader(eval_subset)


# MAIN SETUP
root_prefix = os.path.expanduser("~/scratch/quietSTAR/")
wandb_cache_dir = root_prefix + "cache/quietstar/wandb_cache"
dataset_name = 'open-web-math/open-web-math'
# dataset_name = 'c4'
project_name = "quiet-star"
os.environ["WANDB_PROJECT"] = project_name + "-" + dataset_name.split("/")[-1]
os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir

# Args
parser = argparse.ArgumentParser()
parser.add_argument("--n_ahead_talk_global", type=int, default=4)
parser.add_argument("--n_passes_global", type=int, default=2)
parser.add_argument("--n_ahead_global", type=int, default=12)
parser.add_argument("--n_examples", type=int, default=1_000)
parser.add_argument("--full_batch_size", type=int, default=8)
parser.add_argument("--train_steps", type=int, default=1000) # kill after this many steps
parser.add_argument("--eval_and_log_every", type=int, default=10) # eval and log after this many train steps
parser.add_argument("--eval_pct", type=int, default=10) # cut eval short after this % of the dataset
parser.add_argument("--save_steps", type=int, default=100)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--use_meta_prompt", action="store_true")

args = parser.parse_args()

def model_init(params):
    original = False
    if params is None:
        params = {}
    else:
        params = params.params
    # save params to file
    n_ahead = params.get("n_ahead", args.n_ahead_global if not original else 1)
    n_ahead_talk = params.get("n_ahead_talk", args.n_ahead_talk_global if not original else 1)
    n_passes = params.get("n_passes", args.n_passes_global if not original else 1)
    gumbel_temperature = params.get("gumbel_temperature", 1)
    use_start_thought_token = params.get("use_start_thought_token", True)
    use_end_thought_token = params.get("use_end_thought_token", True)
    include_policy_loss = params.get("include_policy_loss", True)
    gumbel_detach = params.get("gumbel_detach", True)
    merged_talk_heads = params.get("merged_talk_heads", True)
    gradient_accumulation_steps = params.get("gradient_accumulation_steps", global_gradient_accumulation_steps)
    residual_think_head = params.get("residual_think_head", False)
    optimize_lm_head_only_at_start = params.get("optimize_lm_head_only_at_start", False)

    model_name = "mistralai/Mistral-7B-v0.1" if args.checkpoint is None else args.checkpoint
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
    model.n_ahead = n_ahead
    model.n_ahead_talk = n_ahead_talk
    model.n_passes = n_passes
    model.n_tokens_print = gradient_accumulation_steps
    model.gradient_accumulation_steps = gradient_accumulation_steps
    model.residual_think_head = residual_think_head
    model.optimize_lm_head_only_at_start = optimize_lm_head_only_at_start
    model.gumbel_temperature = gumbel_temperature
    model.wandb_enabled = True
    model.original_mode = original
    model.config_params = params
    model.run_start = int(time.time())
    model.kill_after = args.train_steps
    model.use_meta_prompt = args.use_meta_prompt
    model.train()
    return model

# Load dataset
import datasets
dataset = load_dataset(
    dataset_name,
    "en" if "c4" in dataset_name else "default",
    split=f"train[:{args.n_examples}]",
    # ignore_verifications=True,
    verification_mode=datasets.VerificationMode.NO_CHECKS,
    # num_proc=16,
    cache_dir=root_prefix + "cache/datasets/",
)

train_dataset = dataset.shuffle(seed=random_seed).map(preprocess_function, batched=True, writer_batch_size=200)
eval_dataset_gsm = load_dataset("gsm8k", "main", split=f"test[:{args.eval_pct}%]", verification_mode=datasets.VerificationMode.NO_CHECKS).map(preprocess_eval_function_gsm, batched=True, writer_batch_size=200)
eval_dataset_csqa = load_dataset("tau/commonsense_qa", "default", split=f"validation[:{args.eval_pct}%]", verification_mode=datasets.VerificationMode.NO_CHECKS).map(preprocess_eval_function_csqa, batched=True, writer_batch_size=200)

eval_datasets = {
    "gsm8k": eval_dataset_gsm,
    "csqa": eval_dataset_csqa,
}

batch_size = args.full_batch_size // args.n_passes_global
global_gradient_accumulation_steps = args.full_batch_size // batch_size
run_id = int(time.time())
training_args = TrainingArguments(
    output_dir=root_prefix + f"cache/quietstar/{run_id}",
    learning_rate=1e-6,
    optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=global_gradient_accumulation_steps,
    max_grad_norm=1.0,
    max_steps=100000,
    warmup_steps=20,
    auto_find_batch_size=True,
    weight_decay=0.001,
    label_names=["labels"],
    include_inputs_for_metrics=True,
    logging_steps=args.eval_and_log_every,
    eval_steps=args.eval_and_log_every,
    evaluation_strategy="steps",
    save_safetensors=False,
    save_steps=args.save_steps,
    run_name=f"n={args.n_ahead_global}_nt={args.n_ahead_talk_global}_np={args.n_passes_global}",
    save_total_limit = 1, #running out of scratch storage, only save latest ckpt
    torch_empty_cache_steps = 10,
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