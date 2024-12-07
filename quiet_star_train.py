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


def main(cfg):

    
    # MAIN SETUP
    root_prefix = "./quietSTAR/"
    wandb_cache_dir = root_prefix + "cache/quietstar/wandb_cache"
    dataset_name = 'open-web-math/open-web-math'
    # dataset_name = 'c4'
    project_name = "quiet-star"
    os.environ["WANDB_PROJECT"] = project_name + "-" + dataset_name.split("/")[-1]
    os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir
    n_ahead_talk_global = cfg.n_ahead_talk_global
    n_passes_global = 2
    n_ahead_global = cfg.n_ahead_global
    n_examples = cfg.n_examples # n_examples
    full_batch_size = cfg.full_batch_size # 4
    eval_and_logging_steps = cfg.eval_and_logging_steps
    save_steps = cfg.save_steps

    pct_test = cfg.pct_test # 1
    max_seq_len = cfg.max_seq_len # 128
    gradient_accumulation_steps = cfg.gradient_accumulation_steps # 4
    use_meta_prompt = cfg.use_meta_prompt # False
    initial_start_token = cfg.initial_start_token # "---"
    embedding_scale = cfg.embedding_scale # 1e2
    wandb_enabled = cfg.wandb_enabled # True
    debug_prefix = cfg.debug_prefix # ""
    cycling_sot_token = cfg.cycling_sot_token # False
    use_rm_loss = cfg.use_rm_loss # False
    rm_beta = cfg.rm_beta # 0.1
    policy_loss_beta = cfg.policy_loss_beta # 1e6
    rm_loss_beta = cfg.rm_loss_beta # 1
    meta_prompt_list = cfg.meta_prompt_list # []
                        # ['identifying useful intermediate quantities',
                        # 'considering relevant problem parameters',
                        # 'exploring relationships between variables',
                        # 'next parameter to compute is',
                        # 'necessary variable to be known is']


    # test the base code with massive gradient accumulation. 
    # debug_prefix = "grad_accum_"
    # full_batch_size = 2 
    # gradient_accumulation_steps = 16 * 2

    # experiment with next as the start token, and no meta_prompt
    # debug_prefix = "SOT_change_"
    # use_meta_prompt = False
    # initial_start_token =  "identifying" # "considering" # "exploring" # "necessary"  # "next" # '---' #
    # embedding_scale = 1
    # gradient_accumulation_steps = 4
    # full_batch_size = 4

    # experiment with cycling sot token ie multiple 1 scaled variations on the sot token. the end of thought token is still scaled, and the same ---
    # cycling_sot_token = True
    # use_meta_prompt = False
    # gradient_accumulation_steps = 4
    # full_batch_size = 4

    # meta prompting setting
                # ['identifying useful intermediate quantities']
                # ['considering relevant problem parameters']
                # ['exploring relationships between variables']
                # ['next parameter to compute is']
                # ['necessary variable to be known is']
    # meta_prompt_list = ['necessary variable to be known is']
    # use_meta_prompt = True
    # gradient_accumulation_steps = 8
    # full_batch_size = 2


    # # reward modeling
    # cycling_sot_token = False
    # use_meta_prompt = False
    # use_rm_loss = True
    # rm_beta = 0.05
    # # policy_loss_beta = 1e4
    # gradient_accumulation_steps = 4
    # full_batch_size = 4
    # rm_loss_beta = 1


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
        # use_start_thought_token = params.get("use_start_thought_token", True)
        # use_end_thought_token = params.get("use_end_thought_token", True)
        include_policy_loss = params.get("include_policy_loss", True)
        gumbel_detach = params.get("gumbel_detach", True)
        merged_talk_heads = params.get("merged_talk_heads", True)
        gradient_accumulation_steps = params.get("gradient_accumulation_steps", global_gradient_accumulation_steps)
        residual_think_head = params.get("residual_think_head", False)
        optimize_lm_head_only_at_start = params.get("optimize_lm_head_only_at_start", False)

        model_name = cfg.model_name # 
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

        use_start_thought_token = cfg.use_start_thought_token
        use_end_thought_token = cfg.use_end_thought_token
        model.use_start_thought_token = use_start_thought_token
        model.use_end_thought_token = use_end_thought_token
        model.use_end_thought_embedding = cfg.use_end_thought_embedding 
        model.use_start_thought_embedding = cfg.use_start_thought_embedding 

        special_tokens_to_add = []
        if model.use_start_thought_token:
            special_tokens_to_add.append("<|startthought|>")
        if model.use_end_thought_token:
            special_tokens_to_add.append("<|endthought|>")
        if special_tokens_to_add:
            tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_to_add})
            if model_name != "mistralai/Mistral-7B-v0.1": # check if we are loading from a checkpoint.
                model.start_token_id = tokenizer.convert_tokens_to_ids("<|startthought|>")
                model.end_token_id = tokenizer.convert_tokens_to_ids("<|endthought|>")
                model.rm_initialized = True
            else:
                model.resize_token_embeddings(len(tokenizer)) # this resizes the LM_head as well with the same weights... how do they have the same weights tho? I think these weights are random weights?
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
        model.kill_after = cfg.kill_after
        model.train()

        model.embedding_scale = embedding_scale
        model.initial_start_token = initial_start_token
        model.initial_end_token = cfg.initial_end_token
        model.cycling_sot_token = cycling_sot_token
        model.use_rm_loss = use_rm_loss
        model.rm_beta = rm_beta
        model.policy_loss_beta = policy_loss_beta
        model.rm_loss_beta = rm_loss_beta
        model.meta_prompt_list = meta_prompt_list
        model.only_train_mixing_head = cfg.only_train_mixing_head
        model.original_loss_weight = cfg.original_loss_weight
        model.only_train_rm_head = cfg.only_train_rm_head
        return model



    # Load dataset
    import datasets
    dataset = load_dataset(
        dataset_name,
        "en" if "c4" in dataset_name else "default",
        split=f"train[:{n_examples}]",
        # ignore_verifications=True,
        verification_mode=datasets.VerificationMode.NO_CHECKS,
        num_proc=16,
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
        weight_decay=0.001,
        learning_rate=cfg.lr,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",  # 
        warmup_steps=20,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=global_gradient_accumulation_steps,
        max_grad_norm=1.0,
        max_steps=100000,
        auto_find_batch_size=True, # doesn't seem to work for my a40 Jakob
        label_names=["labels"],
        include_inputs_for_metrics=True,
        logging_steps=eval_and_logging_steps,
        eval_steps=eval_and_logging_steps,
        evaluation_strategy="steps",
        save_safetensors=False,
        save_steps=save_steps,
        run_name=(f"{debug_prefix}n={n_ahead_global}_nt={n_ahead_talk_global}"
                  f"_np={n_passes_global}_fbz={full_batch_size}_seql={max_seq_len}"
                  f"_gacc={gradient_accumulation_steps}_sot={initial_start_token}"
                  f"_eot={cfg.initial_end_token}"
                  f"_embsc={embedding_scale}_cyc={cycling_sot_token}"
                  f"_mp={use_meta_prompt}_{meta_prompt_list}_plb={policy_loss_beta}"
                  f"_rm={use_rm_loss}_rmb={rm_beta}_rmlb={rm_loss_beta}"
                  f"_mixo={cfg.only_train_mixing_head}_opti={cfg.optimizer_init}"
                  f"_olw={cfg.original_loss_weight}_lr={cfg.lr}"),
        save_total_limit = 1, #running out of scratch storage, only save latest ckpt
    )
    if cfg.only_train_rm_head:
        assert cfg.optimizer_init != -1, "only one optimizer available when only training reward modeling head"

    if cfg.optimizer_init == -1:
        # just do the same as quiet-star original repo.

        trainer = Trainer(
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_datasets,
            compute_metrics=compute_metrics,
            model_init=model_init,
        )
    else:
        
        model = model_init(None)
        import transformers
        if cfg.only_train_rm_head:
            parameters = model.reward_modeling_head.parameters()
        elif cfg.optimizer_init == 0:
            parameters = model.parameters()
            # other parameter options for different tuning ideas....
        elif cfg.optimizer_init == 1:
            parameters = [
                        #   {"params": (p for parameters in [model.model.parameters(), model.lm_head.parameters()] for p in parameters)},
                          {"params": model.model.parameters()}, # this should work but doesn't. I have to move on. (this means I have to abandon the ideas associated with training my own value head?) In which case keep investigating.
                          {"params": model.lm_head.parameters()}, # this parameter is weight tied, so don't optimize it..
                          {"params": [model.start_embedding]},
                          {"params": [model.end_embedding]},
                          {"params": model.reward_modeling_head.parameters(), "lr": cfg.lr},
                          {"params": model.talk_head.parameters(), "lr": cfg.lr}] 
        elif cfg.optimizer_init == 2:
            parameters = [
                          {"params": model.model.parameters()},
                        #   {"params": model.lm_head.parameters()}, 
                          {"params": [model.start_embedding]},
                          {"params": [model.end_embedding]},
                          {"params": model.reward_modeling_head.parameters(), "lr": cfg.lr},
                          {"params": model.talk_head.parameters(), "lr": cfg.lr}] 
        elif cfg.optimizer_init == 3:
            parameters = [
                          {"params": model.model.parameters()},
                          {"params": model.lm_head.parameters()}, 
                        #   {"params": [model.start_embedding]},
                        #   {"params": [model.end_embedding]},
                          {"params": model.reward_modeling_head.parameters(), "lr": cfg.lr},
                          {"params": model.talk_head.parameters(), "lr": cfg.lr}] 
        else:
            print("no implementation for", cfg.optimizer_init)
            raise NotImplemented
        optim = torch.optim.AdamW(parameters,
                                lr=cfg.lr,
                                weight_decay=0.001,
                                fused=True)
        scheduler = transformers.get_linear_schedule_with_warmup(optim, num_warmup_steps=20, num_training_steps=100000)


        trainer = Trainer(
            model = model,
            args=training_args,
            optimizers=(optim, scheduler),
            train_dataset=train_dataset,
            eval_dataset=eval_datasets,
            compute_metrics=compute_metrics,
            # model_init=model_init,
        )
    if cfg.train:
        trainer.train()
    else:
        print(trainer.evaluate(eval_datasets))
    #enter full path of checkpoint, including checkpoint-xxxx
if __name__ == "__main__":
    from launch import DPORunConfig, DebugConfig
    # Check if the following produces good rm_loss also in debugger check to make sure that the only thing which gets gradients is the rm_head not the entire model. I 
    confg = DPORunConfig(optimizer_init=4, debug_prefix="DEBUG", model_name="/nethome/jbjorner3/dev/hallucination-fun/quiet-star/quietSTAR/cache/quietstar/1733353818/checkpoint-500", only_train_rm_head=True, rm_loss_beta=1) #, eval_and_logging_steps=1) #, **DebugConfig)
    # , use_end_thought_token=False, use_start_thought_token=False, use_end_thought_embedding=True, use_start_thought_embedding=True
    
    main(confg)