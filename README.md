# Quiet-STaR

Code for [Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629).

This project is implemented by simply patching the base Mistral implementation in Huggingface `transformers` using a new `modeling_mistral.py` and a new `configuration_mistral.py` and otherwise applying standard `transformers` features (e.g. the default Trainer). Our patches were applied to Huggingface's `transformers` version `4.37.0.dev0` under `src/transformers/models/mistral/` -- we cannot guarantee that other changes to their implementation will not affect our implementation, so for reproducibility, we encourage using the same version.

One pitfall to be wary of: the model is not taught not to generate start and end thought tokens. Thus, when performing actual inference, it is necessary to mask these out.

We make an 8-thought-token ahead (including start and end tokens) model [available via Huggingface](https://huggingface.co/ezelikman/quietstar-8-ahead).


python launch.py -m model_name="/nethome/jbjorner3/dev/hallucination-fun/quiet-star/quietSTAR/cache/quietstar/1733353818/checkpoint-500" only_train_rm_head=True rm_loss_beta=1 pct_test=10 optimizer_init=4

python launch.py -m original_loss_weight=1 n_ahead_global=1 n_ahead_talk_global=1 policy_loss_beta=0 rm_loss_beta=0 optimizer_init=-1 debug_prefix=continuepretrainingmoreevalhigherbatchlr pct_test=50 lr=5e-6 gradient_accumulation_steps=16

python launch.py -m train=False model_name="/nethome/jbjorner3/dev/hallucination-fun/quiet-star/quietSTAR/cache/quietstar/1733079553/checkpoint-100" pct_test=100

python launch.py -m original_loss_weight=1 n_ahead_global=1 n_ahead_talk_global=1 policy_loss_beta=0 rm_loss_beta=0 optimizer_init=-1 debug_prefix=rmembscaling use_end_thought_token=False use_start_thought_token=False use_end_thought_embedding=True use_start_thought_embedding=True optimizer_init=3