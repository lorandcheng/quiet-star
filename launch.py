from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from hydra import main as hydra_main
from hydra.utils import instantiate
import os
import random
import numpy as np
# from run_config import RunConfig
from omegaconf import MISSING, OmegaConf
from hydra.core.config_store import ConfigStore
from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf

cs = ConfigStore.instance()
# from config import CustomKargoLauncherConfig # necesary for config store TODO: make this better.


from typing import Any, List, Dict
from dataclasses import dataclass, field

@dataclass
class CustomKargoLauncherConfig(SlurmQueueConf): 
    """ https://hydra.cc/docs/1.3/plugins/submitit_launcher/ then go to github and look at config.py this is what I extend.
        to run things locally, use the option on launch `python run.py hydra/launcher=submitit_local`, 
        or in this case, without -m it launches things locally.
    """
    # submitit_folder: str = 
    # the default submitit_folder = "${hydra.sweep.dir}/.submitit/%j"
    # so reasonable and can't make it anything more reasonable it seems, because 
    # they launch with map_executor on the backend, which is the best for my 
    # parallel jobs, but prevents nicely putting the submitit loggs into more 
    # careful foldering. Perhaps in the future I can follow a per experiment 
    # foldering, and the specificity of the sweep.dir folder will be more useful 
    # to me.
    timeout_min: int = 2880 # 60 * 24 * 2
    # cpus_per_task: int|None = 6 # type: ignore
    gpus_per_node: int|None = None
    tasks_per_node: int =  1
    mem_gb: int|None =  None
    nodes: int = 1
    _target_: str = "hydra_plugins.hydra_submitit_launcher.submitit_launcher.SlurmLauncher"
    partition: str|None = "kargo-lab" # overcap
    qos: str|None = "short"
    exclude: str|None = "major,crushinator,nestor,voltron,xaea-12,samantha,protocol,puma,chappie"
    additional_parameters: Dict[str, Any] = field(default_factory=lambda: {"cpus-per-gpu": 6, "gpus-per-node": "a40:4", "requeue": True})
    array_parallelism: int = 20
cs.store(name="custom_kargo_submitit", node=CustomKargoLauncherConfig, group="hydra/launcher")

cs.store(name="custom_overcap_submitit", node=CustomKargoLauncherConfig(partition="overcap"), group="hydra/launcher")



@dataclass
class DPORunConfig:
    defaults: List[Any] = field(default_factory=lambda: [
        {"override hydra/launcher": "custom_overcap_submitit"},
        # {"override hydra/sweeper": "optuna"}, # https://hydra.cc/docs/plugins/optuna_sweeper/
        # {"override hydra/sweeper/sampler": "random"}, 
        "_self_",
        ])
    run_type: str = "quietSTAR"
    output_dir: str = "${hydra:runtime.output_dir}"
    

    pct_test: int = 1
    max_seq_len: int = 128
    gradient_accumulation_steps: int = 4
    use_meta_prompt: bool = False
    initial_start_token: str = "---"
    initial_end_token: str = "---"
    embedding_scale: float = 1e2
    wandb_enabled: bool = True
    debug_prefix: str = ""
    cycling_sot_token: bool = False
    use_rm_loss: bool = False
    rm_beta: float = 0.1
    policy_loss_beta: float = 1e6
    rm_loss_beta: float = 0
    meta_prompt_list: list[str] = field(default_factory=list)
    full_batch_size: int = 4
    kill_after: int = 500
    only_train_mixing_head: bool = False
    optimizer_init: int = -1
    original_loss_weight: float = 0.5
    n_ahead_global: int = 12
    n_ahead_talk_global: int = 4
    n_examples: int = 1_000
    use_end_thought_token: bool = True
    use_start_thought_token: bool = True
    use_end_thought_embedding: bool = False
    use_start_thought_embedding: bool = False
    eval_and_logging_steps: int = 20
    save_steps: int = 1000
    lr: float = 1e-6
    model_name: str = "mistralai/Mistral-7B-v0.1" # /nethome/jbjorner3/dev/hallucination-fun/quiet-star/quietSTAR/cache/quietstar/1733353818/checkpoint-500
    train: bool = True
    only_train_rm_head: bool = False

    # known issue: the multirun.yaml is saved to the sweep dir, and not the subdirs, so it is not saved! (don't think I will need this to be saved tho, and makes folders easier to read)
    hydra: Any = field(default_factory=lambda: {
        "sweep":{"dir": "star_runs", 
                 "subdir": "${run_type}_${now:%Y-%m-%d}/${now:%H-%M-%S}_${hydra.job.num}" },
        "run":{"dir":  "star_runs/${run_type}_${now:%Y-%m-%d}/${now:%H-%M-%S}"},
        # "sweeper": {"sampler": "random"},
    })
cs.store(name="DPORunConfig", node=DPORunConfig)


    # # reward modeling
    # cycling_sot_token = False
    # use_meta_prompt = False
    # use_rm_loss = True
    # rm_beta = 0.05
    # # policy_loss_beta = 1e4
    # gradient_accumulation_steps = 4
    # full_batch_size = 4
    # rm_loss_beta = 1
    # debug_prefix='rm_'
RewardModelingConfig= {
                        "cycling_sot_token": False,
                        "use_meta_prompt": False,
                        "use_rm_loss": True,
                        "rm_beta": 0.1,
                        # policy_loss_beta: 1e4 found 1e6 to perform well still????
                        "gradient_accumulation_steps": 4,
                        "full_batch_size": 4,
                        "rm_loss_beta": 1e6,
                        "debug_prefix":'rmdg_', # reward model disconnect gradient from hidden states before using rm_head.
                        'only_train_mixing_head': False,
                        # "rm_disco"
                      }
cs.store(name="RewardModelingConfig", node=RewardModelingConfig, group='experiment', package='__global__')


MetaPromptConfig= dict(
                        # "cycling_sot_token": False,
                        # "use_meta_prompt": False,
                        # "use_rm_loss": True,
                        # "rm_beta": 0.1,
                        # # policy_loss_beta: 1e4
                        # "gradient_accumulation_steps": 4,
                        # "full_batch_size": 4,
                        # "rm_loss_beta": 1,
                        # meta prompting setting
                                    # ['identifying useful intermediate quantities']
                                    # ['considering relevant problem parameters']
                                    # ['exploring relationships between variables']
                                    # ['next parameter to compute is']
                                    # ['necessary variable to be known is']
                        meta_prompt_list = ['necessary variable to be known is'],
                        use_meta_prompt = True,
                        gradient_accumulation_steps = 8,
                        full_batch_size = 2,
                        debug_prefix = 'mp_',
                       )
cs.store(name="MetaPromptConfig", node=MetaPromptConfig, group='experiment', package='__global__')


DebugConfig = dict(
    gradient_accumulation_steps = 1,
    full_batch_size = 2,
    max_seq_len = 16,
    # eval_and_logging_steps = 1,
    # initial_start_token = 'next',
    debug_prefix = "DEBUG_",
)
cs.store(name="DebugConfig", node=DebugConfig, group='experiment', package='__global__')

@hydra_main(version_base=None, config_name='DPORunConfig')
def my_app(cfg: DPORunConfig) -> None:
    from quiet_star_train import main
    main(cfg)
    
if __name__ == "__main__":
    my_app()