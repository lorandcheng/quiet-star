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
    exclude: str|None = "major,crushinator,nestor,voltron,xaea-12,samantha,protocol"
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
    # known issue: the multirun.yaml is saved to the sweep dir, and not the subdirs, so it is not saved! (don't think I will need this to be saved tho, and makes folders easier to read)
    hydra: Any = field(default_factory=lambda: {
        "sweep":{"dir": "star_runs", 
                 "subdir": "${run_type}_${now:%Y-%m-%d}/${now:%H-%M-%S}_${hydra.job.num}" },
        "run":{"dir":  "star_runs/${run_type}_${now:%Y-%m-%d}/${now:%H-%M-%S}"},
        # "sweeper": {"sampler": "random"},
    })
cs.store(name="DPORunConfig", node=DPORunConfig)


@hydra_main(version_base=None, config_name='DPORunConfig')
def my_app(cfg: DPORunConfig) -> None:
    from quiet_star_train import main
    main()
    
if __name__ == "__main__":
    my_app()