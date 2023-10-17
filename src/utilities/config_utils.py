import math
import os
import time
import warnings
from datetime import datetime
from typing import List, Sequence, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.utilities import rank_zero_only

from src.utilities import wandb_api
from src.utilities.naming import clean_name, get_detailed_name, get_group_name
from src.utilities.utils import get_logger, no_op
from src.utilities.wandb_api import get_existing_wandb_group_runs, get_run_api


log = get_logger(__name__)


@rank_zero_only
def print_config(
    config,
    fields: Union[str, Sequence[str]] = (
        "datamodule",
        "model",
        "trainer",
        # "callbacks",
        # "logger",
        "seed",
    ),
    resolve: bool = True,
    rich_style: str = "magenta",
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure (if installed: ``pip install rich``).

    Credits go to: https://github.com/ashleve/lightning-hydra-template

    Args:
        config (ConfigDict): Configuration
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
        rich_style (str, optional): Style of Rich library to use for printing. E.g "magenta", "bold", "italic", etc.
    """
    import importlib

    if not importlib.util.find_spec("rich") or not importlib.util.find_spec("omegaconf"):
        # no pretty printing
        print(OmegaConf.to_yaml(config, resolve=resolve))
        return
    import rich.syntax  # IMPORTANT to have, otherwise errors are thrown
    import rich.tree

    tree = rich.tree.Tree(":gear: CONFIG", style=rich_style, guide_style=rich_style)
    if isinstance(fields, str):
        if fields.lower() == "all":
            fields = config.keys()
        else:
            fields = [fields]

    for field in fields:
        branch = tree.add(field, style=rich_style, guide_style=rich_style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)


def extras(
    config: DictConfig, if_wandb_run_already_exists: str = "resume", allow_permission_error: bool = False
) -> DictConfig:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration
    - forcing multi-gpu friendly configuration
    - checking if config values are valid
    - init wandb if wandb logging is being used
    - Merge config with wandb config if resuming a run

    Credits go to: https://github.com/ashleve/lightning-hydra-template

    While this method modifies DictConfig mostly in place,
    please make sure to use the returned config as the new config, especially when resuming a run.

    Args:
        if_wandb_run_already_exists (str): What to do if wandb run already exists. Wandb logger must be enabled!
            Options are:
            - 'resume': resume the run
            - 'new': create a new run
            - 'abort': raise an error if run already exists and abort
        allow_permission_error (bool): Whether to allow PermissionError when creating working dir.
    """
    # Create working dir if it does not exist yet
    if config.get("work_dir"):
        try:
            os.makedirs(name=config.get("work_dir"), exist_ok=True)
        except PermissionError as e:
            if allow_permission_error:
                log.warning(f"PermissionError: {e}")
            else:
                raise e

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # set <config.trainer.fast_dev_run=True> if <config.debug=True>
    if config.get("debug_mode"):
        log.info("Running in debug mode! <config.debug_mode=True>")
        config.trainer.fast_dev_run = True
        os.environ["HYDRA_FULL_ERROR"] = "1"
        os.environ["OC_CAUSE"] = "1"
        torch.autograd.set_detect_anomaly(True)

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("devices"):
            config.trainer.devices = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0
    elif config.datamodule.get("num_workers") == -1:
        # set num_workers to #CPU cores if <config.datamodule.num_workers=-1>
        config.datamodule.num_workers = os.cpu_count()
        log.info(f"Automatically setting num_workers to {config.datamodule.num_workers} (CPU cores).")

    # force multi-gpu friendly configuration if <config.trainer.accelerator=ddp>
    strategy = config.trainer.get("strategy", "")
    if strategy.startswith("ddp") or strategy.startswith("dp"):
        if config.datamodule.get("pin_memory"):
            log.info(f"Forcing ddp friendly configuration! <config.trainer.strategy={strategy}>")
            config.datamodule.pin_memory = False

    torch_matmul_precision = config.get("torch_matmul_precision", "highest")
    if torch_matmul_precision != "highest":
        log.info(f"Setting torch matmul precision to ``{torch_matmul_precision}``.")
        torch.set_float32_matmul_precision(torch_matmul_precision)

    # Scale learning rate by effective batch size
    bs = config.datamodule.get("batch_size", 1)
    acc = config.trainer.get("accumulate_grad_batches", 1)
    n_gpus = config.trainer.get("devices", 1)
    if n_gpus == "auto":
        n_gpus = torch.cuda.device_count()
    elif isinstance(n_gpus, str) and "," in n_gpus:
        n_gpus = len(n_gpus.split(","))
    elif isinstance(n_gpus, Sequence):
        n_gpus = len(n_gpus)

    with open_dict(config):
        config.n_gpus = n_gpus
        config.effective_batch_size = bs * acc * n_gpus

    # Check if CUDA is available. If not, switch to CPU.
    if not torch.cuda.is_available():
        if config.trainer.get("accelerator") == "gpu":
            config.trainer.accelerator = "cpu"
            log.warning(
                "CUDA is not available, switching to CPU.\n"
                "\tIf you want to use GPU, please re-install pytorch: https://pytorch.org/get-started/locally/."
                "\n\tIf you want to use a different accelerator, specify it with ``trainer.accelerator=...``."
            )

    try:
        _ = config.datamodule.get("data_dir")
    except omegaconf.errors.InterpolationResolutionError as e:
        # Provide more helpful error message for e.g. Windows users where $HOME does not exist by default
        raise ValueError(
            "Could not resolve ``datamodule.data_dir`` in config. See error message above for details.\n"
            "   If this is a Windows machine, you may need to set ``data_dir`` to an absolute path, e.g. ``C:/data``.\n"
            "       You can do so in ``src/configs/datamodule/_base_data_config.yaml`` or with the command line."
        ) from e

    if config.module.get("num_predictions", 1) > 1:
        is_ipol_exp = "InterpolationExperiment" in config.module.get("_target_", "")
        monitor = config.module.get("monitor", "") or ""
        if "crps" not in monitor:
            new_monitor = f"val/{config.module.num_predictions}ens_mems/"
            new_monitor += "ipol/avg/crps" if is_ipol_exp else "avg/crps"
            config.module.monitor = new_monitor
            log.info(f"Setting monitor to {new_monitor} since num_predictions > 1")

    # fix monitor for model_checkpoint and early_stopping callbacks
    monitor = config.module.get("monitor", "") or ""
    if monitor:
        clbk_ckpt = config.callbacks.get("model_checkpoint", None)
        clbk_es = config.callbacks.get("early_stopping", None)
        if clbk_ckpt is not None and clbk_ckpt.get("monitor"):
            config.callbacks.model_checkpoint.monitor = monitor
        if clbk_es is not None and clbk_es.get("monitor"):
            config.callbacks.early_stopping.monitor = monitor

    # Set a short name for the model
    model_name = config.model.get("name")
    if model_name is None or model_name == "":
        model_class = config.model.get("_target_")
        mixer = config.model.mixer.get("_target_") if config.model.get("mixer") else None
        dm_type = config.datamodule.get("_target_")
        config.model.name = clean_name(model_class, mixer=mixer, dm_type=dm_type)

    USE_WANDB = "logger" in config.keys() and config.logger.get("wandb") and hasattr(config.logger.wandb, "_target_")
    if USE_WANDB:
        wandb_cfg = config.logger.wandb
        wandb_api.PROJECT = wandb_cfg.get("project", wandb_api.PROJECT)
        wandb_api._ENTITY = config.logger.wandb.entity = wandb_api.get_entity(wandb_cfg.get("entity"))
        log.info(f"Using wandb project {wandb_api.PROJECT} and entity {wandb_api._ENTITY}")

        if wandb_cfg.get("id"):
            wandb_status = "resume"
            log.info(f"Resuming experiment with wandb run ID = {config.logger.wandb.id}")
            run_api = get_run_api(run_id=wandb_cfg.id)
            # Set config wandb keys in case they were none, to the wandb defaults
            for k in wandb_cfg.keys():
                config.logger.wandb[k] = getattr(run_api, k) if hasattr(run_api, k) else wandb_cfg[k]
        else:
            if not wandb_cfg.get("group"):  # no wandb group has been assigned yet
                group_name = get_group_name(config)
                # potentially truncate the group name to 128 characters (W&B limit)
                if len(group_name) >= 128:
                    group_name = group_name.replace("-fcond", "").replace("DynTime", "DynT")
                    group_name = group_name.replace("UNetResNet", "UNetRN")

                if len(group_name) >= 128:
                    raise ValueError(f"Group name is too long, ({len(group_name)} > 128): {group_name}")
                config.logger.wandb.group = group_name
            group = config.logger.wandb.group

            if if_wandb_run_already_exists in ["abort", "resume"]:
                wandb_status = "new"
                runs_in_group = get_existing_wandb_group_runs(config, ckpt_must_exist=True, only_best_and_last=False)
                for other_run in runs_in_group:
                    other_seed = other_run.config.get("seed")
                    if "," in str(other_seed):
                        # wrong seed has been logged, use the one from the name
                        #  which goes along e.g. 'DDPM_h7-5T_UNet_64x3h_5e-5lr_<seed>seed_02h37m_on_Jan_20_gm5hmk7'
                        other_seed = other_run.name.split("seed_")[0].split("_")[-1]

                    if int(other_seed) != int(config.seed):
                        continue
                    # seeds are the same, so we treat this as a duplicate run
                    state = other_run.state
                    if if_wandb_run_already_exists == "abort":
                        raise RuntimeError(
                            f"Run with seed {config.seed} already exists in group {group}. State: {state}"
                        )
                    elif if_wandb_run_already_exists == "resume":
                        wandb_status = "resume"
                        config.logger.wandb.resume = True
                        config.logger.wandb.id = other_run.id
                        config.logger.wandb.name = other_run.name
                        log.info(
                            f"Resuming run {other_run.id} from group {group}. Seed={other_seed}; State was: ``{state}``"
                        )
                    else:
                        raise ValueError(f"if_wandb_run_already_exists={if_wandb_run_already_exists} is not supported")
                    break
            elif if_wandb_run_already_exists in [None, "ignore"]:
                wandb_status = "resume"
            else:
                wandb_status = "???"

            if config.logger.wandb.get("id") is None:
                # no wandb id has been assigned yet
                if "SLURM_JOB_ID" in os.environ:
                    # we are on a Slurm cluster... using the job ID helps when requeuing jobs to resume the same run
                    config.logger.wandb.id = os.environ["SLURM_JOB_ID"]
                else:
                    # we are not on a Slurm cluster, so just generate a random id
                    config.logger.wandb.id = wandb.sdk.lib.runid.generate_id()

            if not config.logger.wandb.get("name"):  # no wandb name has been assigned yet
                suffix = "_" + time.strftime("%Hh%Mm%b%d") + "_" + config.logger.wandb.id
                config.logger.wandb.name = get_detailed_name(config) + suffix

    elif if_wandb_run_already_exists in ["abort", "resume"]:
        wandb_status = "not_used"
        log.warning("Not checking if run already exists, since wandb logging is not being used")

    else:
        wandb_status = None

    if wandb_status == "resume":
        # Reload config from wandb
        run_path = f"{config.logger.wandb.entity}/{config.logger.wandb.project}/{config.logger.wandb.id}"
        override_config = get_only_overriden_config(config)
        config = wandb_api.load_hydra_config_from_wandb(run_path, override_config=override_config)

    check_config_values(config)

    # Init to wandb from rank 0 only in multi-gpu mode
    if USE_WANDB and int(os.environ.get("LOCAL_RANK", 0)) == 0 and os.environ.get("NODE_RANK", 0) == 0:
        # wandb_kwargs: dict = OmegaConf.to_container(config.logger.wandb, resolve=True)  # DictConfig -> simple dict
        wandb_kwargs = {
            k: config.logger.wandb.get(k)
            for k in ["id", "project", "entity", "name", "group", "tags", "notes", "reinit", "mode", "resume"]
        }
        wandb_kwargs["dir"] = config.logger.wandb.get("save_dir")
        wandb_kwargs["resume"] = wandb_kwargs.get("resume", "allow")
        try:
            wandb.init(**wandb_kwargs)
        except wandb.errors.UsageError as e:
            log.warning(" You need to login to wandb! Otherwise, choose a different/no logger with `logger=none`!")
            raise e
        # log.info(f"Wandb kwargs: {wandb_kwargs}")

    # Print config
    if config.get("print_config"):
        # pretty print config yaml -- requires rich package to be installed
        print_fields = ("model", "diffusion", "datamodule", "module", "trainer", "seed", "work_dir")  # or "all"
        print_config(config, fields=print_fields)
        # print early stopping config
        if config.get("callbacks") and config.callbacks.get("early_stopping"):
            patience = config.callbacks.early_stopping.get("patience")
            monitor = config.callbacks.early_stopping.get("monitor")
            mode = config.callbacks.early_stopping.get("mode")
            log.info(f"Early stopping: patience={patience}, monitor={monitor}, mode={mode}")

    # Save config to wandb
    if USE_WANDB:
        save_hydra_config_to_wandb(config)
    with open_dict(config):
        config.wandb_status = wandb_status

    return config


def get_only_overriden_config(config: DictConfig) -> DictConfig:
    """
    Get only the config values that are different from the default values in configs/main_config.yaml

    Args:
        config: Hydra config object with all the config values.

    Returns:
        DictConfig: Hydra config object with only the config values that are different from the default values.
    """
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with hydra.initialize(version_base=None, config_path="../configs"):
        config_default = hydra.compose(config_name="main_config.yaml", overrides=[])
    diff = get_difference_between_configs(config_default, config, one_sided=True)
    # Merge with explicit CLI args in case they happened to be equal to the default values.
    # This is needed because the default values may differ from the ones in a reloaded run config.
    cli_kwargs = OmegaConf.from_cli()
    diff = OmegaConf.merge(diff, cli_kwargs)
    return diff


def get_difference_between_configs(config1: DictConfig, config2: DictConfig, one_sided: bool = False) -> DictConfig:
    """
    Get the difference between two OmegaConf DictConfig objects (potentially use the values of config2).

    Args:
        config1: OmegaConf DictConfig object.
        config2: OmegaConf DictConfig object. Use the values of this config if they are different from config1.
        one_sided: If False, values of config1 are included if they don't exist in config2. If True, they are not.

    Returns:
        DictConfig: OmegaConf DictConfig object with only the config values that are different between config1 and config2.
            That is, values that are either contained in config1 but not config2, or vice versa, or have different values.
    """
    # We can convert the DictConfig to a simple dict, and then use set operations to get the difference
    # However, we need to resolve the DictConfig first, otherwise we get a TypeError
    config1 = OmegaConf.to_container(config1, resolve=True)
    config2 = OmegaConf.to_container(config2, resolve=True)
    # Get the difference between the two configs
    diff = get_difference_between_dicts_nested(config1, config2, one_sided=one_sided)
    # Convert back to DictConfig
    diff = OmegaConf.create(diff)
    return diff


def get_difference_between_dicts_nested(dict1: dict, dict2: dict, one_sided: bool = False) -> dict:
    """
    Get the difference between two nested dictionaries (potentially use the values of dict2).

    Args:
        dict1: Nested dictionary.
        dict2: Nested dictionary. Use the values of this dictionary if they are different from dict1.
        one_sided: If False, values of config1 are included if they don't exist in config2. If True, they are not.

    Returns:
        dict: Nested dictionary with only the values that are different between dict1 and dict2.
            That is, values that are either contained in dict1 but not dict2, or vice versa, or have different values.
    """
    if dict1 is None:
        return dict2
    if dict2 is None:
        return dict1
    # Get the difference between the two dicts
    if one_sided:
        diff = dict()
    else:
        diff = {k: dict1[k] for k in set(dict1) - set(dict2)}  # keys in dict1 but not dict2
    diff.update({k: dict2[k] for k in set(dict2) - set(dict1)})  # keys in dict2 but not dict1
    # Keys in both dicts but with different values (use the values of dict2)
    for k in set(dict1) & set(dict2):
        if dict1[k] != dict2[k]:
            # If the value is a dict, recursively get the difference between the nested dicts
            diff[k] = dict() if isinstance(dict2[k], dict) else dict2[k]
    # Recursively get the difference between the nested dicts
    for k in diff:
        if isinstance(diff[k], dict):
            diff[k] = get_difference_between_dicts_nested(dict1.get(k), dict2.get(k), one_sided=one_sided)
    return diff


def check_config_values(config: DictConfig):
    """Check if config values are valid."""
    with open_dict(config):
        if "net_normalization" in config.model.keys():
            if config.model.net_normalization is None:
                config.model.net_normalization = "none"
            config.model.net_normalization = config.model.net_normalization.lower()

        if config.get("diffusion", default_value=False):
            # Check that diffusion model has same hparams as the model it is based on
            for k, v in config.model.items():
                if k in config.diffusion.keys() and k not in ["_target_", "name"]:
                    assert v == config.diffusion[k], f"Diffusion model and model have different values for {k}!"

            ipolator_id = config.diffusion.get("interpolator_run_id")
            if ipolator_id is not None:
                get_run_api(ipolator_id)

        scheduler_cfg = config.module.get("scheduler")
        if scheduler_cfg and "LambdaWarmUpCosineScheduler" in scheduler_cfg._target_:
            # set base LR of optim to 1.0, since we will scale it by the warmup factor
            config.module.optimizer.lr = 1.0

        USE_WANDB = (
            "logger" in config.keys() and config.logger.get("wandb") and hasattr(config.logger.wandb, "_target_")
        )
        if USE_WANDB:
            if "callbacks" in config and config.callbacks.get("model_checkpoint"):
                wandb_model_run_id = config.logger.wandb.get("id")
                d = config.callbacks.model_checkpoint.dirpath
                if wandb_model_run_id is not None and wandb_model_run_id not in d:
                    # Save model checkpoints to special folder <ckpt-dir>/<wandb-run-id>/
                    new_dir = os.path.join(d, wandb_model_run_id)
                    config.callbacks.model_checkpoint.dirpath = new_dir
                    os.makedirs(new_dir, exist_ok=True)
                    log.info(f" Model checkpoints will be saved in: {os.path.abspath(new_dir)}")
        else:
            if config.get("callbacks") and "wandb" in config.callbacks:
                raise ValueError("You are trying to use wandb callbacks but you aren't using a wandb logger!")
            # log.warning("Model checkpoints will not be saved because you are not using wandb!")
            config.save_config_to_wandb = False

        if config.module.get("num_predictions", 1) > 1:
            # adapt the evaluation batch size to the number of predictions
            bs, ebs = config.datamodule.batch_size, config.datamodule.eval_batch_size
            if ebs >= bs:
                # reduce the eval batch size to account for the number of predictions
                config.datamodule.eval_batch_size = max(1, int(bs // math.sqrt(config.module.num_predictions)))
                log.info(
                    f"Reducing eval batch size from {ebs} to {config.datamodule.eval_batch_size} to match the number of predictions."
                )


def get_all_instantiable_hydra_modules(config, module_name: str):
    modules = []
    if module_name in config:
        for _, module_config in config[module_name].items():
            if module_config is not None and "_target_" in module_config:
                if "early_stopping" in module_config.get("_target_"):
                    diffusion = config.get("diffusion", default_value=False)
                    monitor = module_config.get("monitor", "")
                    # If diffusion model: Add _step to the early stopping callback key
                    if diffusion and "step" not in monitor and "epoch" not in monitor:
                        module_config.monitor += "_step"
                        print("*** Early stopping monitor changed to: ", module_config.monitor)
                        print("----------------------------------------\n" * 20)

                try:
                    modules.append(hydra.utils.instantiate(module_config))
                except omegaconf.errors.InterpolationResolutionError as e:
                    log.warning(f" Hydra could not instantiate {module_config} for module_name={module_name}")
                    raise e
                # except hydra.errors.InstantiationException:
                #     log.warning(f" Hydra had trouble instantiating {module_config} for module_name={module_name}")
                #     log.warning(
                #         " Make sure that you are logged in to wandb if you are using a wandb logger!"
                #         " I.e. call `wandb login` in your terminal. (or, e.g., use `logger=none`)"
                #     )
                #     modules.append(
                #         hydra.utils.instantiate(module_config, settings=wandb.Settings(start_method="fork"))
                #     )

    return modules


@rank_zero_only
def log_hyperparameters(
    config,
    model: pl.LightningModule,
    data_module: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.
    Credits go to: https://github.com/ashleve/lightning-hydra-template

    Additionally saves:
        - number of {total, trainable, non-trainable} model parameters
    """

    def copy_and_ignore_keys(dictionary, *keys_to_ignore):
        if dictionary is None:
            return None
        new_dict = dict()
        for k in dictionary.keys():
            if k not in keys_to_ignore:
                new_dict[k] = dictionary[k]
        return new_dict

    log_params = dict()
    log_params["start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if "seed" in config:
        log_params["seed"] = config["seed"]

    # Remove redundant keys or those that are not important to know after training -- feel free to edit this!
    log_params["datamodule"] = copy_and_ignore_keys(config["datamodule"])
    log_params["model"] = copy_and_ignore_keys(config["model"])
    log_params["exp"] = copy_and_ignore_keys(config["module"], "optimizer", "scheduler")
    log_params["trainer"] = copy_and_ignore_keys(config["trainer"])
    # encoder, optims, and scheduler as separate top-level key
    if "n_gpus" in config.keys():
        log_params["trainer"]["gpus"] = config["n_gpus"]
    log_params["optim"] = copy_and_ignore_keys(config["module"]["optimizer"])
    if "base_lr" in config.keys():
        log_params["optim"]["base_lr"] = config["base_lr"]
    if "effective_batch_size" in config.keys():
        log_params["optim"]["effective_batch_size"] = config["effective_batch_size"]
    if "diffusion" in config:
        log_params["diffusion"] = copy_and_ignore_keys(config["diffusion"])
    log_params["scheduler"] = copy_and_ignore_keys(config["module"].get("scheduler", None))
    # Add a clean name for the model, for easier reading (e.g. src.model.MLP.MLP -> MLP)
    model_class = config.model.get("_target_")
    mixer = config.model.mixer.get("_target_") if config.model.get("mixer") else None
    log_params["model/name_id"] = clean_name(model_class, mixer=mixer)

    if "callbacks" in config:
        skip_callbacks = ["summarize_best_val_metric", "learning_rate_logging"]
        for k, v in config["callbacks"].items():
            if k in skip_callbacks:
                continue
            elif k == "model_checkpoint":
                log_params[k] = copy_and_ignore_keys(v, "save_top_k")
            else:
                log_params[k] = copy_and_ignore_keys(v)

    # save number of model parameters
    log_params["model/params_total"] = sum(p.numel() for p in model.parameters())
    log_params["model/params_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_params["model/params_not_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    log_params["dirs/work_dir"] = config.get("work_dir")
    log_params["dirs/ckpt_dir"] = config.get("ckpt_dir")
    log_params["dirs/wandb_save_dir"] = (
        config.logger.wandb.get("save_dir") if (config.get("logger") and config.logger.get("wandb")) else None
    )

    # send hparams to all loggers (if any logger is used)
    if trainer.logger is not None:
        log.info("Logging hyperparameters to the PyTorch Lightning loggers.")
        trainer.logger.log_hyperparams(log_params)

        # disable logging any more hyperparameters for all loggers
        # this is just a trick to prevent trainer from logging hparams of model,
        # since we already did that above
        trainer.logger.log_hyperparams = no_op


@rank_zero_only
def save_hydra_config_to_wandb(config: DictConfig):
    # Save the config to the Wandb cloud (if wandb logging is enabled)
    if config.get("save_config_to_wandb"):
        filename = "hydra_config.yaml"
        # Check if ``filename`` already exists in wandb cloud. If so, append a version number to it.
        run_api = get_run_api(run_path=wandb.run.path)
        version = 2
        run_api_files = [f.name for f in run_api.files()]
        while filename in run_api_files:
            filename = f"hydra_config-v{version}.yaml"
            version += 1

        log.info(f"Config will be saved to wandb as {filename} and in wandb.run.dir: {os.path.abspath(wandb.run.dir)}")
        # files in wandb.run.dir folder get directly uploaded to wandb
        filepath = os.path.join(wandb.run.dir, filename)
        with open(filepath, "w") as fp:
            OmegaConf.save(config, f=fp.name, resolve=True)
        wandb.save(filename)
    else:
        log.info("Hydra config will NOT be saved to WandB.")


def get_config_from_hydra_compose_overrides(
    overrides: List[str],
    config_path: str = "../configs",
    config_name: str = "main_config.yaml",
) -> DictConfig:
    """
    Function to get a Hydra config manually based on a default config file and a list of override strings.
    This is an alternative to using hydra.main(..) and the command-line for overriding the default config.

    Args:
        overrides: A list of strings of the form "key=value" to override the default config with.
        config_path: Relative path to the folder where the default config file is located.
        config_name: Name of the default config file (.yaml ending).

    Returns:
        The resulting config object based on the default config file and the overrides.

    Examples:

    .. code-block:: python

        config = get_config_from_hydra_compose_overrides(overrides=['model=mlp', 'model.optimizer.lr=0.001'])
        print(f"Lr={config.model.optimizer.lr}, MLP hidden_dims={config.model.hidden_dims}")
    """
    from hydra.core.global_hydra import GlobalHydra

    overrides = list(set(overrides))
    if "-m" in overrides:
        overrides.remove("-m")  # if multiruns flags are mistakenly in overrides
    # Not true?!: log.info(f" Initializing Hydra from {os.path.abspath(config_path)}/{config_name}")
    GlobalHydra.instance().clear()  # clear any previous hydra config
    hydra.initialize(config_path=config_path, version_base=None)
    try:
        config = hydra.compose(config_name=config_name, overrides=overrides)
    finally:
        GlobalHydra.instance().clear()  # always clean up global hydra
    return config


def get_model_from_hydra_compose_overrides(overrides: List[str]):
    """
    Function to get a torch model manually based on a default config file and a list of override strings.

    Args:
        overrides: A list of strings of the form "key=value" to override the default config with.

    Returns:
        The model instantiated from the resulting config.

    Examples:

    .. code-block:: python

        mlp_model = get_model_from_hydra_compose_overrides(overrides=['model=mlp'])
        random_mlp_input = torch.randn(1, 100)
        random_prediction = mlp_model(random_mlp_input)
    """
    from src.interface import get_lightning_module

    cfg = get_config_from_hydra_compose_overrides(overrides)
    return get_lightning_module(cfg)
