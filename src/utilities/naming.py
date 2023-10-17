from typing import Optional

from omegaconf import DictConfig


def get_name_for_hydra_config_class(config: DictConfig) -> Optional[str]:
    """Will return a string that can describe the class of the (sub-)config."""
    if "name" in config and config.get("name") is not None:
        return config.get("name")
    elif "_target_" in config:
        return config._target_.split(".")[-1]
    return None


def get_clean_float_name(lr: float) -> str:
    """Stringify floats <1 into very short format (use for learning rates, weight-decay etc.)"""
    # basically, map Ae-B to AB (if lr<1e-5, else map 0.0001 to 1e-4)
    # convert first to scientific notation:
    if lr >= 0.1:
        return str(lr)
    lr_e = f"{lr:.1e}"  # 1e-2 -> 1.0e-02, 0.03 -> 3.0e-02
    # now, split at the e into the mantissa and the exponent
    lr_a, lr_b = lr_e.split("e-")
    # if the decimal point is 0 (e.g 1.0, 3.0, ...), we return a simple string
    if lr_a[-1] == "0":
        return f"{lr_a[0]}{int(lr_b)}"
    else:
        return str(lr).replace("e-", "")


def remove_float_prefix(string, prefix_name: str = "lr", separator="_") -> str:
    # Remove the lr and/or wd substrings like:
    # 0.0003lr_0.01wd -> ''
    # 0.0003lr -> ''
    # 0.0003lr_0.5lrecs_0.01wd -> '0.5lrecs'
    # 0.0003lr_0.5lrecs -> '0.5lrecs'
    # 0.0003lr_0.5lrecs_0.01wd_0.5lrecs -> '0.5lrecs_0.5lrecs'
    if prefix_name not in string:
        return string
    part1, part2 = string.split(prefix_name)
    # split at '_' and keep all but the last part
    part1keep = "_".join(part1.split(separator)[:-1])
    return part1keep + part2


def get_detailed_name(config, add_seed: bool = True) -> str:
    """This is a detailed name for naming the runs for logging."""
    s = config.get("name") + "_" if config.get("name") is not None else ""
    hor = config.datamodule.get("horizon", 1)
    if (
        hor > 1
        and f"H{hor}" not in s
        and f"horizon{hor}" not in s.lower()
        and f"h{hor}" not in s.lower()
        and f"{hor}h" not in s.lower()
    ):
        print(f"WARNING: horizon {hor} not in name, but should be!", s, config.get("name_suffix"))
        s = s[:-1] + f"-MH{hor}_"

    s += config.get("name_suffix") + "_" if config.get("name_suffix") is not None else ""
    kwargs = dict(mixer=config.model.mixer._target_) if config.model.get("mixer") else dict()
    s += clean_name(config.model._target_, **kwargs) + "_"

    w = config.datamodule.get("window", 1)
    if w > 1:
        s += f"{w}w_"

    if config.datamodule.get("train_start_date") is not None:
        s += f"{config.datamodule.train_start_date}tst_"

    use_ema, ema_decay = config.module.get("use_ema", False), config.module.get("ema_decay", 0.9999)
    if use_ema:
        s += "EMA_"
        if ema_decay != 0.9999:
            s = s.replace("EMA", f"EMA{config.module.ema_decay}")

    is_diffusion = config.get("diffusion") is not None
    if is_diffusion:
        if config.diffusion.get("interpolator_run_id"):
            replace = {
                "01H8DH3DHFA49S8KVE9PAXWVTX": "v1",
                "01H98TQ31SWZZD0AS49YB05YR2": "v2",
            }
            i_id = replace.get(config.diffusion.interpolator_run_id, config.diffusion.interpolator_run_id)
            s += f"{i_id}-ipolID_"

        default = "linear"
        if config.diffusion.get("beta_schedule", default) != default:
            s += f"{config.diffusion.beta_schedule}_"

        if config.get("sampling_timesteps") and config.diffusion.get("timesteps") != config.get("sampling_timesteps"):
            if f"{config.diffusion.timesteps}T" not in s:
                s += f"{config.diffusion.timesteps}T_"
            s += f"{config.diffusion.sampling_timesteps}sT_"

        extra1 = config.diffusion.get("additional_interpolation_steps", 0)
        extra2 = config.diffusion.get("additional_interpolation_steps_factor", 0)
        if config.diffusion.get("schedule") == "linear":
            if extra2 > 0:
                # additional_steps = config.diffusion.additional_interpolation_steps_factor * (config.datamodule.horizon - 2)
                if config.diffusion.get("interpolate_before_t1", False):
                    s += f"{extra2}k-Xa_"
                else:
                    s += f"{extra2}k-Xb_"
        elif config.diffusion.get("schedule") == "before_t1_only":
            if extra1 > 0:
                s += f"{extra1}k-preT1_"

        fcond = config.diffusion.get("forward_conditioning", "data")
        if fcond != "data":
            s += f"{fcond}-fcond_" if "noise" not in fcond else f"{fcond}_"

        if config.diffusion.get("time_encoding", "discrete") != "discrete":
            tenc = config.diffusion.get("time_encoding")
            if tenc == "normalized":
                s += "01Time_"
            elif tenc == "dynamics":
                s += "DynT_"
            else:
                s += f"{config.diffusion.time_encoding}-timeEnc_"

    hdims = config.model.get("hidden_dims")
    if hdims is None:
        num_L = config.model.get("num_layers") or config.model.get("depth")
        if num_L is None:
            dim_mults = config.model.get("dim_mults")
            if dim_mults is None:
                pass
            elif tuple(dim_mults) == (1, 2, 4):
                num_L = "3"
            else:
                num_L = "-".join([str(d) for d in dim_mults])
        hdim = config.model.get("hidden_dim") or config.model.get("dim") or config.model.get("embed_dim")
        if hdim is not None:
            hdims = f"{hdim}x{num_L}" if num_L is not None else f"{hdim}"
    elif all([h == hdims[0] for h in hdims]):
        hdims = f"{hdims[0]}x{len(hdims)}"
    else:
        hdims = str(hdims)

    s += f"_{hdims}d_"
    if config.model.get("mlp_ratio", 2.0) != 2.0:
        s += f"{config.model.mlp_ratio}dxMLP_"

    loss = config.model.get("loss_function")
    if isinstance(loss, str):
        loss = loss.lower()
    else:
        loss = loss.get("_target_").split(".")[-1].lower().replace("loss", "")
    if is_diffusion and loss != "l1" or (not is_diffusion and loss != "mse"):
        s += f"{loss.upper()}_"
    if config.model.get("patch_size") is not None:
        p = config.model.patch_size
        p1, p2 = p if isinstance(p, (list, tuple)) else (p, p)
        s += f"{p1}x{p2}patch_"
    time_emb = config.model.get("with_time_emb", False)
    if time_emb not in [False, True, "scale_shift"]:
        s += f"{time_emb}_"
    if (isinstance(time_emb, str) and "scale_shift" in time_emb) and not config.model.get(
        "time_scale_shift_before_filter"
    ):
        s += "tSSA_"  # time scale shift after filter

    optim = config.module.get("optimizer")
    if optim is not None:
        if "adamw" not in optim.name.lower():
            s += f"{optim.name.replace('Fused', '').replace('fused', '')}_"
        if "fused" in optim.name.lower() or optim.get("fused", False):
            s = s[:-1] + "F_"
    scheduler_cfg = config.module.get("scheduler")
    if scheduler_cfg is not None and ("lr_max" in scheduler_cfg or "lr_start" in scheduler_cfg):
        lr_start = get_clean_float_name(scheduler_cfg.get("lr_start", 0))
        lr_max = get_clean_float_name(scheduler_cfg.get("lr_max", 0))
        lr_min = get_clean_float_name(scheduler_cfg.get("lr_min", 0))
        wup_steps = scheduler_cfg.get("warm_up_steps", 0)
        s += f"{lr_start}-{lr_max}-{lr_min}lr_" if wup_steps > 0 else f"{lr_max}-{lr_min}lr_"
        if wup_steps != 500:
            s = s[:-1]
            s += f"{scheduler_cfg.warm_up_steps / 100}Kw_"
        if scheduler_cfg.get("max_decay_steps", 1000) != 1000:
            s = s[:-1]
            s += f"{scheduler_cfg.max_decay_steps / 100}Kd_"
    else:
        lr = config.get("base_lr") or optim.get("lr")
        s += f"{get_clean_float_name(lr)}lr_"

    if is_diffusion:
        lam1 = config.diffusion.get("lambda_reconstruction")
        lam2 = config.diffusion.get("lambda_reconstruction2")
        nonzero_lams = len([1 for lam in [lam1, lam2] if lam is not None and lam > 0])
        uniform_lams = [1 / nonzero_lams if nonzero_lams > 0 else 0, 0.33 if nonzero_lams == 3 else 0]
        if config.diffusion.get("detach_interpolated_data", False):
            s += "detXi_"
        if config.diffusion.get("lambda_reconstruction2", 0) > 0:
            if lam1 == lam2:
                s += f"{lam1}lRecs_"
            else:
                s += f"{lam1}-{lam2}lRecs_"

        elif lam1 is not None and lam1 not in uniform_lams:
            s += f"{lam1}lRec_"

    d1 = config.model.get("dropout", 0)
    d2, d3 = config.model.get("attn_dropout", 0), config.model.get("block_dropout", 0)
    dinput, d3a = config.model.get("input_dropout", 0), config.model.get("block_dropout1", 0)
    dposemb = config.model.get("pos_emb_dropout", 0)
    any_nonzero = d1 > 0 or d2 > 0 or d3 > 0 or dinput > 0 or d3a > 0 or dposemb > 0
    if dinput > 0:
        s += f"{int(dinput * 100)}inDr_"  # input dropout
    if d1 > 0:
        s += f"{int(d1 * 100)}Dr_"
    if dposemb > 0:
        s += f"{int(dposemb * 100)}posDr_"
    if d2 > 0:
        s += f"{int(d2 * 100)}atDr_"
    if d3 > 0 and d3a > 0:
        s += f"{int(d3a * 100)}-{int(d3 * 100)}bDr_"  # block dropout
    elif d3 > 0:
        s += f"{int(d3 * 100)}bDr_"  # block dropout
    elif d3a > 0:
        s += f"{int(d3a * 100)}b1Dr_"
    if any_nonzero:  # remove redundant 'Dr_'   (should be done for all dropout later on -->todo)
        s = s.replace("Dr_", "-")
        # replace last '-' with 'Dr_'
        s = s[:-1] + "Dr_"

    if any_nonzero and is_diffusion and config.diffusion.get("enable_interpolator_dropout", False):
        s += "iDr_"  # interpolator dropout

    if config.module.optimizer.get("weight_decay") and config.module.optimizer.get("weight_decay") > 0:
        s += f"{get_clean_float_name(config.module.optimizer.get('weight_decay'))}wd_"
    if scheduler_cfg is not None and "CosineAnnealingLR" in scheduler_cfg.get("_target_", ""):
        s += "cos_"

    if config.get("suffix", "") != "":
        s += f"{config.get('suffix')}_"

    if add_seed:
        s += f"{config.get('seed')}seed"
    return s.replace("None", "").rstrip("_-").lstrip("_-")


def clean_name(class_name, mixer=None, dm_type=None) -> str:
    """This names the model class paths with a more concise name."""
    if "AFNONet" in class_name or "Transformer" in class_name:
        if mixer is None or "AFNO" in mixer:
            s = "AFNO"
        elif "SelfAttention" in mixer:
            s = "self-attention"
        else:
            raise ValueError(class_name)
    elif "SphericalFourierNeuralOperatorNet" in class_name:
        return "SFNO"
    elif "UnetConvNext" in class_name:
        s = "UnetConvNext"
    elif "unet_simple" in class_name:
        s = "SimpleUnet"
    elif "AutoencoderKL" in class_name:
        s = "LDM"
    elif "SimpleChannelOnlyMLP" in class_name:
        s = "SiMLP"
    elif "MLP" in class_name:
        s = "MLP"
    elif "Unet" in class_name:
        s = "UNetR"
    elif "SimpleConvNet" in class_name:
        s = "SimpleCNN"
    elif "graph_network" in class_name:
        s = "GraphNet"
    elif "CNN_Net" in class_name:
        s = "CNN"
    elif "NCSN" in class_name:
        s = "NCSN"
    else:
        raise ValueError(f"Unknown class name: {class_name}, did you forget to add it to the clean_name function?")

    return s.lstrip("_")


def get_group_name(config) -> str:
    """
    This is a group name for wandb logging.
    On Wandb, the runs of the same group are averaged out when selecting grouping by `group`
    """
    # s = get_name_for_hydra_config_class(config.model)
    # s = s or _shared_prefix(config, init_prefix=s)
    return get_detailed_name(config, add_seed=False)


def clean_metric_name(metric: str) -> str:
    """This is a clean name for the metrics (e.g. for plotting)"""
    metric_dict = {
        "mae": "MAE",
        "mse": "MSE",
        "crps": "CRPS",
        "rmse": "RMSE",
        "mape": "MAPE",
        "ssr": "Spread / RMSE",
        "nll": "NLL",
        "r2": "R2",
        "corr": "Correlation",
        "corrcoef": "Correlation",
        "corr_spearman": "Spearman Correlation",
        "corr_pearson": "Pearson Correlation",
    }
    return metric_dict.get(metric.lower(), metric)
