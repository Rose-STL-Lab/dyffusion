from omegaconf import DictConfig


def get_dims_of_dataset(datamodule_config: DictConfig):
    """Returns the number of features for the given dataset."""
    target = datamodule_config.get("_target_", datamodule_config.get("name"))
    conditional_dim = 0
    if "oisstv2" in target:
        box_size = datamodule_config.box_size
        input_dim, output_dim, spatial_dims = 1, 1, (box_size, box_size)
    elif "physical_systems_benchmark" in target:
        if datamodule_config.physical_system == "navier-stokes":
            input_dim, output_dim, spatial_dims = 3, 3, (221, 42)
            conditional_dim = 2
        elif datamodule_config.physical_system == "spring-mesh":
            input_dim, output_dim, spatial_dims = 4, 4, (10, 10)
            conditional_dim = 1
        else:
            raise ValueError(f"Unknown physical system: {datamodule_config.physical_system}")
    else:
        raise ValueError(f"Unknown dataset: {target}")
    return {"input": input_dim, "output": output_dim, "spatial": spatial_dims, "conditional": conditional_dim}
