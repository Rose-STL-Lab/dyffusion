# Using the code-base
**IMPORTANT NOTE:** 
All commands in this README assume that you are in the [root of the repository](../) (and need to be run from there)!

## Train a model
Run the [run.py](../run.py) script from the repository root. 
For example, running the following on the command line

    python run.py trainer.accelerator=gpu model=unet_resnet logger=none callbacks=default

will train a UNet on the GPU using some default callbacks and hyperparameters, but no logging.

## Running experiments
It is recommended to define all the parameters of your experiment 
in a [YAML](https://yaml.org/) file inside the [configs/experiment](configs/experiment) folder.

For example, the config file [spring_mesh_mh_time_conditioned](configs/experiment/spring_mesh_time_conditioned.yaml) defines
the experiment to train a bare-bone Dropout CNN on the spring mesh dataset with some particular (hyper-)parameters.
You can then easily run such an experiment with the following command:

    python run.py experiment=spring_mesh_mh_time_conditioned 

## Resume training from a wandb run
If you want to resume training from a previous run, you can use the following command:

    python run.py logger.wandb.id=<run_id>  

where `<run_id>` is the wandb ID of the run you want to resume training.  
You can add any extra arguments, e.g. ``datamodule.num_workers=8``, to change the values from the previous run.
Note that if you want to run for more epoch, you need to add ``trainer.max_epochs=<new_max_epochs>``.

## Important Training Arguments and Hyperparameters
- To run on CPU use ``trainer.devices=0``, to use a single GPU use ``trainer.devices=1``, etc.
- To override the data directory you can override the flag ``datamodule.data_dir=<data-dir>``.
- A random seed for reproducibility can be set with ``seed=<seed>`` (by default it is ``11``).

### Directories for logging and checkpoints
By default,
- the checkpoints (i.e. model weights) are saved in ``results/checkpoints/``,
- any logs are saved in ``results/logs/``.

To change the name of ``results/`` in both subdirs above, you may simply use the flag ``work_dir=YOUR-OUT-DIR``.
To only change the name of the checkpoints' directory, you may use the flag ``ckpt_dir=YOUR-CHECKPOINTS-DIR``.
To only change the name of the logs' directory, you may use the flag ``log_dir=YOUR-LOGS-DIR``.
All together, you could override all these dirs with

    python run.py work_dir=YOUR-OUT-DIR ckpt_dir=YOUR-CHECKPOINTS-DIR log_dir=YOUR-LOGS-DIR

### Data parameters and structure
#### General data-specific parameters
Important data-specific parameters can be all found in the 
[configs/datamodule/base_data_config](configs/datamodule/_base_data_config.yaml) file. 
In particular:
- ``datamodule.data_dir``: the directory where the data must be stored
- ``datamodule.batch_size``: the batch size to use for training.
- ``datamodule.eval_batch_size``: the batch size to use for validation and testing.
- ``datamodule.num_workers``: the number of workers to use for loading the data.

You can override any of these parameters by adding ``datamodule.<parameter>=<value>`` to the command line.

### ML model parameters and architecture

#### Define the architecture
To train a pre-defined model do ``model=<model_name>``, e.g. ``model=cnn_simple``, ``model=unet_resnet``, etc.,
    where [configs/model/](configs/model)<model_name>.yaml must be the configuration file for the respective model.

You can also override any model hyperparameter by adding ``model.<hyperparameter>=<value>`` to the command line.
E.g.:
- to change the number of layers and dimensions in an MLP you would use 
``model=mlp 'model.hidden_dims=[128, 128, 128]'`` (note that the parentheses are needed when the value is a list).

#### General parameters
Important training/evaluation parameters can be all found in the 
[configs/module/_base_model_config](configs/model/_base_experiment_config.yaml) file. 
In particular:
- ``module.scheduler``: the scheduler to use for the learning rate.
- ``module.monitor``: the logged metric to track for early-stopping, model-checkpointing and LR-scheduling.

### Wandb support
<details>
  <summary><b> Requirements & Logging in </b></summary>
The following requires you to have a wandb (team) account, and you need to log in with ``wandb login`` before you can use it.
You can also simply export the environment variable ``WANDB_API_KEY`` with your wandb API key, 
and the [run.py](../run.py) script will automatically log in for you.

</details>

- To log metrics to [wandb](https://wandb.ai/site) use ``logger=wandb``.
- To use some nice wandb specific callbacks in addition, use ``callbacks=wandb`` (e.g. save the best trained model to the wandb cloud).

## Tips

<details>
    <summary><b> hydra.errors.InstantiationException </b></summary>

The ``hydra.errors.InstantiationException`` itself is not very informative, 
so you need to look at the preceding exception(s) (i.e. scroll up) to see what went wrong.
</details>

<details>
    <summary><b> Overriding nested Hydra config groups </b></summary>

Nested config groups need to be overridden with a slash - not with a dot, since it would be interpreted as a string otherwise.
For example, if you want to change the optimizer, you should run:
``python run.py optimizer@module.optimizer=SGD``
</details>

<details>
  <summary><b> Local configurations </b></summary>

You can easily use a local config file that, defines the local data dir, working dir etc., by putting a ``default.yaml`` config 
in the [configs/local/](configs/local) subdirectory. Hydra searches for & uses by default the file configs/local/default.yaml, if it exists.
</details>

<details>
    <summary><b> Grouping wandb charts</b></summary>

If you use Wandb, make sure to select the "Group first prefix" option in the panel/workspace settings of the web app inside the project (in the top right corner).
This will make it easier to browse through the logged metrics.
</details>




