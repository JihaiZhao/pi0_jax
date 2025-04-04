import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import optax
import tqdm_loggable.auto as tqdm
import wandb
import os
import gc
import numpy as np
import matplotlib.pyplot as plt

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        # If we are resuming, we don't create a new TrainState here; we'll restore it from checkpoint
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Compute forward/backward pass and update the model parameters."""
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        # The model internally does forward pass and returns chunked_loss
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels (for grad_norm log)
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def plot_action_scatter(gt_actions: np.ndarray, pred_actions: np.ndarray, step: int, plot_dir: str):
    """Create scatter plots comparing ground truth and predicted actions."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # We'll just visualize the first sample in the batch
    gt_sample = gt_actions[0].reshape(-1, gt_actions.shape[-1])
    pred_sample = pred_actions[0].reshape(-1, pred_actions.shape[-1])

    # x-y scatter plot
    axs[0].scatter(gt_sample[:, 0], gt_sample[:, 1], label='Ground Truth', alpha=0.5)
    axs[0].scatter(pred_sample[:, 0], pred_sample[:, 1], label='Prediction', alpha=0.5)
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].legend()
    axs[0].set_title(f'x-y Scatter Plot (Step {step})')

    # y-z scatter plot
    axs[1].scatter(gt_sample[:, 1], gt_sample[:, 2], label='Ground Truth', alpha=0.5)
    axs[1].scatter(pred_sample[:, 1], pred_sample[:, 2], label='Prediction', alpha=0.5)
    axs[1].set_xlabel('y')
    axs[1].set_ylabel('z')
    axs[1].legend()
    axs[1].set_title(f'y-z Scatter Plot (Step {step})')

    # z-x scatter plot
    axs[2].scatter(gt_sample[:, 2], gt_sample[:, 0], label='Ground Truth', alpha=0.5)
    axs[2].scatter(pred_sample[:, 2], pred_sample[:, 0], label='Prediction', alpha=0.5)
    axs[2].set_xlabel('z')
    axs[2].set_ylabel('x')
    axs[2].legend()
    axs[2].set_title(f'z-x Scatter Plot (Step {step})')

    plt.tight_layout()
    outpath = os.path.join(plot_dir, f"action_scatter_plot_{step}.jpg")
    plt.savefig(outpath)
    wandb.log({"action_scatter_plots": wandb.Image(fig)}, step=step)
    plt.close(fig)


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    # Create an FSDP mesh if needed
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize checkpoint manager
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Create data loader
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        num_workers=config.num_workers,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    first_batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(first_batch)}")

    # Build or restore train state
    train_state_or_shape, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)

    if resuming:
        # If resuming, train_state_or_shape is the shape; we now restore full state:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state_or_shape, data_loader)
    else:
        train_state = train_state_or_shape

    jax.block_until_ready(train_state)
    logging.info(f"Train state initialized or restored:\n{training_utils.array_tree_to_info(train_state.params)}")

    # -------------------------------------------------------
    #  Build the Pi0-FAST output transform to convert tokens -> real actions
    # -------------------------------------------------------
    #
    # By following the same logic as policy inference, we replicate the pipeline:
    #   model_transforms.outputs (which includes ExtractFASTActions) +
    #   Unnormalize(...) +
    #   data_transforms.outputs +
    #   repack_transforms.outputs
    #
    # So that we get *the same final continuous actions* as if we used policy.infer().
    #
    data_config = config.data.create(config.assets_dirs, config.model)  # fetch norm_stats, transforms, etc.

    # Gather them:
    # model_transforms.outputs might have ExtractFASTActions for pi0-fast
    # If you prefer to replicate exactly the "policy_config.py" pipeline, do:
    output_tfm_list = []
    # 1) e.g. model transforms for pi0-fast (token->action)
    output_tfm_list.extend(data_config.model_transforms.outputs)
    # 2) unnormalize if we have norm_stats
    output_tfm_list.append(_transforms.Unnormalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm))
    # 3) data transforms outputs
    output_tfm_list.extend(data_config.data_transforms.outputs)
    # 4) repack transforms outputs (often empty, but let's be consistent)
    output_tfm_list.extend(data_config.repack_transforms.outputs)

    # Compose into one transform
    final_output_transform = _transforms.compose(output_tfm_list)

    # -------------------------------------------------------
    #  Compile the train step
    # -------------------------------------------------------
    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    # Make a dir for plots
    plot_dir = os.path.join(config.checkpoint_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Start training loop
    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    # We'll reuse rng for each step
    info_buffer = []
    batch = first_batch

    for step in pbar:
        with sharding.set_mesh(mesh):
            # 1) run training step
            train_state, info = ptrain_step(train_rng, train_state, batch)

        info_buffer.append(info)

        # 2) logging
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(info_buffer)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            info_buffer = []

        # 3) plotting
        # For pi0-fast, we can't directly use `model.sample_actions` result as real actions;
        # we must pass it through transforms.
        if step % config.plot_interval == 0:
            with sharding.set_mesh(mesh):
                # We'll disable jit for sample+plot to avoid big recompiles
                with jax.disable_jit():
                    # 3.1) re-merge model
                    model = nnx.merge(train_state.model_def, train_state.params)
                    observation, gt_actions = batch

                    # Let's just visualize the first item in the batch
                    plot_observation = jax.tree_map(lambda x: x[:1], observation)
                    plot_gt_actions = jax.tree_map(lambda x: x[:1], gt_actions)

                    # 3.2) get predicted tokens from sample_actions
                    #     ( Pi0FAST sample_actions returns tokens, not real actions )
                    pred_tokens = model.sample_actions(
                        rng=train_rng,
                        observation=plot_observation,
                    )

                    # 3.3) build a small dictionary to feed into final_output_transform
                    print(f"plot_observation fields: {dataclasses.fields(plot_observation)}")
                    # print(f"state shape: {plot_observation.state.shape}")
                    print(f"state content: {plot_observation.state}")
                    out_dict = {
                        "state": plot_observation.state,   
                        "actions": pred_tokens,            # tokens in "actions" key
                    }
                    # 3.4) Convert JAX-> NumPy so transforms can run on CPU
                    out_dict = jax.device_get(out_dict)

                    # 3.5) run output transform => real continuous actions
                    out_dict = final_output_transform(out_dict)
                    pred_actions_np = out_dict["actions"]  # shape [1, horizon, action_dim]

                    # 3.6) convert GT to np
                    gt_actions_np = jax.device_get(plot_gt_actions)

                    # 3.7) do scatter plot
                    plot_action_scatter(
                        gt_actions_np,
                        pred_actions_np,
                        step=step,
                        plot_dir=plot_dir
                    )

                    # optionally free memory
                    del model
                    del batch
                    del plot_observation
                    del plot_gt_actions
                    del pred_tokens
                    del pred_actions_np
                    del gt_actions_np
                    gc.collect()

        # 4) next batch
        batch = next(data_iter)

        # 5) checkpoint saving
        if (step % config.save_interval == 0 and step > start_step) or (step == config.num_train_steps - 1):
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish.")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    # Example usage:
    #   CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python train_FAST.py pi0_fast_droid --exp_name=whatever
    main(_config.cli())
