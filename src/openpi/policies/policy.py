from collections.abc import Sequence
import logging
import pathlib
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        
        # # Debug: Log input before transformation
        # print("Input before transformation")
        # for key, value in inputs.items():
        #     if isinstance(value, (np.ndarray, jnp.ndarray)):
        #         print(f"  {key}: shape={value.shape}, min={np.min(value)}, max={np.max(value)}")
        #     # Check for nested image dictionaries
        #     elif isinstance(value, dict) and key == "image":
        #         print(f"  {key}: (dictionary with {len(value)} images)")
        #         for img_key, img_val in value.items():
        #             if isinstance(img_val, (np.ndarray, jnp.ndarray)):
        #                 print(f"    {img_key}: shape={img_val.shape}, min={np.min(img_val)}, max={np.max(img_val)}")
        
        # Apply input transforms
        inputs = self._input_transform(inputs)
        
        # # Debug: Log input after transformation
        # print("Input after transformation")
        # for key, value in inputs.items():
        #     if isinstance(value, (np.ndarray, jnp.ndarray)):
        #         print(f"  {key}: shape={value.shape}, min={np.min(value)}, max={np.max(value)}")
        #     elif key == "state":
        #         print(f"  {key}: value={value[:20]}")
        #     # Check for nested image dictionaries
        #     elif isinstance(value, dict) and key == "image":
        #         print(f"  {key}: (dictionary with {len(value)} images)")
        #         for img_key, img_val in value.items():
        #             if isinstance(img_val, (np.ndarray, jnp.ndarray)):
        #                 print(f"    {img_key}: shape={img_val.shape}, min={np.min(img_val)}, max={np.max(img_val)}")
        
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        # print("inputs.shape", inputs["state"].shape)

        self._rng, sample_rng = jax.random.split(self._rng)
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng, _model.Observation.from_dict(inputs), **self._sample_kwargs),
        }
        outputs["actions"] = outputs["actions"][..., :20]

        # # Print output statistics
        # print("Output statistics:")
        # for key, value in outputs.items():
        #     if isinstance(value, (np.ndarray, jnp.ndarray)):
        #         print(f"  {key}:")
        #         print(f"    shape: {value.shape}")
        #         print(f"    min: {np.min(value)}")
        #         print(f"    max: {np.max(value)}")
        
        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        # print("outputs actions value", outputs["actions"])
        
        # Apply output transforms
        final_outputs = self._output_transform(outputs)
        
        # # Debug: Log final outputs
        # print("Final outputs after transformation:")
        # for key, value in final_outputs.items():
        #     if isinstance(value, (np.ndarray, jnp.ndarray)):
        #         print(f"  {key}: shape={value.shape}, min={np.min(value)}, max={np.max(value)}")
        
        return final_outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
