'''
Author: JihaiZhao jihai518@gmail.com
Date: 2025-03-27 18:23:51
LastEditors: JihaiZhao jihai518@gmail.com
LastEditTime: 2025-03-28 16:18:45
FilePath: /openpi/src/openpi/policies/xarm_single_policy.py
Description: 
'''
import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

def make_xarm_example() -> dict:
    """Creates a random input example for the Xarm policy."""
    return {
        "observation_images_d455": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        "observation_images_gopro": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),

        "observation_states_joint_angle": np.random.rand(7),        # joint angles
        "observation_states_gripper_position": np.random.rand(1),    # gripper position
    }

def _parse_image(image) -> np.ndarray:
    try:
        image = np.asarray(image)
        if np.issubdtype(image.dtype, np.floating):
            image = (255 * image).astype(np.uint8)
        if image.shape[0] == 3:
            image = einops.rearrange(image, "c h w -> h w c")
        return image
    except Exception as e:
        print(f"Error parsing image: {e}")
        print(f"Image type: {type(image)}")
        print(f"Image shape: {getattr(image, 'shape', None)}")
        raise


@dataclasses.dataclass(frozen=True)
class XarmInputs(transforms.DataTransformFn):
    """Transform raw dataset inputs into the format expected by the model."""
    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # Add debug print to see what's happening inside the transform
        print("\nInside XarmInputs transform:")
        print("Input data keys:", data.keys())
        
        # We only mask padding for pi0 model, not pi0-FAST
        mask_padding = self.model_type == _model.ModelType.PI0

        # Get state from raw inputs
        state_components = [
            data["observation_states_joint_angle"],
            data["observation_states_gripper_position"]
        ]
        state = np.concatenate(state_components)
        state = transforms.pad_to_dim(state, self.action_dim)
        
        # Process images
        base_image = _parse_image(data["observation_images_d455"])
        gopro_image = _parse_image(data["observation_images_gopro"])
        
        # Add debug print for image processing
        print("Base image shape:", base_image.shape if base_image is not None else None)
        print("GoPro image shape:", gopro_image.shape if gopro_image is not None else None)
        
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": gopro_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            }
        }

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "action" in data:
            action = np.asarray(data["action"])
            action = transforms.pad_to_dim(action, self.action_dim)
            inputs["action"] = action

        # Add debug print for output
        print("Output keys:", inputs.keys())
        return inputs


@dataclasses.dataclass(frozen=True)
class XarmOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][:, :8])}