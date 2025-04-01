'''
Author: JihaiZhao jihai518@gmail.com
Date: 2025-03-27 18:23:51
LastEditors: JihaiZhao jihai518@gmail.com
LastEditTime: 2025-03-28 16:18:45
FilePath: /openpi/src/openpi/policies/xarm_dual_policy.py
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
        "observation_images_head": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        "observation_images_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        "observation_images_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),

        "observation_states_joint_angle_left": np.random.rand(7),        # joint angles
        "observation_states_joint_angle_right": np.random.rand(7),       # joint angles   
        "observation_states_gripper_position_left": np.random.rand(1),    # gripper position
        "observation_states_gripper_position_right": np.random.rand(1),   # gripper position
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
    pose: bool = False  # Add pose parameter
    _has_printed: bool = dataclasses.field(default=False, init=False)

    def __call__(self, data: dict) -> dict:
        # print("\nXarmInputs input keys:", list(data.keys()))
        mask_padding = self.model_type == _model.ModelType.PI0
        
        # Print state type message only once
        if not object.__getattribute__(self, '_has_printed'):
            print("Using pose as state" if self.pose else "Using joint angle as state")
            object.__setattr__(self, '_has_printed', True)
        
        if self.pose:
            # Get state from raw inputs using pose data
            state_components = [
                data["observation_states_ee_pose_left"],
                data["observation_states_gripper_position_left"],
                data["observation_states_ee_pose_right"],
                data["observation_states_gripper_position_right"]
            ]
            state = np.concatenate(state_components)
            # print(f"Original state shape: {state.shape}")  # Should be (..., 16)
            state = transforms.pad_to_dim(state, self.action_dim)
            # print(f"Padded state shape: {state.shape}")    # Should be (..., 32)
        else:
            # Get state from raw inputs using joint angles
            state_components = [
                data["observation_states_joint_angle_left"],
                data["observation_states_gripper_position_left"],
                data["observation_states_joint_angle_right"],
                data["observation_states_gripper_position_right"]
            ]
            state = np.concatenate(state_components)
            # print(f"Original state shape: {state.shape}")  # Should be (..., 16)
            state = transforms.pad_to_dim(state, self.action_dim)
            # print(f"Padded state shape: {state.shape}")    # Should be (..., 32)
        
        # Process images
        base_image = _parse_image(data["observation_images_head"])
        left_wrist_image = _parse_image(data["observation_images_left_wrist"])
        right_wrist_image = _parse_image(data["observation_images_right_wrist"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            }
        }

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "action_left" in data:
            action_left = np.asarray(data["action_left"])
            # print(f"Original action_left shape: {action_left.shape}")
            # action_left = transforms.pad_to_dim(action_left, self.action_dim)
            # print(f"Padded action_left shape: {action_left.shape}")
            inputs["action_left"] = action_left

        if "action_right" in data:
            action_right = np.asarray(data["action_right"])
            # print(f"Original action_right shape: {action_right.shape}")
            # action_right = transforms.pad_to_dim(action_right, self.action_dim)
            # print(f"Padded action_right shape: {action_right.shape}")
            inputs["action_right"] = action_right
        
        if "action" in data:
            action = np.asarray(data["action"])
            # print(f"Original action shape: {action.shape}")
            # action = transforms.pad_to_dim(action, self.action_dim)
            # print(f"Padded action shape: {action.shape}")
            inputs["action"] = action

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
        return {"actions": np.asarray(data["actions"][:, :20])}
