import dataclasses
import logging
import os
import jax
import numpy as np
import dora
from dora import Node
import pyarrow as pa

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

@dataclasses.dataclass
class Args:
    # Config and checkpoint parameters
    config_name: str = "pi0_xarm_dual"  # or get from env var
    ckpt_dir: str = "path/to/checkpoint"  # or get from env var

def main(args: Args):
    # Get config and checkpoint from environment variables or args
    config_name = str(os.getenv("CONFIG_NAME", args.config_name))
    ckpt_dir = str(os.getenv("CKPT_PATH", args.ckpt_dir))
    
    # Initialize policy
    config = _config.get_config(config_name)
    policy = _policy_config.create_trained_policy(
        config, 
        ckpt_dir,
        default_prompt=None
    )
    
    # Initialize JAX
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    
    logging.info(f"Initialized pi0 policy: {policy.metadata}")
    logging.basicConfig(level=logging.INFO)
    node = dora.Node()

    for event in node:
        if event["type"] == "INPUT" and event["id"] == "obs_dict":
            obs_dict = event["value"].to_numpy(zero_copy_only=False)[0]
            
            # Format observation to match XarmInputs expectations
            observation = {
                "observation_images_head": obs_dict["observation_images_head"],
                "observation_images_left_wrist": obs_dict["observation_images_left_wrist"],
                "observation_images_right_wrist": obs_dict["observation_images_right_wrist"],
                "observation_states_ee_pose_left": obs_dict["observation_states_ee_pose_left"],
                "observation_states_gripper_position_left": obs_dict["observation_states_gripper_position_left"],
                "observation_states_ee_pose_right": obs_dict["observation_states_ee_pose_right"],
                "observation_states_gripper_position_right": obs_dict["observation_states_gripper_position_right"],
                # Add prompt if using language instructions
                # "prompt": "your instruction here"  # optional
            }

            # Use local policy inference
            action_pred = policy.infer(observation)["actions"]
            print(f"[Policy Node] Action predicted: {action_pred.shape}")
            print(f"[Policy Node] Action predicted: {action_pred}")

            event["metadata"]["action_step"] = action_pred.shape[0]
            node.send_output(
                "action_pred",
                pa.array(action_pred.ravel()),
                metadata=event["metadata"],
            )

if __name__ == "__main__":
    main(Args())
