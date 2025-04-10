'''
Author: JihaiZhao jihai518@gmail.com
Date: 2025-04-04 16:15:52
LastEditors: JihaiZhao jihai518@gmail.com
LastEditTime: 2025-04-08 12:46:54
FilePath: /pi0_jax/scripts/test.py
Description: 
'''
import dataclasses

import jax

from openpi.models import model as _model
from openpi.policies import droid_policy
from openpi.policies import xarm_dual_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.transforms import Group

# config = _config.get_config("pi0_fast_droid")
# checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_fast_droid")

# # Create a trained policy.
# policy = _policy_config.create_trained_policy(config, checkpoint_dir)

# # Run inference on a dummy example. This example corresponds to observations produced by the DROID runtime.
# example = droid_policy.make_droid_example()
# result = policy.infer(example)

# # Delete the policy to free up memory.
# del policy

# print("Result:", result)
# print("Actions shape:", result["actions"].shape)

config = _config.get_config("pi0_xarm_dual")
# checkpoint_dir = "/scratch/wty/pi0_jax/checkpoints/pi0_xarm_dual_fast/dual_joint_1_fast/33000"
checkpoint_dir = "/media/qtus/T7/checkpoints/pi0_jax/abs_pose/27000"

# Create a trained policy.
policy = _policy_config.create_trained_policy(config, 
                                              checkpoint_dir, 
                                              default_prompt="Pour the breans from the cup to the target, and place the cup on the holder")
# Run inference on a dummy example. This example corresponds to observations produced by the DROID runtime.
example = xarm_dual_policy.make_xarm_example()
result = policy.infer(example)

# Delete the policy to free up memory.
del policy

print("Result:", result)
print("Actions shape:", result["actions"].shape)