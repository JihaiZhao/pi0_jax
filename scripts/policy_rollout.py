#!/usr/bin/env python3
"""
Policy FastAPI Client

This script:
1. Connects to the FastAPI endpoint to get observation data
2. Runs the policy inference on that data
3. Optionally sends actions to another endpoint or system

Usage:
    python policy_fastapi_client.py [--host HOST] [--port PORT] [--interval INTERVAL]

Options:
    --host HOST           API server hostname or IP (default: localhost)
    --port PORT           API server port (default: 8000)
    --interval INTERVAL   Polling interval in seconds (default: 0.1)
    --ckpt_path CKPT_PATH Path to policy checkpoint
    --config CONFIG       Config name for policy (default: pi0_xarm_dual)
"""

import argparse
import dataclasses
import json
import logging
import os
import time
import sys
from typing import Dict, Any, Optional, List, Union, Tuple

import numpy as np
import requests
import jax

# Import policy-related modules
# You may need to adjust these imports based on your actual module structure
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


class PolicyFastAPIClient:
    def __init__(
        self, 
        host: str = "localhost", 
        port: int = 8000, 
        interval: float = 0.1,
        config_name: str = "pi0_xarm_dual",
        ckpt_path: str = None,
        action_endpoint: Optional[str] = None
    ):
        """
        Initialize the policy client.
        
        Args:
            host: API server hostname or IP for observation data
            port: API server port for observation data
            interval: Polling interval in seconds
            config_name: Policy configuration name
            ckpt_path: Path to policy checkpoint
            action_endpoint: Optional endpoint to send actions to
        """
        self.host = host
        self.port = port
        self.interval = interval
        self.base_url = f"http://{host}:{port}"
        self.last_observation = None
        self.session = requests.Session()
        self.config_name = config_name
        self.ckpt_path = ckpt_path
        self.action_endpoint = action_endpoint
        self.policy = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("PolicyFastAPIClient")
        
    def initialize_policy(self):
        """Initialize the policy with the given config and checkpoint."""
        if self.policy is not None:
            return
            
        self.logger.info(f"Initializing policy with config {self.config_name} from {self.ckpt_path}")
        
        # Initialize JAX
        jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
        
        # Initialize policy
        config = _config.get_config(self.config_name)
        self.policy = _policy_config.create_trained_policy(
            config, 
            self.ckpt_path,
            default_prompt=None
        )
        
        self.logger.info(f"Initialized policy: {self.policy.metadata}")
        
    def get_observation(self) -> Optional[Dict[str, Any]]:
        """
        Fetch the latest observation from the API server.
        
        Returns:
            The observation data dictionary or None if request failed
        """
        try:
            response = self.session.get(f"{self.base_url}/observation", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"Error: Received status code {response.status_code}")
                return None
        except requests.RequestException as e:
            self.logger.error(f"Request error: {e}")
            return None
            
    def check_connection(self) -> bool:
        """
        Check if the API server is available.
        
        Returns:
            True if server is available, False otherwise
        """
        try:
            response = self.session.get(f"{self.base_url}/", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def process_observation(self, obs_dict: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Process the observation through the policy.
        
        Args:
            obs_dict: The observation data dictionary
            
        Returns:
            The predicted action or None if processing failed
        """
        if not obs_dict or not self.policy:
            return None
            
        try:
            # Format observation to match policy expectations
            observation = {
                "observation_images_head": np.array(obs_dict["observation_images_head"]),
                "observation_images_left_wrist": np.array(obs_dict["observation_images_left_wrist"]),
                "observation_images_right_wrist": np.array(obs_dict["observation_images_right_wrist"]),
                "observation_states_ee_pose_left": np.array(obs_dict["observation_states_ee_pose_left"]),
                "observation_states_gripper_position_left": np.array(obs_dict["observation_states_gripper_position_left"]),
                "observation_states_ee_pose_right": np.array(obs_dict["observation_states_ee_pose_right"]),
                "observation_states_gripper_position_right": np.array(obs_dict["observation_states_gripper_position_right"]),
                # Add prompt if using language instructions
                # "prompt": "your instruction here"  # optional
            }

            # Run policy inference
            result = self.policy.infer(observation)
            action_pred = result["actions"]
            
            self.logger.info(f"Action predicted shape: {action_pred.shape}")
            self.logger.debug(f"Action predicted: {action_pred}")
            
            return action_pred
            
        except Exception as e:
            self.logger.error(f"Error processing observation: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
            
    def send_action(self, action: np.ndarray) -> bool:
        """
        Send the action to the configured endpoint (if any).
        
        Args:
            action: The action to send
            
        Returns:
            True if action was sent successfully, False otherwise
        """
        if not self.action_endpoint or action is None:
            return False
            
        try:
            # Convert action to serializable format
            action_data = {
                "action": action.tolist(),
                "timestamp": time.time()
            }
            
            # Send action to endpoint
            response = self.session.post(
                self.action_endpoint, 
                json=action_data, 
                timeout=5
            )
            
            return response.status_code == 200
            
        except requests.RequestException as e:
            self.logger.error(f"Error sending action: {e}")
            return False
    
    def run(self, max_iterations: Optional[int] = None) -> None:
        """
        Run the client, continuously polling for observations.
        
        Args:
            max_iterations: Maximum number of polling iterations, or None for infinite
        """
        # Initialize policy
        self.initialize_policy()
        
        if not self.check_connection():
            self.logger.error(f"Cannot connect to server at {self.base_url}")
            return
            
        self.logger.info(f"Connected to server at {self.base_url}")
        self.logger.info(f"Polling for observations every {self.interval} seconds...")
        self.logger.info("Press Ctrl+C to stop")
        
        iteration = 0
        try:
            while max_iterations is None or iteration < max_iterations:
                # Get observation
                observation = self.get_observation()
                
                # Process observation if different from last one
                if observation != self.last_observation and observation is not None:
                    # Process through policy
                    action = self.process_observation(observation)
                    
                    # Send action (if configured)
                    if action is not None:
                        if self.action_endpoint:
                            success = self.send_action(action)
                            if success:
                                self.logger.info(f"Action sent successfully")
                            else:
                                self.logger.warning(f"Failed to send action")
                        else:
                            # If no endpoint is configured, just print the action
                            self.logger.info(f"Action (not sent): shape={action.shape}")
                    
                    self.last_observation = observation
                
                time.sleep(self.interval)
                iteration += 1
                
        except KeyboardInterrupt:
            self.logger.info("Stopping client...")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Policy FastAPI Client")
    parser.add_argument("--host", default="localhost", help="API server hostname or IP")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    parser.add_argument("--interval", type=float, default=0.1, help="Polling interval in seconds")
    parser.add_argument("--ckpt_path", required=True, help="Path to policy checkpoint")
    parser.add_argument("--config", default="pi0_xarm_dual", help="Policy configuration name")
    parser.add_argument("--action_endpoint", help="Optional endpoint to send actions to")
    return parser.parse_args()


def main():
    """Main entry point of the script."""
    args = parse_args()
    
    # Create and run the client
    client = PolicyFastAPIClient(
        host=args.host,
        port=args.port,
        interval=args.interval,
        config_name=args.config,
        ckpt_path=args.ckpt_path,
        action_endpoint=args.action_endpoint
    )
    
    client.run()


if __name__ == "__main__":
    main() 