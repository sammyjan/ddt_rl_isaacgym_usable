"""
Deployment script for using the exported ONNX policy model on actual robots.

This script demonstrates how to load and run the ONNX model with actual observations
from a robot or simulation.

Usage:
    python deploy_onnx_model.py --onnx_path <path_to_model.onnx>
"""

import argparse
import numpy as np
import torch

try:
    import onnxruntime as ort
    HAS_ONNX_RUNTIME = True
except ImportError:
    HAS_ONNX_RUNTIME = False
    print("Warning: onnxruntime not installed. Install with: pip install onnxruntime")


class OnnxPolicyDeployer:
    """
    Wrapper for deploying ONNX policy model on robots.
    
    This class handles:
    1. Loading the ONNX model
    2. Automatically detecting input/output dimensions
    3. Managing observation history
    4. Running inference
    5. Returning normalized actions
    """
    
    def __init__(self, onnx_path, use_cuda=False):
        """
        Initialize the ONNX policy deployer.
        
        Args:
            onnx_path: Path to the exported ONNX model file
            use_cuda: Whether to use CUDA for inference (if available)
        """
        if not HAS_ONNX_RUNTIME:
            raise RuntimeError("onnxruntime is required. Install with: pip install onnxruntime")
        
        # Create ONNX Runtime session
        providers = []
        if use_cuda:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Get input/output node info
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        
        self.input_names = [input.name for input in inputs]
        self.output_names = [output.name for output in outputs]
        
        # Auto-detect dimensions from input/output shapes
        # inputs[0]: nn_input0 (batch, prop_dim)
        # inputs[1]: nn_input1 (batch, num_hist, prop_dim)
        # outputs[0]: nn_output (batch, num_actions)
        
        self.prop_dim = None
        self.num_hist = None
        self.num_actions = None
        self.proprioceptive_dim_full = None
        
        # Extract from input/output shapes
        for inp in inputs:
            if inp.name == 'nn_input0':
                shape = inp.shape
                # Shape might be like [None, 33] or ['batch_size', 33]
                self.prop_dim = shape[-1] if isinstance(shape[-1], int) else 33
            elif inp.name == 'nn_input1':
                shape = inp.shape
                # Shape might be like [None, 10, 33] or ['batch_size', 10, 33]
                self.num_hist = shape[1] if len(shape) > 1 and isinstance(shape[1], int) else 10
                self.prop_dim = shape[-1] if isinstance(shape[-1], int) else 33
        
        for out in outputs:
            if out.name == 'nn_output':
                shape = out.shape
                # Shape might be like [None, 8] or ['batch_size', 8]
                self.num_actions = shape[-1] if isinstance(shape[-1], int) else 8
        
        # Set defaults if not detected
        if self.prop_dim is None:
            self.prop_dim = 33
        if self.num_hist is None:
            self.num_hist = 10
        if self.num_actions is None:
            self.num_actions = 8
        
        # Full proprioceptive dimension is prop_dim + 3 (missing velocity part)
        self.proprioceptive_dim_full = self.prop_dim + 3
        
        # Initialize observation history buffer
        # Shape: (1, num_hist, prop_dim)
        self.history_buffer = np.zeros((1, self.num_hist, self.prop_dim), dtype=np.float32)
        
        print(f"✓ ONNX model loaded successfully")
        print(f"  Detected dimensions:")
        print(f"    - proprioceptive (reduced): {self.prop_dim}D [3:{self.proprioceptive_dim_full}]")
        print(f"    - proprioceptive (full): {self.proprioceptive_dim_full}D")
        print(f"    - history length: {self.num_hist} timesteps")
        print(f"    - action dimension: {self.num_actions}D")
        print(f"  Inputs:  {self.input_names}")
        print(f"  Outputs: {self.output_names}")
        print(f"  Execution providers: {self.session.get_providers()}")
    
    def update_history(self, proprioceptive_obs):
        """
        Update the observation history buffer.
        
        Args:
            proprioceptive_obs: Current proprioceptive observation (full, prop_dim+3 array)
        
        Returns:
            None
        """
        if len(proprioceptive_obs) != self.proprioceptive_dim_full:
            raise ValueError(
                f"Expected proprioceptive observation of size {self.proprioceptive_dim_full}, "
                f"got {len(proprioceptive_obs)}"
            )
        
        # Extract reduced observation [3:]
        obs_reduced = proprioceptive_obs[3:]  # Shape: (prop_dim,)
        
        # Shift history: remove oldest, keep recent
        history_2d = self.history_buffer[0]  # (num_hist, prop_dim)
        
        # Roll: shift rows up, new observation added at bottom
        history_2d = np.roll(history_2d, -1, axis=0)
        history_2d[-1] = obs_reduced
        
        self.history_buffer[0] = history_2d
    
    def infer(self, current_proprioceptive_obs):
        """
        Run inference on the ONNX model.
        
        Args:
            current_proprioceptive_obs: Current proprioceptive observation (full prop_dim+3 array)
                                       Will be sliced to [3:] internally
        
        Returns:
            actions: Action commands as numpy array of shape (1, num_actions)
        """
        # Update history with current observation
        self.update_history(current_proprioceptive_obs)
        
        # Prepare inputs for ONNX model
        # nn_input0: current obs [3:] (prop_dim)
        current_obs_reduced = current_proprioceptive_obs[3:].astype(np.float32)
        current_obs_input = np.expand_dims(current_obs_reduced, axis=0)  # (1, prop_dim)
        
        # nn_input1: history (1, num_hist, prop_dim)
        history_input = self.history_buffer.astype(np.float32)  # (1, num_hist, prop_dim)
        
        # Run inference
        inputs = {
            self.input_names[0]: current_obs_input,
            self.input_names[1]: history_input
        }
        
        outputs = self.session.run(None, inputs)
        
        # Output is actions (1, num_actions)
        actions = outputs[0]
        
        return actions
    
    def reset_history(self):
        """Reset the observation history buffer."""
        self.history_buffer = np.zeros((1, self.num_hist, self.prop_dim), dtype=np.float32)
        print("✓ History buffer reset")


def demo_inference():
    """
    Demonstration of using the ONNX policy in a loop.
    """
    print("\n" + "="*60)
    print("ONNX Policy Deployment Demo")
    print("="*60)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Deploy ONNX policy model')
    parser.add_argument('--onnx_path', type=str, required=True,
                        help='Path to the ONNX model file')
    parser.add_argument('--use_cuda', action='store_true',
                        help='Use CUDA for inference if available')
    args = parser.parse_args()
    
    # Initialize deployer
    deployer = OnnxPolicyDeployer(args.onnx_path, use_cuda=args.use_cuda)
    
    # Simulate 100 steps of inference
    print("\nRunning 100 inference steps...")
    print("-"*60)
    
    for step in range(100):
        # Generate random proprioceptive observation (normally from robot/sensor)
        # Dimension auto-detected from ONNX model
        current_obs = np.random.randn(deployer.proprioceptive_dim_full).astype(np.float32)
        
        # Run inference
        actions = deployer.infer(current_obs)
        
        # Actions should be in range [-1, 1] (normalized motor commands)
        actions_clamped = np.clip(actions, -1.0, 1.0)
        
        if (step + 1) % 20 == 0:
            print(f"Step {step+1:3d}: obs_mean={current_obs.mean():.4f}, "
                  f"actions_mean={actions_clamped.mean():.4f}, "
                  f"actions_std={actions_clamped.std():.4f}")
    
    print("-"*60)
    print("✓ Demo completed successfully!")
    
    # Show usage pattern
    print("\n" + "="*60)
    print("Usage Pattern")
    print("="*60)
    print(f"""
# Initialize the deployer once
deployer = OnnxPolicyDeployer('path/to/policy.onnx', use_cuda=True)

# Detected configuration:
#   - Proprioceptive dimension: {deployer.proprioceptive_dim_full}D
#   - History timesteps: {deployer.num_hist}
#   - Action dimension: {deployer.num_actions}D

# In your control loop:
while robot_is_running:
    # Get observation from robot/sim ({deployer.proprioceptive_dim_full}D proprioceptive data)
    current_obs = get_robot_observation()  # shape: ({deployer.proprioceptive_dim_full},)
    
    # Run inference (history managed automatically)
    actions = deployer.infer(current_obs)  # shape: (1, {deployer.num_actions})
    
    # Apply to robot
    apply_actions_to_robot(actions[0])  # Extract batch dimension
    
    # History is automatically maintained internally
    """)
    print("="*60 + "\n")


if __name__ == '__main__':
    demo_inference()
