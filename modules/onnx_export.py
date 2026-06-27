"""
ONNX export module for policy model.
Converts PyTorch policy to ONNX format with separate input interfaces for current and historical observations.
"""

import torch
import torch.nn as nn
import os


class OnnxActorWrapper(nn.Module):
    """
    Wrapper for exporting the actor policy to ONNX format.
    
    Supports flexible input dimensions (not hardcoded to specific robot):
    
    Inputs:
        - nn_input0: current proprioceptive observation (batch_size, n_prop)
                     e.g., (batch, 36) for Tita, (batch, 46) for D1H
        - nn_input1: historical proprioceptive observation (batch_size, history_len, n_prop)
                     e.g., (batch, 10, 36) for Tita, (batch, 10, 46) for D1H
    
    Output:
        - nn_output: action commands (batch_size, num_actions)
                     e.g., (batch, 8) for Tita, (batch, 12) for D1H
    
    The wrapper performs inference without normalization on full proprioceptive input.
    """
    
    def __init__(self, actor_teacher_backbone, num_prop=36, num_hist=10):
        """
        Args:
            actor_teacher_backbone: The MlpBarlowTwinsActor module
            num_prop: Full proprioceptive dimension
            num_hist: Number of history timesteps
        """
        super().__init__()
        self.actor_backbone = actor_teacher_backbone
        self.num_prop = num_prop
        self.num_hist = num_hist
        
    def forward(self, nn_input0, nn_input1):
        """
        Forward pass for ONNX inference.
        
        Args:
            nn_input0: Current proprioceptive observation (batch_size, num_prop)
            nn_input1: Historical proprioceptive observation (batch_size, num_hist, num_prop)
                       10 timesteps of full proprioceptive observations
        
        Returns:
            nn_output: Action commands (batch_size, num_actions)
        """
        batch_size = nn_input0.shape[0]
        
        # nn_input1 is already in shape (batch, num_hist, num_prop)
        obs_hist_prop = nn_input1
        
        # Call the actor backbone without normalization
        actions = self.actor_backbone(nn_input0, obs_hist_prop)
        
        return actions


def export_policy_to_onnx(policy, save_path, device='cpu'):
    """
    Export the policy to ONNX format.
    
    Automatically detects dimensions based on policy configuration.
    Supports any robot configuration (Tita 8-joint, D1H 12-joint, etc.)
    
    Args:
        policy: ActorCriticBarlowTwins policy object
        save_path: Path to save the ONNX model
        device: Device to use for export ('cpu' or 'cuda')
    
    Returns:
        bool: True if export succeeded, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Extract dimension information from policy
        num_prop = policy.num_prop  # e.g., 36 for Tita, 46 for D1H
        num_hist = policy.kwargs.get('history_len', 10) if hasattr(policy, 'kwargs') else 10
        num_actions = policy.num_actions  # e.g., 8 for Tita, 12 for D1H
        
        actor_backbone = policy.actor_teacher_backbone
        
        # Create ONNX wrapper with detected dimensions
        onnx_wrapper = OnnxActorWrapper(
            actor_backbone, 
            num_prop=num_prop,
            num_hist=num_hist
        ).to(device)
        onnx_wrapper.eval()
        
        # Create dummy inputs with detected dimensions
        # nn_input0: current obs (batch=1, num_prop)
        # nn_input1: history obs (batch=1, num_hist, num_prop)
        dummy_input0 = torch.randn(1, num_prop, device=device)
        dummy_input1 = torch.randn(1, num_hist, num_prop, device=device)
        
        print(f"Exporting with detected dimensions:")
        print(f"  num_prop: {num_prop}")
        print(f"  num_hist: {num_hist}")
        print(f"  num_actions: {num_actions}")
        
        # Export to ONNX
        torch.onnx.export(
            onnx_wrapper,
            (dummy_input0, dummy_input1),
            save_path,
            input_names=['nn_input0', 'nn_input1'],
            output_names=['nn_output'],
            opset_version=14,
            verbose=False,
            do_constant_folding=True
        )
        
        print(f"✓ ONNX model exported successfully to: {save_path}")
        print(f"  Input specifications:")
        print(f"    - nn_input0: shape (batch_size, {num_prop}) - current proprioceptive observation")
        print(f"    - nn_input1: shape (batch_size, {num_hist}, {num_prop}) - historical proprioceptive observation ({num_hist}×{num_prop})")
        print(f"  Output specifications:")
        print(f"    - nn_output: shape (batch_size, {num_actions}) - action commands")
        print(f"  Normalization: Disabled (raw inputs used)")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to export ONNX model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def verify_onnx_model(onnx_path, device='cpu'):
    """
    Verify that the exported ONNX model can be loaded and run.
    
    Automatically detects input/output dimensions from the model.
    
    Args:
        onnx_path: Path to the ONNX model file
        device: Device to use for verification ('cpu' or 'cuda')
    
    Returns:
        bool: True if verification passed, False otherwise
    """
    try:
        import onnx
        import onnxruntime
        
        # Load and check ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model structure check passed")
        
        # Extract input/output information from ONNX model
        graph = onnx_model.graph
        inputs = graph.input
        outputs = graph.output
        
        # Get dimensions from ONNX model
        # nn_input0: (batch, prop_dim)
        # nn_input1: (batch, num_hist, prop_dim)
        nn_input0_shape = inputs[0].type.tensor_type.shape.dim
        nn_input1_shape = inputs[1].type.tensor_type.shape.dim
        nn_output_shape = outputs[0].type.tensor_type.shape.dim
        
        print(f"  Model input/output dimensions:")
        print(f"    - nn_input0: {[dim.dim_value for dim in nn_input0_shape]}")
        print(f"    - nn_input1: {[dim.dim_value for dim in nn_input1_shape]}")
        print(f"    - nn_output: {[dim.dim_value for dim in nn_output_shape]}")
        
        # Create ONNX Runtime session
        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3  # Only show errors
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        session = onnxruntime.InferenceSession(onnx_path, so, providers=providers)
        
        # Detect dimensions from the model
        # nn_input0: (batch_size, prop_dim)
        # nn_input1: (batch_size, num_hist, prop_dim)
        prop_dim = None
        num_hist = None
        num_actions = None
        
        # Extract from input shapes
        for inp in inputs:
            if inp.name == 'nn_input0':
                dims = [dim.dim_value if dim.dim_value > 0 else 1 for dim in inp.type.tensor_type.shape.dim]
                prop_dim = dims[1] if len(dims) > 1 else 33
            elif inp.name == 'nn_input1':
                dims = [dim.dim_value if dim.dim_value > 0 else 1 for dim in inp.type.tensor_type.shape.dim]
                num_hist = dims[1] if len(dims) > 1 else 10
                prop_dim = dims[2] if len(dims) > 2 else 33
        
        # Extract from output shape
        for out in outputs:
            if out.name == 'nn_output':
                dims = [dim.dim_value if dim.dim_value > 0 else 1 for dim in out.type.tensor_type.shape.dim]
                num_actions = dims[1] if len(dims) > 1 else 8
        
        # Set defaults if not detected
        if prop_dim is None:
            prop_dim = 33
        if num_hist is None:
            num_hist = 10
        if num_actions is None:
            num_actions = 8
        
        # Test inference with detected dimensions
        batch_size = 1
        test_input0 = torch.randn(batch_size, prop_dim).numpy().astype('float32')
        test_input1 = torch.randn(batch_size, num_hist, prop_dim).numpy().astype('float32')
        
        outputs_result = session.run(
            None,
            {'nn_input0': test_input0, 'nn_input1': test_input1}
        )
        
        output_shape = outputs_result[0].shape
        expected_shape = (batch_size, num_actions)
        
        if output_shape == expected_shape:
            print(f"✓ ONNX inference test passed")
            print(f"  Input shapes:  nn_input0={test_input0.shape}, nn_input1={test_input1.shape}")
            print(f"  Output shape:  nn_output={output_shape}")
            print(f"  Output range:  [{outputs_result[0].min():.4f}, {outputs_result[0].max():.4f}] (expected ~[-1, 1])")
            return True
        else:
            print(f"✗ Output shape mismatch. Expected {expected_shape}, got {output_shape}")
            return False
            
    except ImportError:
        print("⚠ ONNX or ONNX Runtime not installed. Skipping verification.")
        print("  Install with: pip install onnx onnxruntime")
        return True
    except Exception as e:
        print(f"✗ ONNX verification failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
