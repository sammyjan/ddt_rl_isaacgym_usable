"""
Quick verification script for ONNX model dimensions.
Run this after exporting the ONNX model to verify dimensions are correct.
"""

import argparse
import numpy as np

try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


def check_onnx_dimensions(onnx_path):
    """Check ONNX model input/output dimensions."""
    
    if not HAS_ONNX:
        print("✗ ONNX or ONNX Runtime not installed")
        print("  Install with: pip install onnx onnxruntime")
        return False
    
    try:
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        graph = onnx_model.graph
        inputs = graph.input
        outputs = graph.output
        
        print("\n" + "="*60)
        print("ONNX Model Dimension Check")
        print("="*60)
        
        # Extract dimensions
        print("\n📥 Input Nodes:")
        for inp in inputs:
            name = inp.name
            shape = inp.type.tensor_type.shape.dim
            dims = [dim.dim_value if dim.dim_value > 0 else '?' for dim in shape]
            print(f"  {name}: {dims}")
        
        print("\n📤 Output Nodes:")
        for out in outputs:
            name = out.name
            shape = out.type.tensor_type.shape.dim
            dims = [dim.dim_value if dim.dim_value > 0 else '?' for dim in shape]
            print(f"  {name}: {dims}")
        
        # Create session and test inference
        print("\n🧪 Testing inference:")
        session = ort.InferenceSession(onnx_path)
        
        # Extract actual dimensions from inputs
        input_shapes = {}
        for inp in inputs:
            shape = inp.type.tensor_type.shape.dim
            dims = tuple(dim.dim_value if dim.dim_value > 0 else 1 for dim in shape)
            input_shapes[inp.name] = dims
        
        # Create test inputs
        test_inputs = {}
        for name, shape in input_shapes.items():
            test_inputs[name] = np.random.randn(*shape).astype(np.float32)
            print(f"  Input '{name}': shape {shape}")
        
        # Run inference
        output = session.run(None, test_inputs)
        
        print(f"  Output shape: {output[0].shape}")
        print(f"  Output range: [{output[0].min():.4f}, {output[0].max():.4f}]")
        
        print("\n✅ ONNX model is valid and can be used for deployment")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print(f"✗ Error checking ONNX model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check ONNX model dimensions')
    parser.add_argument('--onnx_path', type=str, required=True,
                        help='Path to the ONNX model file')
    args = parser.parse_args()
    
    check_onnx_dimensions(args.onnx_path)
