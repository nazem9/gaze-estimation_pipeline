import torch
import argparse
from pathlib import Path
import sys
import os

# --- Add l2cs path if necessary ---
# If the 'l2cs' library isn't installed globally or in your env's site-packages,
# you might need to add its parent directory to the Python path.
# Example:
# L2CS_PROJECT_DIR = '/path/to/your/L2CS-Net/project' # Adjust this path
# if L2CS_PROJECT_DIR not in sys.path:
#    sys.path.insert(0, L2CS_PROJECT_DIR)
# ----------------------------------

try:
    # Need access to getArch and potentially internal utils
    from l2cs.utils import getArch
except ImportError:
    print("Error: Failed to import 'getArch' from l2cs.utils.")
    print("Please ensure the l2cs library is installed correctly and accessible.")
    print("You might need to add the l2cs project directory to your PYTHONPATH.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during l2cs import: {e}")
    sys.exit(1)

def export_gaze_network_to_torchscript(
    weights_path: Path,
    arch: str,
    output_path: Path,
    input_height: int = 224,
    input_width: int = 224
):
    """
    Exports the core L2CS gaze estimation network to TorchScript format.

    Args:
        weights_path: Path to the .pkl file containing the model's state_dict.
        arch: The architecture string (e.g., 'ResNet50', 'ResNet18').
        output_path: Path where the TorchScript model (.pt file) will be saved.
        input_height: Expected input image height for the model.
        input_width: Expected input image width for the model.
    """
    print(f"--- Exporting L2CS Gaze Network ({arch}) to TorchScript ---")
    print(f"Loading weights from: {weights_path}")
    print(f"Saving TorchScript model to: {output_path}")
    print(f"Expected input size: ({input_height}, {input_width})")

    if not weights_path.exists():
        print(f"Error: Weights file not found at {weights_path}")
        return False

    try:
        # 1. Instantiate the model architecture
        # Assuming the second argument to getArch (output dimensions, 90 here) is consistent
        print(f"Instantiating model architecture: {arch}...")
        model = getArch(arch, 90)
        if model is None:
             print(f"Error: getArch returned None for architecture '{arch}'. Check if arch name is correct.")
             return False

        # 2. Load the weights
        # We load onto CPU for export consistency. The C++ code can load onto GPU if needed.
        print("Loading state dictionary...")
        device = torch.device('cpu')
        try:
            state_dict = torch.load(weights_path, map_location=device)
            # Check if loaded object is a state dict
            if not isinstance(state_dict, dict):
                print("Error: Loaded .pkl file does not appear to be a state dictionary.")
                print("If it's a fully pickled model, loading needs adjustment (not standard for .pkl weights).")
                # Attempt to extract state_dict if it's a model object (less common for .pkl)
                if hasattr(state_dict, 'state_dict') and callable(state_dict.state_dict):
                    print("Attempting to extract state_dict from loaded object...")
                    state_dict = state_dict.state_dict()
                else:
                    return False

            model.load_state_dict(state_dict)
        except Exception as load_err:
            print(f"Error loading state dict from {weights_path}: {load_err}")
            return False

        # 3. Set model to evaluation mode
        model.eval()
        model.to(device) # Ensure model parameters are on the CPU device used for tracing

        # 4. Create a dummy input tensor
        # Shape: (batch_size, channels, height, width)
        # The L2CS pipeline preprocesses the input (normalization etc.). The exported
        # model expects input *after* this preprocessing. The dummy input shape
        # just needs to match what the model's forward pass expects dimensionally.
        dummy_input = torch.randn(1, 3, input_height, input_width, device=device)
        print(f"Created dummy input tensor with shape: {dummy_input.shape}")

        # 5. Trace the model
        print("Tracing model with torch.jit.trace...")
        # Ensure requires_grad is False for dummy input if model wasn't trained with it
        # dummy_input.requires_grad_(False)
        try:
            traced_script_module = torch.jit.trace(model, dummy_input, strict=False) # Use strict=False for flexibility
        except Exception as trace_err:
            print(f"Error during torch.jit.trace: {trace_err}")
            print("Tracing failed. The model might contain data-dependent control flow")
            print("unsuitable for tracing. Consider using torch.jit.script if possible,")
            print("or ensure the model structure is trace-compatible.")
            return False


        # 6. Save the traced model
        print(f"Saving traced model to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        traced_script_module.save(str(output_path))

        print("--------------------------------------------------")
        print(f"Successfully exported gaze network to TorchScript: {output_path}")
        print("NOTE: This model outputs raw pitch and yaw tensors.")
        print("You need to implement the post-processing (softmax, weighted sum, radian conversion)")
        print("in your C++ application separately.")
        print("Expected input: Preprocessed face image tensor [1, 3, H, W].")
        print("--------------------------------------------------")
        return True

    except Exception as e:
        print(f"\nAn unexpected error occurred during export: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Export L2CS Gaze Estimation Network to TorchScript.")
    parser.add_argument("--weights", type=str, required=True, help="Path to the input L2CS .pkl weights file (e.g., models/L2CSNet/Gaze360/L2CSNet_gaze360.pkl).")
    parser.add_argument("--arch", type=str, default="ResNet50", help="Model architecture name (e.g., 'ResNet50', 'ResNet18') used in getArch.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the exported TorchScript model (e.g., exported_models/l2cs_gaze_network.pt).")
    parser.add_argument("--height", type=int, default=224, help="Input image height expected by the model (default: 224).")
    parser.add_argument("--width", type=int, default=224, help="Input image width expected by the model (default: 224).")

    args = parser.parse_args()

    weights_path = Path(args.weights)
    output_path = Path(args.output)

    if not output_path.name.endswith(".pt"):
         print(f"Warning: Output filename '{output_path.name}' does not end with '.pt'. Adding '.pt'.")
         output_path = output_path.with_suffix('.pt')


    export_gaze_network_to_torchscript(
        weights_path=weights_path,
        arch=args.arch,
        output_path=output_path,
        input_height=args.height,
        input_width=args.width
    )

if __name__ == "__main__":
    main()