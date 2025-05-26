import argparse
from pprint import pprint
from time import perf_counter

import numpy as np
import onnxruntime as ort


def generate_onnx_inputs(input_metadata):
    """
    Generate random input data for an ONNX model based on session input metadata.

    Args:
        input_metadata (list): List of ONNX input metadata from session.get_inputs().
                              Each item has name, shape, and type attributes.

    Returns:
        dict: Dictionary mapping input names to random NumPy arrays with correct shapes and types.

    Raises:
        ValueError: If an unsupported ONNX tensor type is encountered.
    """

    onnx_to_numpy_type = {
        "tensor(int64)": np.int64,
        "tensor(float)": np.float32,
        "tensor(int32)": np.int32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        # Add more mappings as needed
    }

    print("=====================")
    print("Model Input Details:")
    print("=====================")
    pprint(
        [
            {
                "name": i.name,
                "shape": i.shape,
                "type": i.type,
            }
            for i in input_metadata
        ]
    )

    inputs = {}
    for input_meta in input_metadata:
        input_name = input_meta.name
        input_shape = input_meta.shape
        input_type = input_meta.type

        # Replace dynamic dimensions (None or string) with 1
        concrete_shape = [
            1 if dim is None or isinstance(dim, str) else dim for dim in input_shape
        ]

        # Get the appropriate NumPy data type
        if input_type not in onnx_to_numpy_type:
            raise ValueError(f"Unsupported ONNX tensor type: {input_type}")
        numpy_dtype = onnx_to_numpy_type[input_type]

        # Generate random input data
        if numpy_dtype in (np.int64, np.int32):  # Integer types
            input_data = np.random.randint(
                0, 100, size=concrete_shape, dtype=numpy_dtype
            )
        else:  # Floating-point types
            input_data = np.random.rand(*concrete_shape).astype(numpy_dtype)

        inputs[input_name] = input_data

    return inputs


def main():
    parser = argparse.ArgumentParser(description="Run ONNX model with NPU")
    parser.add_argument(
        "--model-path",
        type=str,
        default="./model.onnx",
        help="Path to the ONNX model file",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Number of samples to test",
    )

    args = parser.parse_args()
    model_path = args.model_path

    options = ort.SessionOptions()
    options.set_provider_selection_policy(
        ort.OrtExecutionProviderDevicePolicy.PREFER_NPU
    )
    assert options.has_providers()

    print("=====================")
    print("Creating inference session...")
    print("=====================")

    session = ort.InferenceSession(
        model_path,
        sess_options=options,
    )
    print(f"Model path: {model_path}")
    print(f"Session EPs: {session.get_providers()}")

    inputs = generate_onnx_inputs(session.get_inputs())

    sample_count = args.max_samples
    print("=====================")
    print(f"Inferencing with random sample {sample_count} times...")
    print("=====================")

    start = perf_counter()
    for _ in range(sample_count):
        session.run(None, inputs)
    print(
        f"Processing {sample_count} samples takes {(perf_counter() - start):.2f} seconds"
    )


if __name__ == "__main__":
    main()
