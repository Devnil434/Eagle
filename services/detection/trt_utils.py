"""
trt_utils.py — Low-level TensorRT inference utilities for Eagle.

Provides the `TensorRTInference` class to load and execute serialized .engine files 
directly on NVIDIA GPUs with optimized CUDA bindings, including memory management 
and asynchronous stream coordination.
"""

from __future__ import annotations
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Safe imports to prevent crashes on systems without NVIDIA drivers/TensorRT installed.
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    trt = None
    TRT_AVAILABLE = False

try:
    import pycuda.driver as cuda
    import pycuda.autoinit  # Automatically handles CUDA context creation/destruction
    CUDA_AVAILABLE = True
except ImportError:
    cuda = None
    CUDA_AVAILABLE = False


class TensorRTInference:
    """
    Handles low-level TensorRT model deserialization, binding memory allocation
    (host-to-device and device-to-host pagelocked buffers), and optimized
    inference for compiled .engine files on NVIDIA GPUs.
    """
    def __init__(self, engine_path: str) -> None:
        """
        Initialize the TensorRT inference engine.

        Args:
            engine_path: Path to the serialized `.engine` model file.
        """
        if not TRT_AVAILABLE:
            raise ImportError(
                "TensorRT python package is not installed. "
                "Please install tensorrt using: pip install tensorrt"
            )
        if not CUDA_AVAILABLE:
            raise ImportError(
                "PyCUDA is not installed or CUDA is unavailable. "
                "Please install pycuda using: pip install pycuda"
            )

        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        logger.info(f"Deserializing TensorRT Engine: {self.engine_path}")
        with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine from {self.engine_path}")
            
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create TensorRT execution context for {self.engine_path}")
            
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        self._allocate_buffers()
        logger.info(f"TensorRT Engine loaded successfully. Inputs: {len(self.inputs)}, Outputs: {len(self.outputs)}")

    def _allocate_buffers(self) -> None:
        """
        Query binding metadata from the engine and allocate pinned/pagelocked
        host memory and GPU device buffers for each input/output tensor.
        """
        # Determine maximum batch size
        max_batch_size = 1
        if hasattr(self.engine, "max_batch_size"):
            max_batch_size = max(1, self.engine.max_batch_size)

        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            # Handle dynamic/undefined batch dimension
            if shape[0] == -1:
                shape = (max_batch_size,) + shape[1:]
                
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Pinned/pagelocked host memory for faster DMA transfers
            host_mem = cuda.pagelocked_empty(size, dtype)
            # CUDA device memory allocation
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            binding_info = {
                "host": host_mem,
                "device": device_mem,
                "name": binding,
                "dtype": dtype,
                "shape": shape
            }
            
            if self.engine.binding_is_input(binding):
                self.inputs.append(binding_info)
            else:
                self.outputs.append(binding_info)

    def infer(self, input_data: np.ndarray) -> list[np.ndarray]:
        """
        Performs synchronized, high-speed inference on a preprocessed input frame.

        Args:
            input_data: Preprocessed input image numpy array.

        Returns:
            A list of numpy arrays representing raw model predictions.
        """
        if not self.inputs:
            raise ValueError("No input bindings allocated in the TensorRT engine.")

        input_info = self.inputs[0]
        # Fast copy to pagelocked host buffer
        np.copyto(input_info["host"], input_data.ravel())
        
        # Host to Device transfer (Asynchronous)
        cuda.memcpy_htod_async(input_info["device"], input_info["host"], self.stream)
        
        # Enqueue inference execution context
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Device to Host transfer (Asynchronous)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
            
        # Synchronize CPU and GPU stream execution
        self.stream.synchronize()
        
        # Reshape output vectors back to standard multi-dimensional tensors
        results = []
        for out in self.outputs:
            reshaped = out["host"].reshape(out["shape"])
            results.append(reshaped)
            
        return results

    def __del__(self) -> None:
        """
        Cleans up GPU bindings and device pointers when the class object is garbage collected.
        """
        self.bindings.clear()
        self.inputs.clear()
        self.outputs.clear()


def is_tensorrt_supported() -> bool:
    """
    Utility check to see if the local machine fully supports native TensorRT execution.
    
    Returns:
        True if tensorrt and pycuda are installed and available, False otherwise.
    """
    return TRT_AVAILABLE and CUDA_AVAILABLE
