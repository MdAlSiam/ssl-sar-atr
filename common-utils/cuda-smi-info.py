import subprocess

def get_cuda_version():
    try:
        # Run the nvcc command to get the CUDA version
        cuda_version = subprocess.check_output(['nvcc', '--version']).decode('utf-8')
        return cuda_version
    except Exception as e:
        return f"Error getting CUDA version: {e}"

def get_nvidia_smi_info():
    try:
        # Run the nvidia-smi command to get GPU information
        nvidia_smi_info = subprocess.check_output(['nvidia-smi']).decode('utf-8')
        return nvidia_smi_info
    except Exception as e:
        return f"Error getting nvidia-smi info: {e}"

# Print CUDA version
print("CUDA Version:")
print(get_cuda_version())

# Print NVIDIA GPU information
print("NVIDIA GPU Info:")
print(get_nvidia_smi_info())
