import torch

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    print(f"Current Device: {torch.cuda.current_device()}")
else:
    print("CUDA is not available. Check your installation.")