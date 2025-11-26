import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# create a small tensor
x = torch.tensor([1.0, 2.0, 3.0])
print("Tensor:", x)
print("Tensor * 2:", x * 2)
