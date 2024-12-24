import torch

gpu_using = False

DEVICE = torch.device("cpu")
if gpu_using:
    DEVICE = torch.device("cuda:0")
