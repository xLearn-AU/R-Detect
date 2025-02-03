import torch
import random
import numpy as np

gpu_using = False

DEVICE = torch.device("cpu")
if gpu_using:
    DEVICE = torch.device("cuda:0")


def init_random_seeds():
    print("Init random seeds")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
