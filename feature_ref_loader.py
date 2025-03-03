import torch
import numpy as np

from utils import get_device, config

DEVICE = get_device()


def feature_ref_loader(feature_ref_file_name, num_ref=5000):
    print("Feature Ref Loader load: ", feature_ref_file_name)
    load_ref_data = torch.load(feature_ref_file_name, map_location=DEVICE)  # cpu
    load_ref_data = load_ref_data.to(DEVICE)
    feature_ref = load_ref_data[np.random.permutation(load_ref_data.shape[0])][
        :num_ref
    ].to(DEVICE)
    return feature_ref
