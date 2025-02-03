import torch
import numpy as np
from utils import DEVICE


class FeatureRefLoader:
    def __init__(self):
        print("Feature Ref Loader init")
        self.test_feature_ref = self.load("./feature_ref_for_test.pt")

    # TODO: The format of feature
    def load(self, feature_ref_file_name, num_ref=5000):
        print("Feature Ref Loader load")
        load_ref_data = torch.load(feature_ref_file_name, map_location=DEVICE)  # cpu
        load_ref_data = load_ref_data.to(DEVICE)
        feature_ref = load_ref_data[np.random.permutation(load_ref_data.shape[0])][
            :num_ref
        ].to(DEVICE)
        return feature_ref


feature_ref = FeatureRefLoader()

# TODO: load MGT and HWT feature ref here
feature_mgt_ref = FeatureRefLoader()
feature_hwt_ref = FeatureRefLoader()
