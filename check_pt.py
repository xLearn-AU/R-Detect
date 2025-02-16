import torch

features = torch.load("./space/feature_ref_MGT.pt", map_location=torch.device("cpu"))
print(type(features), len(features), features[0].shape, features[0].dtype)
