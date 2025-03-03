import torch

features = torch.load("./fea_test.pt", map_location=torch.device("cpu"))
print(type(features), len(features), features[0].shape, features[0].dtype)
