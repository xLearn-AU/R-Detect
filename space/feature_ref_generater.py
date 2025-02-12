import torch
import numpy as np
from utils import DEVICE, FeatureExtractor, HWT, MGT
from roberta_model_loader import roberta_model
from meta_train import net
from data_loader import load_HC3, filter_data


feature_extractor = FeatureExtractor(roberta_model, net)

data_o = load_HC3()
data = filter_data(data_o)
real = data[HWT]  # [:args.train_real_num]  len== n_samples, many sentences of words
generated = data[MGT]
feature_real = feature_extractor.process(real)
print(feature_real.shape)
