import torch
import tqdm
import numpy as np
from utils import DEVICE, FeatureExtractor, HWT, MGT
from roberta_model_loader import roberta_model
from meta_train import net
from data_loader import load_HC3, filter_data


feature_extractor = FeatureExtractor(roberta_model, net)

data_o = load_HC3()
data = filter_data(data_o)
real = data[HWT]  # [:args.train_real_num]  len== n_samples, many sentences of words
# generated = data[MGT]
feature_ref_real = []
for i in tqdm.tqdm(range(len(real)), desc="Generating feature ref"):
    feature_ref_real.append(feature_extractor.process(real[i]))


print(feature_ref_real.shape, type(feature_ref_real))
