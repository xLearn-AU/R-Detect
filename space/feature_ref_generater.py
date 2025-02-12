import torch
import tqdm
import numpy as np
from utils import DEVICE, FeatureExtractor, HWT, MGT
from roberta_model_loader import roberta_model
from meta_train import net
from data_loader import load_HC3, filter_data


feature_extractor = FeatureExtractor(roberta_model, net)

target = MGT

data_o = load_HC3()
data = filter_data(data_o)
target_data = data[
    target
]  # [:args.train_real_num]  len== n_samples, many sentences of words
feature_ref = []
for i in tqdm.tqdm(range(500), desc=f"Generating feature ref for {target}"):
    feature_ref.append(feature_extractor.process(target_data[i]))
torch.save(torch.cat(feature_ref, dim=0), f"feature_ref_{target}.pt")
