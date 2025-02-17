import torch
import tqdm
import numpy as np
import nltk

from utils import DEVICE, FeatureExtractor, HWT, MGT
from roberta_model_loader import roberta_model
from meta_train import net
from data_loader import load_HC3, filter_data


feature_extractor = FeatureExtractor(roberta_model, net)

target = MGT

# load target data
data_o = load_HC3()
data = filter_data(data_o)
data = data[target]
# print(data[:3])

# split with nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
paragraphs = [nltk.sent_tokenize(paragraph)[1:-1] for paragraph in data]
data = [sent for paragraph in paragraphs for sent in paragraph if 5 < len(sent.split())]
# print(data[:3])

# extract features
feature_ref = []
for i in tqdm.tqdm(range(1000), desc=f"Generating feature ref for {target}"):
    feature_ref.append(
        feature_extractor.process(data[i], False).detach()
    )  # detach to save memory
torch.save(torch.cat(feature_ref, dim=0), f"feature_ref_{target}_1000.pt")
