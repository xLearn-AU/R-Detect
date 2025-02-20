import torch
import random
import numpy as np

gpu_using = False
DEVICE = torch.device("cpu")
if gpu_using:
    DEVICE = torch.device("cuda:0")

HWT = "HWT"
MGT = "MGT"


def init_random_seeds():
    print("Init random seeds")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class FeatureExtractor:
    def __init__(self, model, net=None):
        self.model = model  # TODO: support different models
        self.net = net

    def process(self, text, net_required=True):
        # Tokenize
        tokens = self.model.tokenizer(
            [text],
            padding="max_length",
            truncation=True,
            max_length=100,
            return_tensors="pt",
        ).to(DEVICE)
        # Predict
        outputs = self.model.model(**tokens)
        # Get the feature for input text
        attention_mask = tokens["attention_mask"].unsqueeze(-1)
        hidden_states_masked = (
            outputs.last_hidden_state * attention_mask
        )  # Ignore the padding tokens
        if net_required and self.net is not None:
            feature = self.net.net(hidden_states_masked)
            return feature
        else:
            return hidden_states_masked

    def process_sents(self, sents, net_required=True):
        features = []
        for sent in sents:
            features.append(self.process(sent, net_required))
        if not features:
            return torch.tensor([])
        return torch.cat(features, dim=0)
