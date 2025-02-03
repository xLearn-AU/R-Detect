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


class FeatureExtractor:
    def __init__(self, model, net):
        self.model = model # TODO: support different models
        self.net = net

    def process(self, text):
        print("Feature Extractor process")
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
        token_mask_10 = tokens["attention_mask"].unsqueeze(-1)
        hidden_states_mask_10 = (
            outputs.last_hidden_state * token_mask_10
        )  # TODO: why we need this?
        feature = self.net.net(hidden_states_mask_10)
        return feature
