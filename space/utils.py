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

# feature_extractor
class FeatureExtractor:
   def __init__(self, model, tokenizer, net, device=DEVICE):
            self.model = model
            self.tokenizer = tokenizer
            self.net = net
            self.device = device

   def process(self, input_text):
        tokens = self.tokenizer(
            [input_text],
            padding="max_length",
            truncation=True,
            max_length=100,
            return_tensors="pt",
		).to(self.device)
        # Predict hidden states using the model
        outputs = self.model(**tokens)
        # Get the feature for input text
        hidden_states_all = outputs.hidden_states
        hidden_states = hidden_states_all[-1]
        # Apply attention mask
        token_mask = tokens["attention_mask"].unsqueeze(-1)
        hidden_states_masked = hidden_states * token_mask
		# 
        feature_for_input_text = self.net(hidden_states_masked)
        return feature_for_input_text
