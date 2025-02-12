import torch
from roberta_model_loader import roberta_model
from feature_ref_loader import feature_mgt_ref, feature_hwt_ref
from meta_train import net
from regression_model_loader import regression_model
from MMD import MMD_3_Sample_Test
from utils import DEVICE


class RelativeTester:
    def __init__(self):
        print("Relative Tester init")

    def test(self, input_text, alpha=0.05):
        print("Relative Tester test")
        tokens = roberta_model.tokenizer(
            [input_text],
            padding="max_length",
            truncation=True,
            max_length=100,
            return_tensors="pt",
        ).to(DEVICE)
        # Generate the outputs for the input text
        outputs = roberta_model.model(**tokens)
        # Get the feature for input text
        hidden_states_all = outputs[2]
        hidden_states = hidden_states_all[-1]
        token_mask_10 = tokens["attention_mask"].unsqueeze(-1)
        hidden_states_mask_10 = hidden_states * token_mask_10
        # feature_for_input_text = net.net(hidden_states_mask_10) # TODO: how we cutoff the features?
        feature_for_input_text = hidden_states_mask_10
        # Cutoff the features
        min_len = min(
            len(feature_for_input_text),
            len(feature_hwt_ref),
            len(feature_mgt_ref),
        )
        # Calculate MMD
        h_u, p_value, t, *rest = MMD_3_Sample_Test(
            net.net(feature_for_input_text[:min_len]),
            net.net(feature_hwt_ref[torch.randperm(len(feature_hwt_ref))[:min_len]]),
            net.net(feature_mgt_ref[torch.randperm(len(feature_mgt_ref))[:min_len]]),
            feature_for_input_text.view(feature_for_input_text.shape[0], -1),
            feature_hwt_ref.view(feature_hwt_ref.shape[0], -1),
            feature_mgt_ref.view(feature_mgt_ref.shape[0], -1),
            net.sigma,
            net.sigma0_u,
            net.ep,
            0.05,
        )
        # Return the result
        return "Human" if p_value > alpha else "AI"


relative_tester = RelativeTester()
