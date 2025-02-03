import torch
from roberta_model_loader import roberta_model
from feature_ref_loader import feature_ref
from meta_train import net
from regression_model_loader import regression_model
from MMD import MMD_batch2
from utils import DEVICE


class TwoSampleTester:
    def __init__(self):
        print("TwoSample Tester init")

    def test(self, input_text):
        print("TwoSample Tester test")
        # Tokenize
        tokens = roberta_model.tokenizer(
            [input_text],
            padding="max_length",
            truncation=True,
            max_length=100,
            return_tensors="pt",
        ).to(DEVICE)
        # Predict
        outputs = roberta_model.model(**tokens)
        # Get the feature for input text
        hidden_states_all = outputs[2]
        hidden_states = hidden_states_all[-1]
        token_mask_10 = tokens["attention_mask"].unsqueeze(-1)
        hidden_states_mask_10 = hidden_states * token_mask_10
        feature_for_input_text = net.net(hidden_states_mask_10)
        # Calculate MMD
        mmd_feature_for_input_text = MMD_batch2(
            torch.cat([feature_ref.test_feature_ref, feature_for_input_text], dim=0),
            feature_ref.test_feature_ref.shape[0],
            0,
            net.sigma,
            net.sigma0_u,
            net.ep,
        ).to("cpu")
        # Use the regression model to get the 2-sample test result
        y_pred_loaded = regression_model.model.predict(
            mmd_feature_for_input_text.detach().numpy().reshape(-1, 1)
        )

        prediction = int(y_pred_loaded[0])
        if prediction == 0:
            return "Human"
        elif prediction == 1:
            return "AI"


two_sample_tester = TwoSampleTester()
