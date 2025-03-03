import torch
from roberta_model_loader import roberta_model
from feature_ref_loader import feature_two_sample_tester_ref
from meta_train import net
from regression_model_loader import regression_model
from MMD import MMD_batch2
from utils import DEVICE, FeatureExtractor


class TwoSampleTester:
    def __init__(self, net=net, model=roberta_model):
        print("TwoSample Tester init")
        self.net = net
        self.model = model
        self.feature_extractor = FeatureExtractor(model, net)

    def test(self, input_text):
        print("TwoSample Tester test")
        # Get the feature for input text
        feature_for_input_text = self.feature_extractor.process(input_text)
        # print(
        #     "DEBUG: feature_for_input_text:",
        #     feature_for_input_text.shape,
        #     feature_two_sample_tester_ref.shape,
        # )
        # Calculate MMD
        mmd_feature_for_input_text = MMD_batch2(
            torch.cat([feature_two_sample_tester_ref, feature_for_input_text], dim=0),
            feature_two_sample_tester_ref.shape[0],
            0,
            self.net.sigma,
            self.net.sigma0_u,
            self.net.ep,
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
