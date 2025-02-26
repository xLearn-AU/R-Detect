import torch
import nltk
from roberta_model_loader import RobertaModelLoader
from feature_ref_loader import feature_ref_loader
from meta_train import net
from regression_model_loader import regression_model
from MMD import MMD_3_Sample_Test
from utils import FeatureExtractor, HWT, MGT, config


class RelativeTester:
    def __init__(self):
        print("Relative Tester init")
        self.feature_extractor = FeatureExtractor(RobertaModelLoader(), net)
        self.feature_hwt_ref = feature_ref_loader(config["feature_ref_HWT"])
        self.feature_mgt_ref = feature_ref_loader(config["feature_ref_MGT"])

    def sents_split(self, text):
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        sents = nltk.sent_tokenize(text)
        return [sent for sent in sents if 5 < len(sent.split())]

    def test(self, input_text, threshold=0.2, round=20):
        print("Relative Tester test")
        # Split the input text
        sents = self.sents_split(input_text)
        print("DEBUG: sents:", len(sents))
        # Extract features
        feature_for_sents = self.feature_extractor.process_sents(sents, False)
        if len(feature_for_sents) <= 1:
            # print("DEBUG: tooshort")
            return "Too short to test! Please input more than 2 sentences."
        # Cutoff the features
        min_len = min(
            len(feature_for_sents),
            len(self.feature_hwt_ref),
            len(self.feature_mgt_ref),
        )
        # Calculate MMD
        h_u_list = []
        p_value_list = []
        t_list = []

        for i in range(round):
            feature_for_sents_sample = feature_for_sents[
                torch.randperm(len(feature_for_sents))[:min_len]
            ]
            feature_hwt_ref_sample = self.feature_hwt_ref[
                torch.randperm(len(self.feature_hwt_ref))[:min_len]
            ]
            feature_mgt_ref_sample = self.feature_mgt_ref[
                torch.randperm(len(self.feature_mgt_ref))[:min_len]
            ]
            h_u, p_value, t, *rest = MMD_3_Sample_Test(
                net.net(feature_for_sents_sample),
                net.net(feature_hwt_ref_sample),
                net.net(feature_mgt_ref_sample),
                feature_for_sents_sample.view(feature_for_sents_sample.shape[0], -1),
                feature_hwt_ref_sample.view(feature_hwt_ref_sample.shape[0], -1),
                feature_mgt_ref_sample.view(feature_mgt_ref_sample.shape[0], -1),
                net.sigma,
                net.sigma0_u,
                net.ep,
                0.05,
            )

            h_u_list.append(h_u)
            p_value_list.append(p_value)
            t_list.append(t)

        power = sum(h_u_list) / len(h_u_list)
        print("DEBUG: power:", power)
        print("DEBUG: power list:", h_u_list)
        # Return the result
        return (
            "Most likely Human Write"
            if power <= threshold
            else "Most likely Machine Generated"
        )
