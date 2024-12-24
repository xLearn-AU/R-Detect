import torch
from roberta_model_loader import roberta_model


class TwoSampleTester:
    def __init__(self):
        print("TwoSample Tester init")

    def test(self, input_text):
        print("TwoSample Tester test")
        tokens = roberta_model.tokenizer(
            [input_text],
            padding="max_length",
            truncation=True,
            max_length=100,
            return_tensors="pt",
        ).to(torch.device("cpu"))
        # print("tokens: ", tokens)
        # print("model output:", roberta_model.model(**tokens))
        return f"TwoSample Tester result {input_text}"


two_sample_tester = TwoSampleTester()
