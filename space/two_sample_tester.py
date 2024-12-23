import torch

class TwoSampleTester:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        print("TwoSample Tester initialized")

    def test(self, input_text):
        tokens = self.tokenizer(
            [input_text],
            padding="max_length",
            truncation=True,
            max_length=100,
            return_tensors="pt",
        ).to(torch.device("cpu"))
        print("tokens: ", tokens)
        print("model output:", self.model(**tokens))
        return f"TwoSample Tester result {input_text}"
