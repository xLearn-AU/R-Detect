from model_loader import ModelLoader
from two_sample_tester import TwoSampleTester


class RelativeTester:
    def __init__(self, model, tokenizer):
        self.tokenizer = tokenizer
        self.model = model
        print("Relative Tester initialized")

    def test(self, input_text):
        two_sample_tester = TwoSampleTester(self.model, self.tokenizer)
        two_sample_tester.test(input_text)
        return f"Relative Tester result {input_text}"
