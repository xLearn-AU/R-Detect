# from model_loader import ModelLoader


class TwoSampleTester:
    def __init__(self):
        print("TwoSample Tester initialized")

    def test(self, input_text):
        print(self.tokenizer(input_text))
        return f"TwoSample Tester result {input_text}"
