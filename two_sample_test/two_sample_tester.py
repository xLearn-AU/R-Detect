from model_loader import ModelLoader


class TwoSampleTester:
    def __init__(self):
        # TODO: Initialize the feature extractor model
        # TODO: Initialize the text converter model
        self.model, self.tokenizer = ModelLoader()

    def test(self, input_text):
        # TODO: Implement the test function
        print(self.tokenizer(input_text))
        return f"Two Sample Tester result {input_text}"
