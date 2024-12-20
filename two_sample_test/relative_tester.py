from model_loader import ModelLoader


class RelativeTester:
    def __init__(self):
        loader = ModelLoader()
        self.model, self.tokenizer = loader.model, loader.tokenizer
        print("Relative Tester initialized")

    def test(self, input_text):
        print(self.tokenizer(input_text))
        return f"Relative Tester result {input_text}"
