from transformers import RobertaTokenizer, RobertaModel
import torch


class RobertaModelLoader:
    def __init__(
        self,
        model_name="roberta-base-openai-detector",
        cache_dir=".cache",
    ):
        print("Roberta Model init")
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.tokenizer, self.model = self.load_base_model_and_tokenizer()

    def load_base_model_and_tokenizer(self):
        print("Load model: ", self.model_name)
        return RobertaTokenizer.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        ), RobertaModel.from_pretrained(
            self.model_name, output_hidden_states=True, cache_dir=self.cache_dir
        )


roberta_model = RobertaModelLoader()
