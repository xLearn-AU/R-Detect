from transformers import RobertaTokenizer, RobertaModel
import torch


class ModelLoader:
    def __init__(
        self,
        model="roberta-base-openai-detector",
        cache_dir=".cache",
    ):
        self.model = model
        self.cache_dir = cache_dir
        self.tokenizer, self.model = self.load_base_model_and_tokenizer()

    def load_base_model_and_tokenizer(self):
        tokenizer = RobertaTokenizer.from_pretrained(
            self.model, cache_dir=self.cache_dir
        )
        model = RobertaModel.from_pretrained(
            self.model, output_hidden_states=True, cache_dir=self.cache_dir
        )
        return tokenizer, model
