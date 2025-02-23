from transformers import RobertaTokenizer, RobertaModel
import torch
from utils import config


class RobertaModelLoader:
    def __init__(
        self,
        model_name="roberta-base-openai-detector",
        cache_dir=".cache",
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.tokenizer, self.model = self.load_base_model_and_tokenizer()

    def load_base_model_and_tokenizer(self):
        if config["local_model"]:  # load model from local
            print("Load model from local: ", self.model_name, config["local_model"])
            return RobertaTokenizer.from_pretrained(
                config["local_model"], cache_dir=self.cache_dir
            ), RobertaModel.from_pretrained(
                config["local_model"],
                output_hidden_states=True,
                cache_dir=self.cache_dir,
            )

        print("Load model from remote: ", self.model_name)
        return RobertaTokenizer.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        ), RobertaModel.from_pretrained(
            self.model_name, output_hidden_states=True, cache_dir=self.cache_dir
        )
