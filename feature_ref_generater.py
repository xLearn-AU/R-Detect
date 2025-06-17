import torch
import tqdm
import numpy as np
import nltk
import argparse

from utils import FeatureExtractor, HWT, MGT, config
from roberta_model_loader import RobertaModelLoader
from meta_train import net
from data_loader import load_HC3, filter_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="R-Detect the file content")
    parser.add_argument(
        "--target",
        type=str,
        help="The target of generated feature ref. Default is MGT",
        default=MGT,
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        help="The sample size of generated feature ref. Default is 1000, must bigger than 100 and smaller than 30000",
        default=1000,
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU or not.",
    )
    parser.add_argument(
        "--local_model",
        type=str,
        help="Use local model or not, you need to download the model first, and set the path. Default is Empty. Script will use remote if this param is empty.",
        default="",
    )
    parser.add_argument(
        "--local_dataset",
        type=str,
        help="Use local dataset or not, you need to download the dataset first, and set the path. Default is Empty",
        default="",
    )
    args = parser.parse_args()
    config["target"] = args.target
    if args.sample_size < 100 or args.sample_size > 30000:
        print("Sample size must be between 100 and 30000, set to 1000")
        config["sample_size"] = 1000
    else:
        config["sample_size"] = args.sample_size
    config["use_gpu"] = args.use_gpu
    config["local_model"] = args.local_model
    config["local_dataset"] = args.local_dataset
    target = HWT if config["target"] == HWT else MGT
    # load model and feature extractor
    roberta_model = RobertaModelLoader()
    feature_extractor = FeatureExtractor(roberta_model, net)
    # load target data
    data_o = load_HC3()
    data = filter_data(data_o)
    data = data[target]
    # print(data[:3])

    # split with nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    paragraphs = [nltk.sent_tokenize(paragraph)[1:-1] for paragraph in data]
    data = [
        sent for paragraph in paragraphs for sent in paragraph if 5 < len(sent.split())
    ]
    # print(data[:3])

    # extract features
    feature_ref = []
    for i in tqdm.tqdm(
        range(config["sample_size"]), desc=f"Generating feature ref for {target}"
    ):
        feature_ref.append(
            feature_extractor.process(data[i], False).detach()
        )  # detach to save memory
    torch.save(
        torch.cat(feature_ref, dim=0),
        f"feature_ref_{target}_{config['sample_size']}.pt",
    )
    print(f"Feature ref for {target} generated successfully")
