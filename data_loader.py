import random
import tqdm
import datasets
import re
import transformers
import numpy as np
from utils import MGT, HWT, config

preproc_tokenizer = transformers.AutoTokenizer.from_pretrained(
    "google-t5/t5-small", model_max_length=512
)


def process_spaces(text):
    text = (
        text.replace(" ,", ",")
        .replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ;", ";")
        .replace(" '", "'")
        .replace(" ’ ", "'")
        .replace(" :", ":")
        .replace("<newline>", "\n")
        .replace("`` ", '"')
        .replace(" ''", '"')
        .replace("''", '"')
        .replace(".. ", "... ")
        .replace(" )", ")")
        .replace("( ", "(")
        .replace(" n't", "n't")
        .replace(" i ", " I ")
        .replace(" i'", " I'")
        .replace("\\'", "'")
        .replace("\n ", "\n")
        .strip()
    )
    text = text.replace("\r\n", "\n").replace("\\n", "").replace("!\n", "")
    return re.sub("\n+", "\n", text)


def trim_to_shorter_length(texta, textb):
    # truncate to shorter of o and s
    shorter_length = min(len(texta.split(" ")), len(textb.split(" ")))
    texta = " ".join(texta.split(" ")[:shorter_length])
    textb = " ".join(textb.split(" ")[:shorter_length])
    return texta, textb


def load_HC3():
    if config["local_dataset"]:
        print("Loading local HC3 dataset", config["local_dataset"])
    else:
        print("Loading remote HC3 dataset")
    ds = (
        datasets.load_dataset(
            config["local_dataset"], name="all"
        )
        if config["local_dataset"]
        else datasets.load_dataset("Hello-SimpleAI/HC3", name="all")
    )
    ds = ds["train"]  # DatasetDict -> Dataset
    filtered_ds = [
        item
        for item in ds
        if (
            len(item["human_answers"]) > 0
            and len(item["chatgpt_answers"]) > 0
            and len(item["human_answers"][0].split()) > 5
            and len(item["chatgpt_answers"][0].split()) > 5
        )
    ]
    # print("DEBUG: filtered_ds[0]:", filtered_ds[0])

    data_new = {"text": [], "label": []}

    for i in tqdm.tqdm(range(len(filtered_ds)), desc="Parsing data"):
        data_new["text"].append(process_spaces(filtered_ds[i]["human_answers"][0]))
        data_new["label"].append(HWT)
        data_new["text"].append(process_spaces(filtered_ds[i]["chatgpt_answers"][0]))
        data_new["label"].append(MGT)
    return data_new


def filter_data(data_o, long_train_threshold_low=150, long_train_threshold_high=512):
    data_HWT = [
        text for text, label in zip(data_o["text"], data_o["label"]) if label == HWT
    ]
    data_MGT = [
        text for text, label in zip(data_o["text"], data_o["label"]) if label == MGT
    ]

    # keep only examples with <= 512 tokens according to mask_tokenizer
    # this step has the extra effect of removing examples with low-quality/garbage content
    tokenized_data = preproc_tokenizer(data_HWT)
    long_HWT = [
        x
        for x, y in zip(data_HWT, tokenized_data["input_ids"])
        if long_train_threshold_low <= len(y) <= long_train_threshold_high
    ]
    tokenized_data = preproc_tokenizer(data_MGT)
    long_MGT = [
        x
        for x, y in zip(data_MGT, tokenized_data["input_ids"])
        if long_train_threshold_low <= len(y) <= long_train_threshold_high
    ]

    # print stats about remainining data
    print(f"Total number of samples: {len(long_HWT)}")
    print(f"Average number of words: {np.mean([len(x.split()) for x in long_HWT])}")

    data = {
        HWT: [],
        MGT: [],
    }

    # print(len(long_HWT), len(long_MGT))
    for o, s in zip(long_HWT, long_MGT):
        o, s = trim_to_shorter_length(o, s)

        # add to the data
        data[HWT].append(o)
        data[MGT].append(s)

    return data


# Test code
# data_o = load_HC3()
# data = filter_data(data_o)
# real = data[HWT]  # [:args.train_real_num]  len== n_samples, many sentences of words
# generated = data[MGT]
# print(real[:5])
# print(generated[:5])
