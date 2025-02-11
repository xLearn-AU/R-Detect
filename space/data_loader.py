import random
import tqdm
import datasets
import re
import transformers
import numpy as np

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


filtered_d_HC3 = None


def load_HC3(train_ratio=0.8):
    global filtered_d_HC3

    ds = datasets.load_dataset("Hello-SimpleAI/HC3", name="all")

    ds = ds["train"]

    if filtered_d_HC3 is None:
        filtered_d_HC3 = [
            _
            for _ in ds
            if (
                len(_["human_answers"]) > 0
                and len(_["chatgpt_answers"]) > 0
                and len(_["human_answers"][0].split()) > 5
                and len(_["chatgpt_answers"][0].split()) > 5
            )
        ]
        filtered_d = filtered_d_HC3
    else:
        filtered_d = filtered_d_HC3

    # filtered_d = [_ for _ in ds if (len(_['human_answers']) > 0 and len(_['chatgpt_answers']) > 0 and len(_['human_answers'][0].split()) > 5 and len(_['chatgpt_answers'][0].split()) > 5)]

    data_new = {
        "train": {
            "text": [],
            "label": [],
        },
        "test": {
            "text": [],
            "label": [],
        },
    }

    # random.seed(0)
    random.shuffle(filtered_d)

    total_num = len(filtered_d)
    # total_num = 100
    for i in tqdm.tqdm(range(total_num), desc="parsing data"):
        if i < total_num * train_ratio:
            data_partition = "train"
        else:
            data_partition = "test"
        data_new[data_partition]["text"].append(
            process_spaces(filtered_d[i]["human_answers"][0])
        )
        data_new[data_partition]["label"].append(0)
        data_new[data_partition]["text"].append(
            process_spaces(filtered_d[i]["chatgpt_answers"][0])
        )
        data_new[data_partition]["label"].append(1)
    return data_new


def extract_data(data_o, long_train_threshold_low=150, long_train_threshold_high=512):
    train_real = [
        text
        for text, label in zip(data_o["train"]["text"], data_o["train"]["label"])
        if label == 0
    ]
    train_generated = [
        text
        for text, label in zip(data_o["train"]["text"], data_o["train"]["label"])
        if label == 1
    ]
    long_train_real = [
        x for x in train_real if len(x.split()) > long_train_threshold_low
    ]
    long_train_generated = [
        x for x in train_generated if len(x.split()) > long_train_threshold_low
    ]

    # keep only examples with <= 512 tokens according to mask_tokenizer
    # this step has the extra effect of removing examples with low-quality/garbage content
    tokenized_data = preproc_tokenizer(long_train_real)
    long_train_real = [
        x
        for x, y in zip(long_train_real, tokenized_data["input_ids"])
        if len(y) <= long_train_threshold_high
    ]
    tokenized_data = preproc_tokenizer(long_train_generated)
    long_train_generated = [
        x
        for x, y in zip(long_train_generated, tokenized_data["input_ids"])
        if len(y) <= long_train_threshold_high
    ]

    # print stats about remainining data
    print(f"Total number of samples: {len(long_train_real)}")
    print(
        f"Average number of words: {np.mean([len(x.split()) for x in long_train_real])}"
    )

    data = {
        "original": [],
        "sampled": [],
    }
    for o, s in zip(long_train_real, long_train_generated):
        o, s = trim_to_shorter_length(o, s)

        # add to the data
        data["original"].append(o)
        data["sampled"].append(s)

    return data


data_o = load_HC3(train_ratio=0.8)
data = extract_data(data_o)
real = data[
    "original"
]  # [:args.train_real_num]  len== n_samples, many sentences of words
generated = data["sampled"]
print(real[:5])
print(generated[:5])
