import argparse
import csv
import datetime
import json
import math
import os
import random
import re
import sys
import time
import logging
from collections import namedtuple
from multiprocessing.pool import ThreadPool

import datasets
import nltk
import numpy as np
import openai
import torch
import torch.nn.functional as F
import tqdm
import transformers
from sklearn import metrics
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from transformers import RobertaTokenizer, RobertaModel

import custom_datasets
from MGTBenchold import dataset_loader
from meta_train import mmdPreModel, Meta
from utils_MMD import (
    MMDu,
    MMD_batch2,
    TST_MMD_u,
    TST_MMD_u_3S,
    TST_MMD_u_3S_Baseline,
    TST_MMD_u_3S_Permutation,
    TST_MMD_u_3S_Permutation_Kernel,
)
from scipy.stats import combine_pvalues

import warnings

warnings.filterwarnings("ignore")

# define regex to match all <extra_id_*> tokens, where * is an integer
pattern = re.compile(r"<extra_id_\d+>")

model_path_dit = {
    "gpt2": "pretrained_models/gpt2-medium",
    "roberta-base-openai-detector": "pretrained_models/roberta-base-openai-detector",
    "EleutherAI/gpt-neo-125m": "pretrained_models/gpt-neo-125m",
    "minhtoan/gpt3-small-finetune-cnndaily-news": "pretrained_models/gpt3-small-finetune-cnndaily-news",
    "t5-large": "pretrained_models/t5-large",
    "t5-small": "pretrained_models/t5-small",
    "Hello-SimpleAI/chatgpt-detector-roberta": "pretrained_models/chatgpt-detector-roberta",
    "roberta-base": "pretrained_models/roberta-base",
    "tiiuae/falcon-rw-1b": "pretrained_models/falcon-rw-1b",
}


def plot_mi(clean, adv):
    mi_nat = clean.numpy()
    label_clean = "Clean"

    mi_svhn = adv.numpy()
    label_adv = "Adv"

    mi_nat = mi_nat[~np.isnan(mi_nat)]
    mi_svhn = mi_svhn[~np.isnan(mi_svhn)]

    x = np.concatenate((mi_nat, mi_svhn), 0)
    y = np.zeros(x.shape[0])
    y[mi_nat.shape[0] :] = 1

    ap = metrics.roc_auc_score(y, x)
    fpr, tpr, thresholds = metrics.roc_curve(y, x)
    accs = {th: tpr[np.argwhere(fpr <= th).max()] for th in [0.01, 0.05, 0.1]}
    print(
        "auroc: {:.4f}; ".format(ap)
        + "; ".join(["TPR: {:.4f} @ FPR={:.4f}".format(v, k) for k, v in accs.items()])
        + "  {}-{}".format(len(mi_nat), len(mi_svhn))
    )
    return ap


def load_base_model():
    print("MOVING BASE MODEL TO GPU...", end="", flush=True)
    start = time.time()
    try:
        mask_model.cpu()
    except NameError:
        pass
    if args.openai_model is None:
        base_model.to(DEVICE)
    print(f"DONE ({time.time() - start:.2f}s)")


def load_mask_model():
    print("MOVING MASK MODEL TO GPU...", end="", flush=True)
    start = time.time()

    if args.openai_model is None:
        base_model.cpu()
    if not args.random_fills:
        mask_model.to(DEVICE)
    print(f"DONE ({time.time() - start:.2f}s)")


def tokenize_and_mask(text, span_length, pct, ceil_pct=False):
    tokens = text.split(" ")
    mask_string = "<<<mask>>>"

    n_spans = pct * len(tokens) / (span_length + args.buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - args.buffer_size)
        search_end = min(len(tokens), end + args.buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1

    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f"<extra_id_{num_filled}>"
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = " ".join(tokens)
    return text


def count_masks(texts):
    return [
        len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts
    ]


# replace each masked span with a sample from T5 mask_model
def replace_masks(texts):
    n_expected = count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    outputs = mask_model.generate(
        **tokens,
        max_length=150,
        do_sample=True,
        top_p=args.mask_top_p,
        num_return_sequences=1,
        eos_token_id=stop_id,
    )
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)


def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills


def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(" ") for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts


def perturb_texts_(texts, span_length, pct, ceil_pct=False):
    masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
    raw_fills = replace_masks(masked_texts)
    extracted_fills = extract_fills(raw_fills)
    perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

    # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
    attempts = 1
    while "" in perturbed_texts:
        idxs = [idx for idx, x in enumerate(perturbed_texts) if x == ""]
        print(
            f"WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}]."
        )
        masked_texts = [
            tokenize_and_mask(x, span_length, pct, ceil_pct)
            for idx, x in enumerate(texts)
            if idx in idxs
        ]
        raw_fills = replace_masks(masked_texts)
        extracted_fills = extract_fills(raw_fills)
        new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
        for idx, x in zip(idxs, new_perturbed_texts):
            perturbed_texts[idx] = x
        attempts += 1
    return perturbed_texts


def perturb_texts(texts, span_length, pct, ceil_pct=False):
    chunk_size = args.chunk_size
    if "11b" in mask_filling_model_name:
        chunk_size //= 2

    outputs = []
    for i in tqdm.tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
        outputs.extend(
            perturb_texts_(
                texts[i : i + chunk_size], span_length, pct, ceil_pct=ceil_pct
            )
        )
    return outputs


def drop_last_word(text):
    return " ".join(text.split(" ")[:-1])


def _openai_sample(p):
    if args.dataset != "pubmed":  # keep Answer: prefix for pubmed
        p = drop_last_word(p)

    # sample from the openai model
    kwargs = {"engine": args.openai_model, "max_tokens": 200}
    if args.do_top_p:
        kwargs["top_p"] = args.top_p

    r = openai.Completion.create(prompt=f"{p}", **kwargs)
    return p + r["choices"][0].text


# sample from base_model using ****only**** the first 30 tokens in each example as context
def sample_from_model(texts, min_words=55, prompt_tokens=30):
    # encode each text as a list of token ids
    if args.dataset == "pubmed":
        texts = [t[: t.index(custom_datasets.SEPARATOR)] for t in texts]
        all_encoded = base_tokenizer(texts, return_tensors="pt", padding=True).to(
            DEVICE
        )
    else:
        all_encoded = base_tokenizer(texts, return_tensors="pt", padding=True).to(
            DEVICE
        )
        all_encoded = {
            key: value[:, :prompt_tokens] for key, value in all_encoded.items()
        }

    if args.openai_model:
        # decode the prefixes back into text
        prefixes = base_tokenizer.batch_decode(
            all_encoded["input_ids"], skip_special_tokens=True
        )
        pool = ThreadPool(args.batch_size)

        decoded = pool.map(_openai_sample, prefixes)
    else:
        decoded = ["" for _ in range(len(texts))]

        # sample from the model until we get a sample with at least min_words words for each example
        # this is an inefficient way to do this (since we regenerate for all inputs if just one is too short), but it works
        tries = 0
        while (min(len(x.split()) for x in decoded)) < min_words:
            m = min(len(x.split()) for x in decoded)
            if tries != 0:
                print()
                print(f"min words: {m}, needed {min_words}, regenerating (try {tries})")

            sampling_kwargs = {}
            if args.do_top_p:
                sampling_kwargs["top_p"] = args.top_p
            elif args.do_top_k:
                sampling_kwargs["top_k"] = args.top_k
            min_length = 50 if args.dataset in ["pubmed"] else 150
            outputs = base_model.generate(
                **all_encoded,
                min_length=min_length,
                max_length=200,
                do_sample=True,
                **sampling_kwargs,
                pad_token_id=base_tokenizer.eos_token_id,
                eos_token_id=base_tokenizer.eos_token_id,
            )
            decoded = base_tokenizer.batch_decode(
                outputs[:, prompt_tokens:], skip_special_tokens=True
            )  # remove the first 30 tokens
            tries += 1

    if args.openai_model:
        global API_TOKEN_COUNTER

        # count total number of tokens with GPT2_TOKENIZER
        total_tokens = sum(len(GPT2_TOKENIZER.encode(x)) for x in decoded)
        API_TOKEN_COUNTER += total_tokens

    return decoded


# Get the log likelihood of each text under the base_model
def get_ll(text):
    if args.openai_model:
        kwargs = {
            "engine": args.openai_model,
            "temperature": 0,
            "max_tokens": 0,
            "echo": True,
            "logprobs": 0,
        }
        r = openai.Completion.create(prompt=f"<|endoftext|>{text}", **kwargs)
        result = r["choices"][0]
        tokens, logprobs = (
            result["logprobs"]["tokens"][1:],
            result["logprobs"]["token_logprobs"][1:],
        )

        assert len(tokens) == len(
            logprobs
        ), f"Expected {len(tokens)} logprobs, got {len(logprobs)}"

        return np.mean(logprobs)
    else:
        with torch.no_grad():
            tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
            labels = tokenized.input_ids
            return -base_model(**tokenized, labels=labels).loss.item()


def get_joint_pros(texts, num_tokens):
    joint_pros = [get_joint_pro(text) for text in texts]
    return [x[:num_tokens] for x in joint_pros if len(x) > num_tokens]


def get_joint_pro(text):
    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits
        labels = tokenized.input_ids

        logits = logits.view(-1, logits.shape[-1])[:-1]
        labels = labels.view(-1)[1:]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(
            -1
        )
    return log_likelihood


# get the average rank of each observed token sorted by model likelihood
def get_rank(text, log=False):
    assert args.openai_model is None, "get_rank not implemented for OpenAI models"

    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:, :-1]
        labels = tokenized.input_ids[:, 1:]

        # get rank of each label token in the model's likelihood ordering
        matches = (
            logits.argsort(-1, descending=True) == labels.unsqueeze(-1)
        ).nonzero()

        assert (
            matches.shape[1] == 3
        ), f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:, -1], matches[:, -2]

        # make sure we got exactly one match for each timestep in the sequence
        assert (
            timesteps == torch.arange(len(timesteps)).to(timesteps.device)
        ).all(), "Expected one match per timestep"

        ranks = ranks.float() + 1  # convert to 1-indexed rank
        if log:
            ranks = torch.log(ranks)

        return ranks.float().mean().item()


# get average entropy of each token in the text
def get_entropy(text):
    assert args.openai_model is None, "get_entropy not implemented for OpenAI models"

    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:, :-1]
        neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        return -neg_entropy.sum(-1).mean().item()


def get_roc_metrics(real_preds, sample_preds):
    fpr, tpr, _ = roc_curve(
        [0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds
    )
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)


def get_precision_recall_metrics(real_preds, sample_preds):
    precision, recall, _ = precision_recall_curve(
        [0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds
    )
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)


def run_baseline_threshold_experiment(criterion_fn, name, n_samples=500):
    results = []
    for batch in tqdm.tqdm(
        range(n_samples // batch_size), desc=f"Computing {name} criterion"
    ):
        original_text = data["original"][batch * batch_size : (batch + 1) * batch_size]
        sampled_text = data["sampled"][batch * batch_size : (batch + 1) * batch_size]

        for idx in range(len(original_text)):
            results.append(
                {
                    "original": original_text[idx],
                    "original_crit": criterion_fn(original_text[idx]),
                    "sampled": sampled_text[idx],
                    "sampled_crit": criterion_fn(sampled_text[idx]),
                }
            )

    # compute prediction scores for real/sampled passages
    predictions = {
        "real": [x["original_crit"] for x in results],
        "samples": [x["sampled_crit"] for x in results],
    }

    fpr, tpr, roc_auc = get_roc_metrics(predictions["real"], predictions["samples"])
    p, r, pr_auc = get_precision_recall_metrics(
        predictions["real"], predictions["samples"]
    )
    print(f"{name}_threshold ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
    return {
        "name": f"{name}_threshold",
        "predictions": predictions,
        "info": {
            "n_samples": n_samples,
        },
        "raw_results": results,
        "metrics": {
            "roc_auc": roc_auc,
            "fpr": fpr,
            "tpr": tpr,
        },
        "pr_metrics": {
            "pr_auc": pr_auc,
            "precision": p,
            "recall": r,
        },
        "loss": 1 - pr_auc,
    }


# strip newlines from each example; replace one or more newlines with a single space
def strip_newlines(text):
    return " ".join(text.split())


# trim to shorter length
def trim_to_shorter_length(texta, textb):
    # truncate to shorter of o and s
    shorter_length = min(len(texta.split(" ")), len(textb.split(" ")))
    texta = " ".join(texta.split(" ")[:shorter_length])
    textb = " ".join(textb.split(" ")[:shorter_length])
    return texta, textb


def truncate_to_substring(text, substring, idx_occurrence):
    # truncate everything after the idx_occurrence occurrence of substring
    assert idx_occurrence > 0, "idx_occurrence must be > 0"
    idx = -1
    for _ in range(idx_occurrence):
        idx = text.find(substring, idx + 1)
        if idx == -1:
            return text
    return text[:idx]


def generate_samples(raw_data, batch_size, dataset=None):
    data = {
        "original": [],
        "sampled": [],
    }

    for batch in range(len(raw_data) // batch_size):
        print("Generating samples for batch", batch, "of", len(raw_data) // batch_size)
        original_text = raw_data[batch * batch_size : (batch + 1) * batch_size]
        sampled_text = sample_from_model(
            original_text,
            min_words=30 if dataset in ["pubmed"] else 50,
            prompt_tokens=20 if args.dataset in ["HC3"] else 30,
        )

        for o, s in zip(original_text, sampled_text):
            if dataset == "pubmed":
                s = truncate_to_substring(s, "Question:", 2)
                o = o.replace(custom_datasets.SEPARATOR, " ")

            o, s = trim_to_shorter_length(o, s)

            # add to the data
            data["original"].append(o)
            data["sampled"].append(s)

    if args.pre_perturb_pct > 0:
        print(
            f"APPLYING {args.pre_perturb_pct}, {args.pre_perturb_span_length} PRE-PERTURBATIONS"
        )
        load_mask_model()
        data["sampled"] = perturb_texts(
            data["sampled"],
            args.pre_perturb_span_length,
            args.pre_perturb_pct,
            ceil_pct=True,
        )
        load_base_model()

    return data


def generate_data(dataset, data):
    # get unique examples, strip whitespace, and remove newlines
    # then take just the long examples, shuffle, take the first 5,000 to tokenize to save time
    # then take just the examples that are <= 512 tokens (for the mask model)
    # then generate n_samples samples

    # remove duplicates from the data
    data = list(dict.fromkeys(data))  # deterministic, as opposed to set()

    # strip whitespace around each example
    data = [x.strip() for x in data]

    # remove newlines from each example
    data = [strip_newlines(x) for x in data]

    # try to keep only examples with > 250 words
    if dataset in ["writing", "squad", "xsum"]:
        long_data = [x for x in data if 300 > len(x.split()) > 150]
        if len(long_data) > 0:
            data = long_data
    else:
        assert False, f"Not approved dataset {dataset}"

    # keep only examples with <= 512 tokens according to mask_tokenizer
    # this step has the extra effect of removing examples with low-quality/garbage content
    tokenized_data = preproc_tokenizer(data)
    data = [x for x, y in zip(data, tokenized_data["input_ids"]) if len(y) <= 512]

    # print stats about remainining data
    print(f"Total number of samples: {len(data)}")
    print(f"Average number of words: {np.mean([len(x.split()) for x in data])}")

    return generate_samples(data[:n_samples], batch_size=batch_size, dataset=dataset)


def generate_fake_and_combine_real(dataset, data):
    ## Keep only the longer examples (more than 150 tokens)
    long_data = [x for x in data if len(x.split()) > 150]
    if len(long_data) > 0:
        data = long_data

    # keep only examples with <= 512 tokens according to mask_tokenizer
    # this step has the extra effect of removing examples with low-quality/garbage content
    tokenized_data = preproc_tokenizer(data)
    ## Keep only the examples with less than 512 tokens, according to the mask_tokenizer
    data = [x for x, y in zip(data, tokenized_data["input_ids"]) if len(y) <= 512]

    # print stats about remainining data
    print(f"Total number of samples: {len(data)}")
    print(f"Average number of words: {np.mean([len(x.split()) for x in data])}")

    return generate_samples(data[:n_samples], batch_size=batch_size, dataset=dataset)


def load_HC3(data_o, long_train_threshold_low=150, long_train_threshold_high=512):
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


def load_dataset(data_o, long_train_threshold_low=0, long_train_threshold_high=512):
    def construct_train_sentences(raw_sentences):
        train_sentences = []
        concat_sentence = ""
        for sentence in raw_sentences:
            if len(sentence.split(".")) > 2:
                train_sentences.append(sentence)
            else:
                concat_sentence += sentence + " "
                if len(concat_sentence.split(".")) > 2:
                    train_sentences.append(concat_sentence)
                    concat_sentence = ""
        return train_sentences

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
    long_train_real = construct_train_sentences(train_real)
    long_train_generated = construct_train_sentences(train_generated)

    # long_train_real = [x for x in train_real if len(x.split()) > long_train_threshold_low]
    # long_train_generated = [x for x in train_generated if len(x.split()) > long_train_threshold_low]

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


def load_base_model_and_tokenizer(name):
    if args.openai_model is None:
        print(
            f"Loading BASE model {name}..."
        )  # print(f'Loading BASE model {args.base_model_name}...')
        base_model_kwargs = {}
        if "gpt-j" in name or "neox" in name:
            base_model_kwargs.update(dict(torch_dtype=torch.float16))
        if "gpt-j" in name:
            base_model_kwargs.update(dict(revision="float16"))
        # base_model = transformers.AutoModelForCausalLM.from_pretrained(name, **base_model_kwargs, cache_dir=cache_dir)
        base_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path_dit[name]
        )
    else:
        base_model = None

    optional_tok_kwargs = {}
    if "facebook/opt-" in name:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs["fast"] = False
    if args.dataset in ["pubmed"]:
        optional_tok_kwargs["padding_side"] = "left"
    # base_tokenizer = transformers.AutoTokenizer.from_pretrained(name, **optional_tok_kwargs, cache_dir=cache_dir)
    base_tokenizer = transformers.AutoTokenizer.from_pretrained(model_path_dit[name])
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
    ## Load the model and tokenizer if the model is supported
    if name in [
        "roberta-base",
        "roberta-base-openai-detector",
        "Hello-SimpleAI/chatgpt-detector-roberta",
    ]:
        # base_tokenizer = RobertaTokenizer.from_pretrained(name, cache_dir=cache_dir)
        # base_model = RobertaModel.from_pretrained(name, output_hidden_states=True, cache_dir=cache_dir)
        base_tokenizer = RobertaTokenizer.from_pretrained(model_path_dit[name])
        base_model = RobertaModel.from_pretrained(
            model_path_dit[name], output_hidden_states=True
        )
    elif name in ["tiiuae/falcon-rw-1b"]:
        config = transformers.AutoConfig.from_pretrained(model_path_dit[name])
        config.output_hidden_states = True
        # config.hidden_size = 768
        # config.gradient_checkpointing = True
        base_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path_dit[name], config=config
        )
    if args.base_half:
        base_model = base_model.half()
    return base_model, base_tokenizer


def eval_supervised(data, model, pos_bit=0):
    print(f"Beginning supervised evaluation with {model}...")
    detector = transformers.AutoModelForSequenceClassification.from_pretrained(
        model, cache_dir=cache_dir
    ).to(DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)

    real, fake = data["original"], data["sampled"]
    with torch.no_grad():
        # get predictions for real
        real_preds = []
        for batch in tqdm.tqdm(range(len(real) // batch_size), desc="Evaluating real"):
            batch_real = real[batch * batch_size : (batch + 1) * batch_size]
            batch_real = tokenizer(
                batch_real,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(DEVICE)
            real_preds.extend(
                detector(**batch_real).logits.softmax(-1)[:, pos_bit].tolist()
            )

        # get predictions for fake
        fake_preds = []
        for batch in tqdm.tqdm(range(len(fake) // batch_size), desc="Evaluating fake"):
            batch_fake = fake[batch * batch_size : (batch + 1) * batch_size]
            batch_fake = tokenizer(
                batch_fake,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(DEVICE)
            fake_preds.extend(
                detector(**batch_fake).logits.softmax(-1)[:, pos_bit].tolist()
            )

    predictions = {
        "real": real_preds,
        "samples": fake_preds,
    }

    fpr, tpr, roc_auc = get_roc_metrics(real_preds, fake_preds)
    p, r, pr_auc = get_precision_recall_metrics(real_preds, fake_preds)
    print(f"{model} ROC AUC: {roc_auc}, PR AUC: {pr_auc}")

    # free GPU memory
    del detector
    torch.cuda.empty_cache()

    return {
        "name": model,
        "predictions": predictions,
        "info": {
            "n_samples": n_samples,
        },
        "metrics": {
            "roc_auc": roc_auc,
            "fpr": fpr,
            "tpr": tpr,
        },
        "pr_metrics": {
            "pr_auc": pr_auc,
            "precision": p,
            "recall": r,
        },
        "loss": 1 - pr_auc,
    }


def fea_get(texts_ls, max_length=300, print_fea_dim=True):
    with torch.no_grad():
        real_feas = []
        for batch in range(math.ceil(len(texts_ls) / batch_size)):

            ## Slice the texts into batches and tokenize them
            batch_real = texts_ls[
                batch * batch_size : min((batch + 1) * batch_size, len(texts_ls))
            ]
            inputs = base_tokenizer(
                batch_real,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(DEVICE)

            if args.base_model_name not in [
                "roberta-base",
                "roberta-base-openai-detector",
                "Hello-SimpleAI/chatgpt-detector-roberta",
                "tiiuae/falcon-rw-1b",
            ]:
                if args.mask_flag and not args.test_flag:
                    input_ids = inputs["input_ids"]
                    mask_ratio = 0.15
                    mask_positions = (
                        (torch.rand_like(input_ids.float()) < mask_ratio)
                        & (input_ids != base_tokenizer.cls_token_id)
                        & (input_ids != base_tokenizer.sep_token_id)
                    )
                    mask_token_id = base_tokenizer.mask_token_id
                    input_ids[mask_positions] = mask_token_id
                    inputs = {
                        "input_ids": input_ids,
                        "attention_mask": inputs["attention_mask"],
                    }
                    hidden_states = base_model.transformer(**inputs)[0]
            elif args.base_model_name == "tiiuae/falcon-rw-1b":
                outputs = base_model(**inputs)
                if outputs.hidden_states is not None:
                    hidden_states = outputs.hidden_states[-1]
                else:
                    raise ValueError(
                        f"The model {args.base_model_name} does not have hidden states"
                    )
            else:
                ## If base_model_name is not supported, then directly use the base_model to get the hidden states
                outputs = base_model(**inputs)
                hidden_states_all = outputs[2]
                hidden_states = hidden_states_all[-1]

            ## Apply the attention mask to the hidden states
            token_mask_10 = inputs["attention_mask"].unsqueeze(-1)
            hidden_states_mask_10 = hidden_states * token_mask_10
            # if args.base_model_name == 'tiiuae/falcon-rw-1b':
            #     hidden_states_mask_10, _ = torch.topk(hidden_states_mask_10, 768, dim=-1)

            ## Append the hidden states to the real_feas list
            real_feas.append(hidden_states_mask_10.to("cpu"))

            # remove the hidden states from memory
            del hidden_states_mask_10

        real_feas_tensor = torch.cat(real_feas, dim=0)
        if print_fea_dim:
            print("Feature dim:", real_feas_tensor.shape)

    return real_feas_tensor.half() if args.base_half else real_feas_tensor


def avg_value(all_auroc_list):
    # if len(all_auroc_list) <= 2:
    #     return 0.0, 0.0

    # Find the index of the maximum and minimum values
    # max_index = all_auroc_list.index(max(all_auroc_list))
    # min_index = all_auroc_list.index(min(all_auroc_list))

    # # Remove the maximum and minimum values
    # filtered_list = [auroc for i, auroc in enumerate(all_auroc_list) if i != max_index and i != min_index]
    filtered_list = all_auroc_list

    # Calculate the mean of the remaining values
    avg_auroc = np.round(sum(filtered_list) / len(filtered_list), 6)

    # Calculate the standard deviation of the remaining values
    std_auroc = np.round(np.std(filtered_list), 6) if len(filtered_list) > 1 else "N/A"

    return avg_auroc, std_auroc


def process_p_values_and_labels_odd(answer_labels, results_list):
    # Flatten sublists and transform p-values
    flat_p_values = [1 - p for sublist in results_list for p in sublist]

    # Calculate expansion ratio
    total_length = len(flat_p_values)
    label_length = len(answer_labels)

    # Calculate the number of times each label needs to be repeated
    repeat_counts = [total_length // label_length] * label_length
    remainder = total_length % label_length
    for i in range(remainder):
        repeat_counts[i] += 1

    # extension tag
    expanded_labels = np.concatenate(
        [np.repeat(label, count) for label, count in zip(answer_labels, repeat_counts)]
    )

    # Calculate AUROC
    all_auroc = metrics.roc_auc_score(expanded_labels, flat_p_values)
    fpr, tpr, thresholds = metrics.roc_curve(expanded_labels, flat_p_values)
    all_avg_fpr = np.mean(fpr)
    all_avg_tpr = np.mean(tpr)
    all_std_fpr = np.std(fpr)
    all_std_tpr = np.std(tpr)

    # Print results
    print(
        "ALL_AUROC: {:.4f}; ALL_AVG_FPR: {:.4f}; ALL_STD_FPR: {:.4f}; ALL_AVG_TPR: {:.4f}; ALL_STD_TPR: {:.4f}".format(
            all_auroc, all_avg_fpr, all_std_fpr, all_avg_tpr, all_std_tpr
        )
    )

    return all_auroc, all_avg_fpr, all_std_fpr, all_avg_tpr, all_std_tpr


def process_p_values_and_labels(answer_labels, results_list):
    # Initialize AUROC list
    auroc_list = []
    fpr_list = []
    tpr_list = []

    # Make sure labels and result list lengths match
    assert len(answer_labels) == len(results_list)

    # Index used to record paired
    used_indices = set()

    # Process each pair of labels
    for i in range(len(answer_labels)):
        if i in used_indices:
            continue

        # current label
        current_label = answer_labels[i]

        # Find the index of the next unused opposite label
        for j in range(len(answer_labels)):
            if answer_labels[j] != current_label and j not in used_indices:
                # Get the p-value of the current pair
                p_values_0 = results_list[i] if current_label == 0 else results_list[j]
                p_values_1 = results_list[j] if current_label == 0 else results_list[i]

                # Convert p-value
                converted_p_values_0 = [1 - p for p in p_values_0]
                converted_p_values_1 = [1 - p for p in p_values_1]

                # Combine p-values ​​and labels
                combined_p_values = converted_p_values_0 + converted_p_values_1
                combined_labels = [0] * len(p_values_0) + [1] * len(p_values_1)

                # Calculate AUROC
                auroc = round(
                    metrics.roc_auc_score(combined_labels, combined_p_values), 6
                )
                fpr, tpr, thresholds = metrics.roc_curve(
                    combined_labels, combined_p_values
                )
                auroc_list.append(auroc)
                fpr_list.append(fpr)
                tpr_list.append(tpr)

                # Mark used indexes
                used_indices.update([i, j])
                break
    # Calculate average AUROC, FPR and TPR, and standard deviation
    fixed_length = 100  # Fixed number of sampling points
    interpolated_fpr = [
        np.interp(np.linspace(0, 1, fixed_length), np.linspace(0, 1, len(fpr)), fpr)
        for fpr in fpr_list
    ]
    interpolated_tpr = [
        np.interp(np.linspace(0, 1, fixed_length), np.linspace(0, 1, len(tpr)), tpr)
        for tpr in tpr_list
    ]

    # Flatten into a single list for avg_value
    flat_fpr_list = [value for arr in interpolated_fpr for value in arr]
    flat_tpr_list = [value for arr in interpolated_tpr for value in arr]

    # Calculate average AUROC, FPR and TPR, and standard deviation
    avg_auroc, std_auroc = avg_value(auroc_list)
    avg_fpr, std_fpr = avg_value(flat_fpr_list)
    avg_tpr, std_tpr = avg_value(flat_tpr_list)

    # Print results
    print(
        "AVG_AUROC: {:.4f}; STD_AUROC: {}; AVG_FPR: {:.4f}; STD_FPR: {}; AVG_TPR: {:.4f}; STD_TPR: {}".format(
            avg_auroc, std_auroc, avg_fpr, std_fpr, avg_tpr, std_tpr
        )
    )

    return auroc_list, flat_fpr_list, flat_tpr_list


class Logger(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        # Flush both the terminal and the log file
        self.terminal.flush()
        self.log.flush()


if __name__ == "__main__":

    ## Set the device to GPU if available
    DEVICE = "cuda:0"

    ## Suppress the warnings
    transformers.logging.set_verbosity_error()

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dataset",
        type=str,
        default="meta_HC3",
        help="HC3|SQuAD1|NarrativeQA|xsum|TruthfulQA",
    )
    parser.add_argument(
        "--target_datasets",
        type=str,
        nargs="+",
        default="HC3",
        help="HC3|SQuAD1|NarrativeQA|xsum|TruthfulQA, writing|squad",
    )
    parser.add_argument(
        "--ref_dataset", type=str, default="", help="HC3|SQuAD1|NarrativeQA|xsum"
    )
    parser.add_argument("--dataset_key", type=str, default="document")
    parser.add_argument(
        "--pct_words_masked", type=float, default=0.3
    )  # pct masked is actually pct_words_masked * (span_length / (span_length + 2 * buffer_size))
    parser.add_argument("--span_length", type=int, default=2)

    parser.add_argument("--n_perturbation_list", type=str, default="1,10")
    parser.add_argument("--n_perturbation_rounds", type=int, default=1)
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="roberta-base-openai-detector",
        help="gpt2-medium|roberta-base|roberta-base-openai-detector|tiiuae/falcon-rw-1b",
    )
    parser.add_argument(
        "--text_generated_model_name",
        type=str,
        nargs="+",
        default="gpt2",
        help="gpt2-medium|roberta-base",
    )
    parser.add_argument(
        "--scoring_model_name",
        type=str,
        nargs="+",
        default="gpt2",
        help="gpt2-medium|roberta-base",
    )
    parser.add_argument("--mask_filling_model_name", type=str, default="t5-large")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=20,
        help="number of perturbed texts in tqdm function",
    )
    parser.add_argument("--n_similarity_samples", type=int, default=20)
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--base_half", action="store_true")
    parser.add_argument("--do_top_k", action="store_true")
    parser.add_argument("--top_k", type=int, default=40)

    parser.add_argument(
        "--meta_lr", type=float, help="meta-level outer learning rate", default=0.0002
    )
    parser.add_argument(
        "--update_lr",
        type=float,
        help="task-level inner update learning rate",
        default=0.001,
    )
    parser.add_argument(
        "--update_step", type=int, help="task-level inner update steps", default=10
    )
    parser.add_argument("--meta_epochs", type=int, default=300)
    parser.add_argument("--n_way", type=int, default=1)
    parser.add_argument("--meta_bs", type=int, default=10)
    parser.add_argument("--k_shot", type=int, default=200)
    parser.add_argument("--k_query", type=int, default=200)

    parser.add_argument("--all_par_num", type=int, default=5000)
    parser.add_argument("--n_samples", type=int, default=1500)
    parser.add_argument("--train_real_num", type=int, default=500)
    parser.add_argument("--target_senten_num", type=int, default=1000)
    parser.add_argument("--train_real_length", type=int, default=0)
    parser.add_argument("--val_num", type=int, default=100)

    parser.add_argument("--reference_par_num", type=int, default=500)
    parser.add_argument("--test_par_num", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=50)

    parser.add_argument("--id_load", type=int, help="number of experiment")
    parser.add_argument("--meta_sigma_use", action="store_true")
    parser.add_argument("--no_meta_flag", action="store_true")
    parser.add_argument("--train_LN_flag", action="store_true")
    parser.add_argument("--meta_test_flag", action="store_true")
    parser.add_argument("--meta_best_model_flag", action="store_true")
    parser.add_argument("--test_flag", action="store_true")
    parser.add_argument("--one_senten_flag", action="store_true")
    parser.add_argument("--one_par_flag", action="store_true")
    parser.add_argument("--senten_par_flag", action="store_true")

    parser.add_argument("--two_sample_test", action="store_true")
    parser.add_argument("--mask_flag", action="store_true")
    parser.add_argument("--full_data", action="store_true")
    parser.add_argument("--num_mlp", type=int, default=0)
    parser.add_argument("--transformer_flag", action="store_true")
    parser.add_argument("--num_hidden_layers", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=200)
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--id", type=int, default=998, help="number of experiment")
    parser.add_argument("--epsilon", type=int, default=10, help="10 for imagenet")
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--sigma0", default=45, type=float, help="0.5 for imagenet")
    parser.add_argument("--sigma", default=30, type=float, help="100 for imagenet")
    parser.add_argument("--coeff_xy", default=2, type=float)
    parser.add_argument("--target_mlp_num", type=int, default=2)
    parser.add_argument("--target_mlp_lr", default=0.01, type=float)
    parser.add_argument("--trial_num", type=int, default=10)
    parser.add_argument("--seed_temp", type=int, default=0)

    parser.add_argument("--pretaining", action="store_true")
    parser.add_argument("--is_yy_zero", action="store_true")
    parser.add_argument("--is_xx_zero", action="store_true")
    parser.add_argument("--MMDO_flag", action="store_true")

    parser.add_argument("--do_top_p", action="store_true")
    parser.add_argument("--top_p", type=float, default=0.96)
    parser.add_argument("--output_name", type=str, default="")
    parser.add_argument("--openai_model", type=str, default=None)
    parser.add_argument("--openai_key", type=str)
    parser.add_argument("--baselines_only", action="store_true")
    parser.add_argument("--skip_baselines", action="store_true")
    parser.add_argument("--buffer_size", type=int, default=1)
    parser.add_argument("--mask_top_p", type=float, default=1.0)
    parser.add_argument("--pre_perturb_pct", type=float, default=0.0)
    parser.add_argument("--pre_perturb_span_length", type=int, default=5)
    parser.add_argument("--random_fills", action="store_true")
    parser.add_argument("--random_fills_tokens", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=".cache")
    parser.add_argument("--print_details", action="store_true")
    parser.add_argument("--output_test_text_file", action="store_true")

    ## Add extra arguments for the command line to determine the preffered metrics for saving the best model
    parser.add_argument(
        "--metric",
        type=str,
        choices=["auroc", "power"],
        default="power",
        help="auroc|power",
    )

    ## Add extra arguments for the command line for relative testing
    parser.add_argument("--relative_test", action="store_true")
    parser.add_argument("--relative_test_baseline", action="store_true")
    parser.add_argument("--relative_test_baseline_simple", action="store_true")
    parser.add_argument("--relative_test_extra_n_samples", type=int, default=0)
    parser.add_argument("--relative_test_alpha", type=float, default=0.05)
    parser.add_argument(
        "--relative_test_reference_mode",
        type=str,
        choices=["random", "all"],
        default="random",
        help="random|all",
    )
    parser.add_argument(
        "--relative_test_mode",
        type=str,
        choices=["normal", "permutation"],
        default="normal",
        help="normal|permutation",
    )
    parser.add_argument(
        "--relative_test_two_sample_mode",
        type=str,
        choices=["normal", "permutation"],
        default="normal",
        help="normal|permutation",
    )

    ## Add extra arguments for the command line to determine the plain text to be used for testing the model
    parser.add_argument(
        "--test_dataset", type=str, default=None, help="HC3|SQuAD1|TruthfulQA|CCNews"
    )
    parser.add_argument(
        "--test_dataset_answer",
        type=str,
        default="ChatGPT",
        help="ChatGPT|ChatGPT-turbo|GPT4|StableLM|BloomZ|ChatGLM|Dolly|Human",
    )
    parser.add_argument("--test_dataset_answer_mix_ratio", type=float, default=None)
    parser.add_argument("--test_text", type=str, default=None)
    parser.add_argument("--test_text_file", type=str, default=None)
    parser.add_argument("--test_text_split", action="store_true")
    parser.add_argument("--test_text_n_samples", type=int, default=None)
    parser.add_argument("--test_text_n_sample_tokens", type=int, default=None)
    parser.add_argument("--test_text_n_sample_rounds", type=int, default=None)
    parser.add_argument(
        "--test_dataset_attack", type=str, default="none", help="any|none"
    )
    parser.add_argument(
        "--raid_split",
        type=str,
        default="train",
        help="train|test|extra",
        choices=["train", "test", "extra"],
    )

    ## Add extra arguments for the command line to speed up reference data loading
    parser.add_argument("--faster", action="store_true")

    ## Add Binoculars comparison
    parser.add_argument("--binoculars", action="store_true")
    args = parser.parse_args()

    ## Check if test_dataset is set to non HC3 dataset while test_text or test_text_file is not None
    assert args.test_dataset == None or (
        args.test_text is None and args.test_text_file is None
    ), "test_dataset must not be set or None if test_text or test_text_file is not None"

    ## Check if the test_text_file exists
    if args.test_text_file is not None:
        assert os.path.exists(args.test_text_file), "test_text_file does not exist"

    ## test_text_n_samples and test_text_n_sample_rounds is only allowed if test_text_split is set to False
    assert not args.test_text_split or (
        args.test_text_n_samples is None and args.test_text_n_sample_rounds is None
    ), "test_text_n_samples and test_text_n_sample_rounds is only allowed if test_text_split is set to False"
    if args.test_text_n_sample_rounds is None:
        args.test_text_n_sample_rounds = 0

    if not args.relative_test:
        args.relative_test_extra_n_samples = 0

    assert args.test_dataset_attack == "none" or args.test_dataset in [
        "RAID",
        "Beemo",
        "DetectRL",
    ], "test_dataset_attack is only allowed for RAID, Beemo and DetectRL dataset"

    assert (
        args.test_dataset_answer != "Human"
        or args.test_dataset_answer_mix_ratio is None
    ), "test_dataset_answer must mix with other answers than Human"

    if args.faster and args.target_datasets != ["HC3"]:
        print(
            "Warning: faster flag is made mainly for HC3 dataset, it may not work well or cause unexpected behavior for other datasets"
        )

    ## Set folder name for saving the model and log to two_sample_test since we're training the model using two sample test
    PATH_exper = "two_sample_test"

    ## Path for saving best model and log
    model_path = f"./{PATH_exper}/HC3-{args.base_model_name}/{args.id}"

    ## Create the folder if it does not exist
    if not os.path.isdir(model_path):
        os.makedirs(model_path, exist_ok=True)
    ## Make stdout and stderr to be written to the file
    sys.stdout = (
        Logger(f"{model_path}/log.log")
        if args.print_details
        else open(model_path + "/log.log", "a")
    )
    print(args)
    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("==========", "Script Started:", current_time_str, "==========")

    API_TOKEN_COUNTER = 0
    all_auroc_list = []
    all_power_list = []

    test_auroc_list_real = []
    test_auroc_list_generated = []
    test_power_list_real = []
    test_power_list_generated = []
    relative_test_p_value_list = []
    relative_test_t_stat_list = []
    relative_test_power_list = []
    relative_test_result_list = []
    baseline_power_list = []
    baseline_result_list = []
    baseline_p_value_list = []
    baseline_mmd_list = []
    simple_relative_baseline_p_value_list = []
    simple_relative_baseline_t_stat_list = []
    simple_relative_baseline_power_list = []
    simple_relative_baseline_result_list = []
    relative_baseline_p_value_list = []
    relative_baseline_t_stat_list = []
    relative_baseline_power_list = []
    relative_baseline_result_list = []
    binoculars_p_value_list = []
    binoculars_t_stat_list = []
    binoculars_power_list = []
    binoculars_result_list = []

    baseline_time_list = []
    relative_baseline_time_list = []
    simple_relative_baseline_time_list = []
    binoculars_time_list = []
    relative_test_time_list = []

    if "xsum" in args.target_datasets:
        args.n_samples = args.n_samples + 200
    current_trial = 0

    ## create a list of 0s and 1s to define actual labels for each trial (only applicable if we're testing a dataset with known answers)
    if args.test_dataset_answer_mix_ratio is not None:
        num_zeros = int(args.trial_num * args.test_dataset_answer_mix_ratio)
        num_ones = args.trial_num - num_zeros
        answer_labels = [0] * num_zeros + [1] * num_ones
    else:
        if args.test_dataset != None:
            answer_labels = (
                [0] * args.trial_num
                if args.test_dataset_answer == "Human"
                else [1] * args.trial_num
            )
        else:
            answer_labels = []

    ## Create a list for storing test text for Binoculars
    fea_test_ori_list = []

    ## Create a cache for faster loading of reference data
    cache_fea_real = None
    cache_fea_generated = None
    cache_fea_train_real = None
    cache_fea_train_generated = None
    cache_fea_reference = None
    cache_val_real = None
    cache_val_generated = None
    cache_val_singe_real = None
    cache_val_singe_generated = None

    ## Sync the seed for all random number generators
    for seed in range(990, 990 + args.trial_num):
        random.seed(seed + args.seed_temp)
        np.random.seed(seed + args.seed_temp)
        torch.manual_seed(seed + args.seed_temp)
        torch.cuda.manual_seed(seed + args.seed_temp)
        torch.cuda.manual_seed_all(seed + args.seed_temp)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if args.test_dataset_answer_mix_ratio is not None:
            if seed == 990:
                random.shuffle(answer_labels)
                test_data_answer = args.test_dataset_answer
            current_trial_label = answer_labels[current_trial]
            args.test_dataset_answer = (
                "Human" if current_trial_label == 0 else test_data_answer
            )
        current_trial += 1
        current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print()  ## Print an empty line
        print(
            "----------",
            f"Start Time for Seed {seed} (Trial {current_trial}/{args.trial_num}):",
            current_time_str,
            "----------",
        )

        if args.train_batch_size >= args.target_senten_num:
            args.train_batch_size = args.target_senten_num

        (
            print("You Are Testing!")
            if args.test_flag
            else print("You Are Training with Metric: " + args.metric + "!")
        )
        if args.openai_model is not None:
            assert (
                args.openai_key is not None
            ), "Must provide OpenAI API key as --openai_key"
            openai.api_key = args.openai_key

        START_DATE = datetime.datetime.now().strftime("%Y-%m-%d")
        START_TIME = datetime.datetime.now().strftime("%H-%M-%S-%f")

        # define SAVE_FOLDER as the timestamp - base model name - mask filling model name
        # create it if it doesn't exist
        precision_string = "int8" if args.int8 else ("fp16" if args.half else "fp32")
        sampling_string = (
            "top_k" if args.do_top_k else ("top_p" if args.do_top_p else "temp")
        )
        output_subfolder = f"{args.output_name}/" if args.output_name else ""
        if args.openai_model is None:
            base_model_name = args.base_model_name.replace("/", "_")
        else:
            base_model_name = "openai-" + args.openai_model.replace("/", "_")
        scoring_model_string = (
            f"-{args.scoring_model_name}" if args.scoring_model_name else ""
        ).replace("/", "_")
        ## Save results and args to a temporary folder, different from the model and log saving folder
        SAVE_FOLDER = f"tmp_results/{output_subfolder}{base_model_name}{scoring_model_string}-{args.mask_filling_model_name}-{sampling_string}/{START_DATE}-{START_TIME}-{precision_string}-{args.pct_words_masked}-{args.n_perturbation_rounds}-{args.dataset}-{args.n_samples}"
        if not os.path.exists(SAVE_FOLDER):
            os.makedirs(SAVE_FOLDER)
        print(f"Saving results to absolute path: {os.path.abspath(SAVE_FOLDER)}")

        # write args to file
        with open(os.path.join(SAVE_FOLDER, "args.json"), "w") as f:
            json.dump(args.__dict__, f, indent=4)

        mask_filling_model_name = args.mask_filling_model_name
        n_samples = args.n_samples
        batch_size = args.batch_size
        n_perturbation_list = [int(x) for x in args.n_perturbation_list.split(",")]
        n_perturbation_rounds = args.n_perturbation_rounds
        n_similarity_samples = args.n_similarity_samples

        cache_dir = args.cache_dir
        os.environ["XDG_CACHE_HOME"] = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        print(f"Using cache dir {cache_dir}")

        # GPT2_TOKENIZER = transformers.GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)
        GPT2_TOKENIZER = transformers.GPT2Tokenizer.from_pretrained(
            model_path_dit["gpt2"]
        )

        # mask filling t5 model
        ## Use the mask filling model to do text perturbations, to get a sense of the model's robustness
        if not args.baselines_only and not args.random_fills:
            int8_kwargs = {}
            half_kwargs = {}
            ## Adjust the model and tokenizer to use given precision according to the command line arguments
            if args.int8:
                int8_kwargs = dict(
                    load_in_8bit=True, device_map="auto", torch_dtype=torch.bfloat16
                )
            elif args.half:
                half_kwargs = dict(torch_dtype=torch.bfloat16)
            print(f"Loading mask filling model {mask_filling_model_name}...")
            # mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(mask_filling_model_name, **int8_kwargs, **half_kwargs, cache_dir=cache_dir)
            mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                model_path_dit[mask_filling_model_name], **int8_kwargs, **half_kwargs
            )
            try:
                n_positions = mask_model.config.n_positions
            except AttributeError:
                n_positions = 512
        else:
            n_positions = 512
        ## Load preprocessing and mask filling tokenizers from the given model names
        # preproc_tokenizer = transformers.AutoTokenizer.from_pretrained('t5-small', model_max_length=512, cache_dir=cache_dir)
        # mask_tokenizer = transformers.AutoTokenizer.from_pretrained(mask_filling_model_name, model_max_length=n_positions, cache_dir=cache_dir)
        preproc_tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path_dit["t5-small"], model_max_length=512
        )
        mask_tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path_dit[mask_filling_model_name], model_max_length=n_positions
        )
        ## If the dataset is in English or German, use the mask tokenizer for preprocessing
        if args.dataset in ["english", "german"]:
            preproc_tokenizer = mask_tokenizer

        # Target Dataset
        fea_train_real_ls = []
        fea_train_generated_ls = []
        fea_reference_ls = []
        fea_real_ls = []
        fea_generated_ls = []
        val_real_ls = []
        val_generated_ls = []
        fea_test_ls = []
        fea_test_ori_ls = []
        fea_test_single_ls = []

        val_sing_real_ls = []
        val_sing_generated_ls = []
        for target_dataset in args.target_datasets:
            ## Load the target dataset from the given dataset name, if supported (xsum, squad, writing, HC3)
            print(f"Loading dataset {target_dataset}...")
            if target_dataset in ["xsum", "squad", "writing"]:
                dataset_key = "document" if target_dataset != "squad" else "context"
                # load data
                if target_dataset in custom_datasets.DATASETS:
                    train_real_o = custom_datasets.load(target_dataset, cache_dir)
                else:
                    train_real_o = datasets.load_dataset(
                        target_dataset, split="train", cache_dir=cache_dir
                    )[dataset_key]

            elif target_dataset in ["HC3", "TruthfulQA", "SQuAD1"]:
                data_o = dataset_loader.load(
                    target_dataset,
                    args.cache_dir,
                    train_ratio=0.8 if not args.faster else 1.0,
                )  # train 38154;   test 9538
                train_real_o = [
                    text
                    for text, label in zip(
                        data_o["train"]["text"], data_o["train"]["label"]
                    )
                    if label == 0
                ]  # 19077[text,text]
            elif target_dataset in ["RAID"]:
                data_o = dataset_loader.load(
                    target_dataset + "_LLMs",
                    args.cache_dir,
                    LLM_name="train",
                    raid_split="train",
                    train_ratio=0.1,
                )
                train_real_o = [
                    text
                    for text, label in zip(
                        data_o["train"]["text"], data_o["train"]["label"]
                    )
                    if label == 0
                ]  # 19077[text,text]
            else:
                assert False, f"Unsupported dataset {target_dataset}"
            ## Load the test dataset from the given dataset name, if supported (HC3, TruthfulQA, SQuAD1)
            if args.test_dataset is not None:
                print(
                    f"Loading test dataset {args.test_dataset}, with answers from {args.test_dataset_answer}..."
                )
                if args.test_dataset in ["CCNews"]:
                    assert args.test_dataset_answer in [
                        "Human"
                    ], "Unsupported test dataset answer"
                    test_o = dataset_loader.load(
                        args.test_dataset, args.cache_dir, train_ratio=0.1
                    )
                    test_o = [
                        text
                        for text, label in zip(
                            test_o["test"]["text"], test_o["test"]["label"]
                        )
                        if label == 0
                    ]
                elif args.test_dataset in [
                    "HC3",
                    "TruthfulQA",
                    "TruthfulQA_adv1",
                    "TruthfulQA_adv2",
                    "SQuAD1",
                    "RAID",
                    "Beemo",
                    "DetectRL",
                ]:
                    if (
                        args.test_dataset_answer == "ChatGPT"
                        and args.test_dataset not in ["RAID", "Beemo", "DetectRL"]
                    ):
                        test_o = dataset_loader.load(
                            args.test_dataset, args.cache_dir, train_ratio=0.1
                        )
                        test_o = [
                            text
                            for text, label in zip(
                                test_o["test"]["text"], test_o["test"]["label"]
                            )
                            if label == 1
                        ]  ## Label = 1: chatgpt_answer, Label = 0: human_answer
                    elif (
                        args.test_dataset_answer == "Human"
                        and args.test_dataset == "HC3"
                    ):
                        test_o = dataset_loader.load(
                            args.test_dataset, args.cache_dir, train_ratio=0.1
                        )
                        test_o = [
                            text
                            for text, label in zip(
                                test_o["test"]["text"], test_o["test"]["label"]
                            )
                            if label == 0
                        ]  ## Label = 1: llm_answer, Label = 0: human_answer
                    else:
                        assert args.test_dataset in [
                            "TruthfulQA",
                            "SQuAD1",
                            "RAID",
                            "Beemo",
                            "DetectRL",
                        ], "Unsupported test dataset for given test dataset LLM"
                        if (
                            args.test_dataset_answer.lower() == "human"
                            and args.test_dataset in ["DetectRL"]
                        ):
                            test_o = dataset_loader.load(
                                args.test_dataset + "_LLMs",
                                args.cache_dir,  # add attack for DetectRL even when test_dataset_answer is human
                                LLM_name="human",
                                raid_split=args.raid_split,
                                attack=args.test_dataset_attack,
                                train_ratio=0.1,
                                max_tokens=args.test_text_n_sample_tokens,
                            )
                            test_o = [
                                text
                                for text, label in zip(
                                    test_o["test"]["text"], test_o["test"]["label"]
                                )
                                if label == 0
                            ]  ## Label = 1: llm_answer, Label = 0: human_answer
                        elif (
                            args.test_dataset_answer.lower() == "human"
                            and args.test_dataset in ["RAID", "Beemo"]
                        ):
                            test_o = dataset_loader.load(
                                args.test_dataset + "_LLMs",
                                args.cache_dir,
                                LLM_name="human",
                                raid_split=args.raid_split,
                                train_ratio=0.1,
                                max_tokens=args.test_text_n_sample_tokens,
                            )
                            test_o = [
                                text
                                for text, label in zip(
                                    test_o["test"]["text"], test_o["test"]["label"]
                                )
                                if label == 0
                            ]  ## Label = 1: llm_answer, Label = 0: human_answer
                        elif (
                            args.test_dataset_answer.lower() == "machine"
                            and args.test_dataset in ["RAID", "Beemo", "DetectRL"]
                        ):
                            test_o = dataset_loader.load(
                                args.test_dataset + "_LLMs",
                                args.cache_dir,
                                LLM_name="machine",
                                raid_split=args.raid_split,
                                attack=args.test_dataset_attack,
                                train_ratio=0.1,
                                max_tokens=args.test_text_n_sample_tokens,
                            )
                            test_o = [
                                text
                                for text, label in zip(
                                    test_o["test"]["text"], test_o["test"]["label"]
                                )
                                if label == 1
                            ]  ## Label = 1: llm_answer, Label = 0: human_answer
                        elif args.test_dataset_answer == "Human":
                            test_o = dataset_loader.load(
                                args.test_dataset + "_LLMs",
                                args.cache_dir,
                                LLM_name="ChatGPT",
                                train_ratio=0.1,
                            )
                            test_o = [
                                text
                                for text, label in zip(
                                    test_o["test"]["text"], test_o["test"]["label"]
                                )
                                if label == 0
                            ]  ## Label = 1: llm_answer, Label = 0: human_answer
                        else:
                            test_o = dataset_loader.load(
                                args.test_dataset + "_LLMs",
                                args.cache_dir,
                                LLM_name=args.test_dataset_answer,
                                train_ratio=0.1,
                            )
                            test_o = [
                                text
                                for text, label in zip(
                                    test_o["test"]["text"], test_o["test"]["label"]
                                )
                                if label == 1
                            ]  ## Label = 1: llm_answer, Label = 0: human_answer
                else:
                    assert False, f"Unsupported test dataset {args.test_dataset}"

            ## Load the model that the MGTs were generated from
            for text_generated_model_name in args.text_generated_model_name:

                if text_generated_model_name in ["EleutherAI/gpt-neo-2.7B"]:
                    batch_size = 20

                elif text_generated_model_name in [
                    "nomic-ai/gpt4all-j",
                    "EleutherAI/gpt-j-6b",
                ]:
                    batch_size = 8

                elif text_generated_model_name in ["EleutherAI/gpt-neo-125m"]:
                    batch_size = 25
                    n_samples = args.n_samples + 300
                else:
                    batch_size = args.batch_size
                    n_samples = args.n_samples

                if text_generated_model_name == "chatGPT":
                    assert target_dataset in ["HC3", "TruthfulQA", "SQuAD1", "RAID"]

                if text_generated_model_name != "chatGPT":
                    base_model, base_tokenizer = load_base_model_and_tokenizer(
                        text_generated_model_name
                    )
                    load_base_model()

                if target_dataset in ["HC3", "TruthfulQA", "SQuAD1", "RAID"]:
                    if text_generated_model_name != "chatGPT":
                        data = generate_fake_and_combine_real(
                            target_dataset, train_real_o
                        )  # data['original'] 2500
                    else:
                        if target_dataset not in ["HC3", "RAID"]:
                            data = load_dataset(data_o)
                            ## Since other datasets are having less number of samples, we need to change some constants
                            DEFAULT_TEST_LENGTH = 20
                            EXTRA_SAMPLES = 50
                            EXTRA_SINGLE_SAMPLES = 30
                        else:
                            data = load_HC3(data_o)
                            DEFAULT_TEST_LENGTH = 100
                            EXTRA_SAMPLES = 250
                            EXTRA_SINGLE_SAMPLES = 150
                else:
                    data = generate_data(target_dataset, train_real_o)

                if text_generated_model_name != "chatGPT":
                    del base_model
                    del base_tokenizer
                    torch.cuda.empty_cache()

                # generic generative model
                base_model, base_tokenizer = load_base_model_and_tokenizer(
                    args.base_model_name
                )
                load_base_model()

                # training data
                real = data[
                    "original"
                ]  # [:args.train_real_num]  len== n_samples, many sentences of words
                generated = data["sampled"]  # [:args.train_real_num]
                if args.two_sample_test:
                    nltk.download("punkt", quiet=True)
                    nltk.download("punkt_tab", quiet=True)
                    ## Tokenize the sentences and remove the first and last sentences
                    real_sent_token = [
                        (
                            nltk.sent_tokenize(text)[1:-1]
                            if target_dataset in ["HC3", "RAID"]
                            else nltk.sent_tokenize(text)
                        )
                        for text in real
                    ]
                    generated_sent_token = [
                        (
                            nltk.sent_tokenize(text)[1:-1]
                            if target_dataset in ["HC3", "RAID"]
                            else nltk.sent_tokenize(text)
                        )
                        for text in generated
                    ]
                    ## If test_text or test_text_file is provided, load and/or tokenize the text and remove the empty sentences
                    if args.test_text is not None:
                        test_sent_token = [nltk.sent_tokenize(args.test_text)]
                    elif args.test_text_file is not None:
                        with open(args.test_text_file, "r", encoding="utf-8") as file:
                            if args.test_text_split:
                                paragraphs = file.readlines()
                            else:
                                paragraphs = [file.read()]
                        test_sent_token = [
                            nltk.sent_tokenize(text)
                            for text in paragraphs
                            if text.strip()
                        ]
                    else:
                        ## If custom test dataset is provided, tokenize the sentences and remove the empty sentences
                        ## Only for non-HC3 datasets, if HC3, use the regular testing procedure
                        if args.test_dataset is not None:
                            if not args.test_text_split:
                                ## If not to split the text, concatenate the text
                                test_o = [" ".join(test_o)]
                            test_sent_token = [
                                nltk.sent_tokenize(text)
                                for text in test_o
                                if text.strip()
                            ]
                        else:
                            test_sent_token = [""]
                    ## Remove the empty sentences
                    if (
                        not args.faster or current_trial == 1
                    ):  ## If not faster mode or first trial
                        real = [text for text in real_sent_token if len(text) > 0]
                        generated = [
                            text for text in generated_sent_token if len(text) > 0
                        ]
                    custom_test = [text for text in test_sent_token if len(text) > 0]
                    ## Record the number of sentences in the real and generated data
                    train_length = args.target_senten_num  # 200
                    train_real_length = args.target_senten_num  # 200
                    if args.train_real_length > args.target_senten_num:
                        train_real_length = args.train_real_length
                    ## Split the paragraphs according to required length and convert them into sentences (ignore sentences less than 5 words), and put them into real and generated data for training
                    if (
                        not args.faster or current_trial == 1
                    ):  ## If not faster mode or first trial
                        train_real = [
                            sen
                            for pa in real[:train_real_length]
                            for sen in pa
                            if 5 < len(sen.split())
                        ]
                        train_generated = [
                            sen
                            for pa in generated[:train_length]
                            for sen in pa
                            if 5 < len(sen.split())
                        ]

                        real_data = real
                        generated_data = generated

                    if (
                        not args.faster or current_trial == 1
                    ):  ## If not faster mode or first trial
                        ## Put the remaining paragraphs into the temporary real and generated data for later use (only keep sentences with more than 5 words in any paragraph)
                        real_data_temp = [
                            [
                                sentence
                                for sentence in sublist
                                if len(sentence.split()) >= 5
                            ]
                            for sublist in real_data[train_real_length:]
                        ]
                        generated_data_temp = [
                            [
                                sentence
                                for sentence in sublist
                                if len(sentence.split()) >= 5
                            ]
                            for sublist in generated_data[train_length:]
                        ]
                    test_data_temp = [
                        [sentence for sentence in sublist if len(sentence.split()) >= 5]
                        for sublist in custom_test
                    ]

                    ## Keep the paragraphs with more than 5 sentences (for TruthfulQA and SQuAD1 datasets, most answers are less than 5 sentences, so keep 2)
                    if target_dataset in ["HC3", "RAID"]:
                        sentence_num = 5
                    else:
                        sentence_num = 2
                    if (
                        not args.faster or current_trial == 1
                    ):  ## If not faster mode or first trial
                        real_data_temp_seletced = [
                            pa_ls
                            for pa_ls in real_data_temp
                            if len(pa_ls) >= sentence_num
                        ]
                        generated_data_temp_seletced = [
                            pa_ls
                            for pa_ls in generated_data_temp
                            if len(pa_ls) >= sentence_num
                        ]
                        len_data = min(
                            len(real_data_temp_seletced),
                            len(generated_data_temp_seletced),
                        )
                    test_data_temp_seletced = [
                        pa_ls for pa_ls in test_data_temp if len(pa_ls) >= 2
                    ]

                    ## Cut the test data to the required number of samples
                    if not args.test_text_split and len(test_data_temp_seletced) == 1:
                        if (
                            not args.test_text_n_samples
                            or args.test_text_n_samples == -1
                        ):
                            args.test_text_n_samples = len(test_data_temp_seletced[0])
                        args.test_text_n_samples = max(
                            min(
                                args.test_text_n_samples,
                                len(test_data_temp_seletced[0]),
                            ),
                            2,
                        )
                        test_data_temp_seletced[0] = (
                            test_data_temp_seletced[0][: args.test_text_n_samples]
                            if args.test_text_n_samples
                            else test_data_temp_seletced[0]
                        )
                    else:
                        if (
                            not args.test_text_n_samples
                            or args.test_text_n_samples == -1
                        ):
                            args.test_text_n_samples = len(test_data_temp_seletced)
                        args.test_text_n_samples = max(
                            min(args.test_text_n_samples, len(test_data_temp_seletced)),
                            2,
                        )
                        test_data_temp_seletced = (
                            test_data_temp_seletced[: args.test_text_n_samples]
                            if args.test_text_n_samples
                            else test_data_temp_seletced
                        )

                    ## Cut the test data to the required number of tokens
                    if not args.test_text_split and len(test_data_temp_seletced) == 1:
                        if (
                            args.test_text_n_sample_tokens is not None
                            and args.test_text_n_sample_tokens != -1
                        ):
                            total_tokens = sum(
                                [len(sen.split()) for sen in test_data_temp_seletced[0]]
                            )
                            args.test_text_n_sample_tokens = max(
                                min(args.test_text_n_sample_tokens, total_tokens), 2
                            )
                            total_tokens = 0
                            for i in range(len(test_data_temp_seletced[0])):
                                total_tokens += len(
                                    test_data_temp_seletced[0][i].split()
                                )
                                if total_tokens >= args.test_text_n_sample_tokens:
                                    token_diff = (
                                        total_tokens - args.test_text_n_sample_tokens
                                    )
                                    test_data_temp_seletced[0] = (
                                        test_data_temp_seletced[0][: i + 1]
                                    )
                                    if token_diff > 0:
                                        test_data_temp_seletced[0][-1] = " ".join(
                                            test_data_temp_seletced[0][-1].split()[
                                                :token_diff
                                            ]
                                        )
                                        if not test_data_temp_seletced[0][-1].endswith(
                                            "."
                                        ):
                                            test_data_temp_seletced[0][-1] += "."
                                    break

                    ## Changed test length default 100->20, extra required number 250->50, single sentence extra number 150 -> 25
                    test_lenth = DEFAULT_TEST_LENGTH
                    if (
                        args.test_text is not None
                        or args.test_text_file is not None
                        or args.test_dataset is not None
                    ):
                        if args.relative_test_extra_n_samples == -1:
                            test_lenth = len_data - args.val_num - EXTRA_SAMPLES
                        else:
                            if args.test_text_split:
                                test_lenth = (
                                    len(test_data_temp_seletced)
                                    + args.relative_test_extra_n_samples
                                )
                            else:
                                count = 0
                                for i in range(len_data):
                                    count += min(
                                        len(real_data_temp_seletced[i]),
                                        len(generated_data_temp_seletced[i]),
                                    )
                                    if count >= len(test_data_temp_seletced[0]):
                                        if args.relative_test_extra_n_samples == -2:
                                            args.relative_test_extra_n_samples = i
                                        test_lenth = (
                                            i + args.relative_test_extra_n_samples + 1
                                        )
                                        break
                                if (
                                    count < len(test_data_temp_seletced[0])
                                    or test_lenth
                                    > len_data - args.val_num - EXTRA_SAMPLES
                                ):
                                    test_lenth = len_data - args.val_num - EXTRA_SAMPLES
                    ## Check if the length of the remaining data is enough for the required number of sentences
                    assert (
                        test_lenth > 0
                        and len_data >= args.val_num + test_lenth + EXTRA_SAMPLES
                    ), print(
                        f"Please reduce the args.target_senten_num:{args.target_senten_num}"
                    )

                else:
                    assert False, f"You should choose the way to construct the samples!"

                if args.test_flag:
                    ## If the test flag is set, only use the first 500 sentences for training
                    train_real = train_real[:500]
                    train_generated = train_generated[:500]

                if (
                    not args.faster or current_trial == 1
                ):  ## If not faster mode or first trial
                    ## Get the hidden states of the real and generated data
                    fea_train_real = fea_get(
                        train_real, max_length=args.max_length
                    )  # .to('cpu')
                    fea_train_generated = fea_get(
                        train_generated, max_length=args.max_length
                    )  # .to('cpu')
                    if args.faster:
                        cache_fea_train_real = fea_train_real
                        cache_fea_train_generated = fea_train_generated
                else:
                    fea_train_real = cache_fea_train_real
                    fea_train_generated = cache_fea_train_generated

                ## Append the hidden states to the real and generated data list
                fea_train_real_ls.append(fea_train_real)
                fea_train_generated_ls.append(fea_train_generated)

                test_lenth = test_lenth

                ## Slice the validation data from the remaining data (other than the training data)
                text_val_real = real_data_temp_seletced[: args.val_num]
                text_val_generated = generated_data_temp_seletced[: args.val_num]
                ## Slice the test data from the remaining data (other than the training and validation data)
                text_real = real_data_temp_seletced[
                    args.val_num : args.val_num + test_lenth
                ]
                text_generated = generated_data_temp_seletced[
                    args.val_num : args.val_num + test_lenth
                ]
                ## Slice the single sentence data from the remaining data (other than the training, validation, and test data)
                text_single_real = real_data_temp_seletced[
                    args.val_num
                    + test_lenth : args.val_num
                    + test_lenth
                    + EXTRA_SINGLE_SAMPLES
                ]
                text_single_generated = generated_data_temp_seletced[
                    args.val_num
                    + test_lenth : args.val_num
                    + test_lenth
                    + EXTRA_SINGLE_SAMPLES
                ]
                text_reference = real_data_temp_seletced[
                    args.val_num
                    + test_lenth
                    + EXTRA_SINGLE_SAMPLES : args.val_num
                    + test_lenth
                    + EXTRA_SAMPLES
                ]

                ## Flatten the paragraphs into sentences
                text_single_real_sen_ls = [sen for pa in text_single_real for sen in pa]
                text_single_generated_sen_ls = [
                    sen for pa in text_single_generated for sen in pa
                ]
                fea_test_single_sen_ls = [
                    sen for pa in test_data_temp_seletced for sen in pa
                ]
                text_reference = [sen for pa in text_reference for sen in pa]

                ## Get the hidden states of the validation, test, and single sentence data
                if not args.faster or current_trial == 1:
                    fea_real = [
                        fea_get(pa_ls, max_length=args.max_length, print_fea_dim=False)
                        for pa_ls in text_real
                    ]
                    fea_generated = [
                        fea_get(pa_ls, max_length=args.max_length, print_fea_dim=False)
                        for pa_ls in text_generated
                    ]
                    if args.faster:
                        cache_fea_real = fea_real
                        cache_fea_generated = fea_generated
                else:
                    fea_real = cache_fea_real
                    fea_generated = cache_fea_generated
                fea_test = (
                    [
                        fea_get(pa_ls, max_length=args.max_length, print_fea_dim=False)
                        for pa_ls in test_data_temp_seletced
                    ]
                    if len(test_data_temp_seletced) > 0
                    else []
                )
                fea_test_ori = (
                    [pa for pa in test_data_temp_seletced]
                    if len(test_data_temp_seletced) > 0
                    else []
                )

                if not args.faster or current_trial == 1:
                    val_real = [
                        fea_get(pa_ls, max_length=args.max_length, print_fea_dim=False)
                        for pa_ls in text_val_real
                    ]
                    val_generated = [
                        fea_get(pa_ls, max_length=args.max_length, print_fea_dim=False)
                        for pa_ls in text_val_generated
                    ]
                    if args.faster:
                        cache_val_real = val_real
                        cache_val_generated = val_generated
                else:
                    val_real = cache_val_real
                    val_generated = cache_val_generated

                fea_real_ls.extend(fea_real)
                fea_generated_ls.extend(fea_generated)
                val_real_ls.extend(val_real)
                val_generated_ls.extend(val_generated)
                fea_test_ls.extend(fea_test)
                fea_test_ori_ls.extend(fea_test_ori)

                if not args.faster or current_trial == 1:
                    fea_reference = fea_get(text_reference, max_length=args.max_length)
                    val_singe_real = fea_get(
                        text_single_real_sen_ls, max_length=args.max_length
                    )
                    val_singe_generated = fea_get(
                        text_single_generated_sen_ls, max_length=args.max_length
                    )
                    if args.faster:
                        cache_fea_reference = fea_reference
                        cache_val_singe_real = val_singe_real
                        cache_val_singe_generated = val_singe_generated
                else:
                    fea_reference = cache_fea_reference
                    val_singe_real = cache_val_singe_real
                    val_singe_generated = cache_val_singe_generated
                fea_test_single = (
                    fea_get(fea_test_single_sen_ls, max_length=args.max_length)
                    if len(fea_test_single_sen_ls) > 0
                    else []
                )

                fea_reference_ls.append(fea_reference)
                val_sing_real_ls.append(val_singe_real)
                val_sing_generated_ls.append(val_singe_generated)
                (
                    fea_test_single_ls.append(fea_test_single)
                    if len(fea_test_single_sen_ls) > 0
                    else []
                )

        ## Concatenate the hidden states of the real and generated data (for training) and shuffle them
        if not args.faster or current_trial == 1:  ## If not faster mode or first trial
            fea_train_real0 = torch.cat(fea_train_real_ls, dim=0)
        fea_train_real0 = fea_train_real0[
            np.random.permutation(fea_train_real0.shape[0])
        ]
        if not args.faster or current_trial == 1:  ## If not faster mode or first trial
            fea_train_generated0 = torch.cat(fea_train_generated_ls, dim=0)
        fea_train_generated0 = fea_train_generated0[
            np.random.permutation(fea_train_generated0.shape[0])
        ]

        ## Concatenate the hidden states of the reference data and real and generated data (for test and validation) and shuffle them
        if not args.faster or current_trial == 1:  ## If not faster mode or first trial
            fea_reference = torch.cat(fea_reference_ls, dim=0)[:500]
        fea_real = random.sample(fea_real_ls, len(fea_real_ls))
        fea_generated = random.sample(fea_generated_ls, len(fea_generated_ls))
        val_real = random.sample(val_real_ls, len(val_real_ls))
        val_generated = random.sample(val_generated_ls, len(val_generated_ls))
        fea_test = random.sample(fea_test_ls, len(fea_test_ls))

        if not args.faster or current_trial == 1:  ## If not faster mode or first trial
            val_sing_real = torch.cat(val_sing_real_ls, dim=0)
            val_sing_generated = torch.cat(val_sing_generated_ls, dim=0)
        val_sing_real = val_sing_real[np.random.permutation(val_sing_real.shape[0])][
            :1000
        ]
        val_sing_generated = val_sing_generated[
            np.random.permutation(val_sing_generated.shape[0])
        ][:1000]
        if not args.faster or current_trial == 1:  ## If not faster mode or first trial
            fea_test_single = (
                torch.cat(fea_test_single_ls, dim=0)
                if len(fea_test_single_ls) > 0
                else None
            )
        fea_test_single = (
            fea_test_single[np.random.permutation(fea_test_single.shape[0])][:1000]
            if fea_test_single is not None
            else None
        )

        ## Release the memory
        del fea_train_real_ls
        del fea_train_generated_ls
        del fea_test_ls
        del fea_reference_ls
        del fea_real_ls
        del fea_generated_ls
        del val_real_ls
        del val_generated_ls
        print("fea_train_real:", fea_train_real0.shape)
        print("fea_train_generated:", fea_train_generated0.shape)
        print("fea_reference:", fea_reference.shape)
        print("fea_real:", len(fea_real))
        print("fea_generated:", len(fea_generated))
        print("val_real:", len(val_real))
        print("val_generated:", len(val_generated))
        print("val_sing_real:", len(val_sing_real))
        print("val_sing_generated:", len(val_sing_generated))
        if (
            args.test_text is not None
            or args.test_text_file is not None
            or args.test_dataset is not None
        ):
            print("fea_test:", len(fea_test))
            print("fea_test_single:", len(fea_test_single))

        train_batch_size = args.train_batch_size
        auroc_list = []
        auroc_value_best = 0
        power_best = 0
        id = args.id

        def single_instance_test(
            epoch,
            test_lenth=2000,
            fea_real=fea_real,
            fea_generated=fea_generated,
            meta_save_model_flag="",
            test_flag=args.test_flag,
        ):
            ## Set the model to evaluation mode
            net.eval()
            global auroc_value_best
            with torch.no_grad():
                feature_ref_ls = []
                ## If the number of reference data is larger than the training batch size, then slice the reference data into batches and get the hidden states
                if len(fea_reference) > train_batch_size:
                    for batch in tqdm.tqdm(
                        range(len(fea_reference) // train_batch_size),
                        desc="Testing for deep MMD",
                    ):
                        feature_ref = net(
                            fea_reference[
                                batch
                                * train_batch_size : (batch + 1)
                                * train_batch_size
                            ].to("cuda")
                        )
                        feature_ref_ls.append(feature_ref)
                ## If the number of reference data is less than the training batch size, then get the hidden states directly
                else:
                    feature_ref = net(fea_reference.to("cuda"))
                    feature_ref_ls.append(feature_ref)
                ## Concatenate the hidden states of the reference data
                feature_ref = torch.cat(feature_ref_ls, dim=0)

                ## Function to get the hidden states of the real and generated data
                def get_feature_cln_ls(fea_real):
                    feature_cln_ls = []
                    if len(fea_real) > train_batch_size:
                        for batch in range(len(fea_real) // train_batch_size):
                            feature_cln = net(
                                fea_real[
                                    batch
                                    * train_batch_size : (batch + 1)
                                    * train_batch_size
                                ].to("cuda")
                            )
                            feature_cln_ls.append(feature_cln)
                    else:
                        feature_cln = net(fea_real.to("cuda"))
                        feature_cln_ls.append(feature_cln)
                    return feature_cln_ls

                ## Get the hidden states of the real and generated data
                feature_cln_ls = get_feature_cln_ls(fea_real)
                feature_adv_ls = get_feature_cln_ls(fea_generated)

                ## Concatenate the hidden states of the real and generated data
                feature_cln = torch.cat(feature_cln_ls, dim=0)
                feature_adv = torch.cat(feature_adv_ls, dim=0)

                ## Calculate the MMD value of the real and generated data
                dt_clean = MMD_batch2(
                    torch.cat([feature_ref, feature_cln], dim=0),
                    feature_ref.shape[0],
                    torch.cat(
                        [
                            fea_reference[: feature_ref.shape[0]].to("cuda"),
                            fea_real[: feature_cln.shape[0]].to("cuda"),
                        ],
                        dim=0,
                    ).view(feature_ref.shape[0] + feature_cln.shape[0], -1),
                    sigma,
                    sigma0_u,
                    ep,
                ).to("cpu")
                dt_adv = MMD_batch2(
                    torch.cat([feature_ref, feature_adv], dim=0),
                    feature_ref.shape[0],
                    torch.cat(
                        [
                            fea_reference[: feature_ref.shape[0]].to("cuda"),
                            fea_generated[: feature_adv.shape[0]].to("cuda"),
                        ],
                        dim=0,
                    ).view(feature_ref.shape[0] + feature_adv.shape[0], -1),
                    sigma,
                    sigma0_u,
                    ep,
                ).to("cpu")
                ## Plot the MMD value and calculate the AUROC value
                auroc_value = plot_mi(dt_clean, dt_adv)
                auroc_list.append(auroc_value)
                model_path = f"./{PATH_exper}/HC3-{args.base_model_name}/{id}"

                state = {
                    "net": net.state_dict(),
                    # 'epsilonOPT': epsilonOPT,
                    # 'sigmaOPT': sigmaOPT,
                    # 'sigma0OPT': sigma0OPT,
                    "sigma": sigma,
                    "sigma0_u": sigma0_u,
                    "ep": ep,
                }

                ## Save the model if the AUROC value is the best and the test flag is not set
                if not test_flag:
                    if not os.path.isdir(model_path):
                        os.makedirs(model_path, exist_ok=True)
                    # os.mkdir(model_path)
                    # if (epoch+1)%100==0:
                    # 	torch.save(state, model_path + '/'+ str(epoch) +'_ckpt.pth')
                    if auroc_value > auroc_value_best:
                        auroc_value_best = auroc_value
                        torch.save(
                            state,
                            model_path + "/" + meta_save_model_flag + "best_ckpt.pth",
                        )
                        print(
                            "Save the best model: auroc_value_best=", auroc_value_best
                        )
            # torch.save(state, model_path + '/'+ meta_save_model_flag +'last_ckpt.pth')
            return auroc_value

        def two_sample_test(
            epoch,
            test_lenth=2000,
            fea_real_ls=fea_real,
            fea_generated_ls=fea_generated,
            fea_test_ls=fea_generated,
            meta_save_model_flag="",
            test_flag=args.test_flag,
            N=100,
            relative_test=args.relative_test,
            return_p_and_stat=False,
            baseline=False,
            relative_test_baseline=False,
            relative_test_permutation=False,
            relative_test_baseline_simple=False,
        ):
            global power_best
            model_path = f"./{PATH_exper}/HC3-{args.base_model_name}/{id}"
            ## Set the model to evaluation mode
            net.eval()
            with torch.no_grad():

                ## Function to get the test power of the real and generated data pairs via N times of MMD test
                def mmd_two_sample(fea_real_ls, fea_generated_ls, N=100):
                    ## Cut the real and generated data to the same length (according to the minimum length of the real and generated data)
                    fea_real_ls = fea_real_ls[
                        : min(len(fea_real_ls), len(fea_generated_ls))
                    ]
                    fea_generated_ls = fea_generated_ls[
                        : min(len(fea_real_ls), len(fea_generated_ls))
                    ]

                    test_power_ls = []
                    mmd_value_ls = []

                    N_per = 50
                    alpha = 0.05
                    for i in range(len(fea_real_ls)):

                        # if i>100:
                        # 	break
                        fea_x_ori = fea_real_ls[i].to("cuda")
                        fea_y_ori = fea_generated_ls[i].to("cuda")
                        fea_x_ori = fea_x_ori[: min(len(fea_x_ori), len(fea_y_ori))]
                        fea_y_ori = fea_y_ori[: min(len(fea_x_ori), len(fea_y_ori))]
                        final_x = net(fea_x_ori)
                        final_y = net(fea_y_ori)
                        count_u = 0
                        for _ in range(N):
                            h_u, threshold_u, mmd_value_u = TST_MMD_u(
                                torch.cat([final_x, final_y], dim=0),
                                N_per,
                                final_x.shape[0],
                                torch.cat([fea_x_ori, fea_y_ori], dim=0).view(
                                    fea_x_ori.shape[0] + fea_y_ori.shape[0], -1
                                ),
                                sigma,
                                sigma0_u,
                                ep,
                                alpha,
                            )
                            count_u = count_u + h_u

                        test_power_ls.append(count_u / N)
                        mmd_value_ls.append(mmd_value_u)

                    return test_power_ls, mmd_value_ls

                def mmd_two_sample_baseline_permutation(
                    fea_test_ls, fea_real_ls, fea_generated_ls, N=100
                ):
                    if not args.test_text_split:
                        ## Concatenate the hidden states of the real and generated data
                        fea_real_ls = [torch.cat(fea_real_ls, dim=0)]
                        fea_generated_ls = [torch.cat(fea_generated_ls, dim=0)]
                    ## Cut the real and generated data to the same length (according to the minimum length of the real and generated data)
                    min_len = min(len(fea_real_ls), len(fea_generated_ls))
                    fea_real_ls = fea_real_ls[:min_len]
                    fea_generated_ls = fea_generated_ls[:min_len]

                    test_power_ls = []
                    mmd_value_ls = []
                    p_value_ls = []

                    N_per = 50

                    for i in range(len(fea_test_ls)):
                        fea_x_ori = fea_test_ls[i].to("cuda")
                        final_x = net(fea_x_ori)

                        for j in range(len(fea_real_ls)):
                            fea_y_ori = fea_real_ls[j].to("cuda")
                            fea_z_ori = fea_generated_ls[j].to("cuda")
                            fea_y_ori = fea_y_ori[: min(len(fea_z_ori), len(fea_y_ori))]
                            fea_z_ori = fea_z_ori[: min(len(fea_z_ori), len(fea_y_ori))]

                            final_y = net(fea_y_ori)
                            final_z = net(fea_z_ori)

                            final_y = final_y[: min(len(final_y), len(final_z))]
                            final_z = final_z[: min(len(final_y), len(final_z))]
                            count_u = 0
                            for _ in range(N):
                                final_real = torch.cat([final_x, final_y], dim=0)[
                                    : ((final_x.shape[0] + final_y.shape[0]) // 2) * 2
                                ]
                                final_generated = torch.cat([final_x, final_z], dim=0)[
                                    : ((final_x.shape[0] + final_z.shape[0]) // 2) * 2
                                ]
                                h_u_real, p_real, mmd_value_u_real = TST_MMD_u(
                                    final_real,
                                    N_per,
                                    (final_real.shape[0] // 2),
                                    final_real.view(final_real.shape[0], -1),
                                    sigma,
                                    sigma0_u,
                                    ep,
                                    args.relative_test_alpha,
                                )
                                ## interchanging the real and generated data
                                h_u_generated, p_generated, mmd_value_u_generated = (
                                    TST_MMD_u(
                                        final_generated,
                                        N_per,
                                        (final_generated.shape[0] // 2),
                                        final_generated.view(
                                            final_generated.shape[0], -1
                                        ),
                                        sigma,
                                        sigma0_u,
                                        ep,
                                        args.relative_test_alpha,
                                    )
                                )
                                ## take the smaller p-value and the corresponding h_u and mmd_value_u
                                p_value = p_real
                                h_u = h_u_real
                                mmd_u = mmd_value_u_real
                                if p_generated < p_real:
                                    p_value = abs(p_generated - 1)
                                    h_u = abs(h_u_generated - 1)
                                    mmd_u = mmd_value_u_generated
                                count_u = count_u + h_u
                                p_value_ls.append(p_value)
                                mmd_value_ls.append(mmd_u)

                            test_power_ls.append(count_u / N)

                    return test_power_ls, p_value_ls, mmd_value_ls

                def mmd_two_sample_baseline(
                    fea_test_ls, fea_real_ls, fea_generated_ls, N=100
                ):
                    ## Concatenate the hidden states of the real and generated data
                    fea_real_ls = torch.cat(fea_real_ls, dim=0).to("cuda")
                    fea_generated_ls = torch.cat(fea_generated_ls, dim=0).to("cuda")
                    ## Cut the real and generated data to the same length (according to the minimum length of the real and generated data)
                    min_len = min(len(fea_real_ls), len(fea_generated_ls))
                    fea_real_ls = fea_real_ls[:min_len]
                    fea_generated_ls = fea_generated_ls[:min_len]

                    test_power_ls = []
                    mmd_value_ls = []
                    p_value_ls = []

                    N_per = 50
                    for _ in range(max(args.test_text_n_sample_rounds, 1)):
                        ## If the reference data is not splitted, then randomly sample n samples from the reference data
                        if (
                            not args.test_text_split
                            and len(fea_test_ls) == 1
                            and args.test_text_n_sample_rounds
                        ):
                            ## Randomly sample n samples from the reference data tensor
                            fea_test_ls[0] = fea_test_ls[0][
                                torch.randperm(len(fea_test_ls[0]))[
                                    : args.test_text_n_samples
                                ]
                            ]
                        for i in range(len(fea_test_ls)):
                            fea_x_ori = fea_test_ls[i].to("cuda")
                            fea_y_ori = fea_real_ls[torch.randperm(len(fea_real_ls))]
                            fea_z_ori = fea_generated_ls[
                                torch.randperm(len(fea_generated_ls))
                            ]

                            min_len = min(
                                len(fea_x_ori), len(fea_y_ori), len(fea_z_ori)
                            )
                            fea_x_ori = fea_x_ori[:min_len]
                            fea_y_ori = fea_y_ori[:min_len]
                            fea_z_ori = fea_z_ori[:min_len]

                            final_x = net(fea_x_ori)
                            final_y = net(fea_y_ori)
                            final_z = net(fea_z_ori)
                            count_u = 0
                            for _ in range(N):
                                h_u_real, p_real, mmd_value_u_real = TST_MMD_u(
                                    torch.cat([final_y, final_x], dim=0),
                                    N_per,
                                    final_x.shape[0],
                                    torch.cat([fea_y_ori, fea_x_ori], dim=0).view(
                                        fea_y_ori.shape[0] + fea_x_ori.shape[0], -1
                                    ),
                                    sigma,
                                    sigma0_u,
                                    ep,
                                    args.relative_test_alpha,
                                )
                                ## interchanging the real and generated data
                                h_u_generated, p_generated, mmd_value_u_generated = (
                                    TST_MMD_u(
                                        torch.cat([final_x, final_z], dim=0),
                                        N_per,
                                        final_x.shape[0],
                                        torch.cat([fea_x_ori, fea_z_ori], dim=0).view(
                                            fea_x_ori.shape[0] + fea_z_ori.shape[0], -1
                                        ),
                                        sigma,
                                        sigma0_u,
                                        ep,
                                        args.relative_test_alpha,
                                    )
                                )
                                ## take the smaller p-value and the corresponding h_u and mmd_value_u
                                p_value = p_real
                                h_u = h_u_real
                                mmd_u = mmd_value_u_real
                                if p_generated < p_real:
                                    p_value = abs(p_generated - 1)
                                    h_u = abs(h_u_generated - 1)
                                    mmd_u = mmd_value_u_generated
                                count_u = count_u + h_u
                                p_value_ls.append(p_value)
                                mmd_value_ls.append(mmd_u)

                            test_power_ls.append(count_u / N)

                    return test_power_ls, p_value_ls, mmd_value_ls

                def mmd_three_sample_base(
                    fea_test_ls,
                    fea_real_ls,
                    fea_generated_ls,
                    use_all_references=False,
                    N=10,
                    TST_MMD=TST_MMD_u_3S,
                    TST_MMD_kwargs={},
                    use_meta_model=False,
                    cut_same_length=True,
                ):
                    ## y = real, z = generated
                    fea_test_ls_ori = fea_test_ls
                    ## Concatenate the hidden states of the real and generated data
                    fea_real_ls = torch.cat(fea_real_ls, dim=0).to("cuda")
                    fea_generated_ls = torch.cat(fea_generated_ls, dim=0).to("cuda")
                    ## Cut the real and generated data to the same length (according to the minimum length of the real and generated data)
                    min_len = min(len(fea_real_ls), len(fea_generated_ls))
                    fea_real_ls = fea_real_ls[:min_len]
                    fea_generated_ls = fea_generated_ls[:min_len]

                    p_value_ls = []
                    t_ls = []
                    test_power_ls = []

                    for _ in range(max(args.test_text_n_sample_rounds, 1)):
                        fea_test_ls = fea_test_ls_ori
                        ## If the reference data is not splitted, then randomly sample n samples from the reference data
                        if (
                            not args.test_text_split
                            and len(fea_test_ls_ori) == 1
                            and args.test_text_n_sample_rounds
                        ):
                            ## Randomly sample n samples from the reference data tensor
                            fea_test_ls[0] = fea_test_ls_ori[0][
                                torch.randperm(len(fea_test_ls_ori[0]))[
                                    : args.test_text_n_samples
                                ]
                            ]

                        for i in range(len(fea_test_ls)):
                            fea_test_ori = fea_test_ls[i].to("cuda")
                            min_len = min(
                                len(fea_real_ls),
                                len(fea_generated_ls),
                                len(fea_test_ori),
                            )
                            if cut_same_length:
                                fea_test_ori = fea_test_ori[:min_len]
                            final_x = net(fea_test_ori)

                            # temp_p_value_ls = []
                            # temp_t_ls = []
                            # temp_test_power_ls = []
                            lowest_p_value = math.inf
                            lowest_h_value = None
                            lowest_t_value = None

                            if use_all_references:
                                max_len = (
                                    len(fea_real_ls) - len(fea_test_ori)
                                    if len(fea_real_ls) > len(fea_test_ori)
                                    else len(fea_real_ls)
                                )
                                step_len = min(
                                    len(fea_real_ls),
                                    len(fea_generated_ls),
                                    len(fea_test_ori),
                                )
                                if not cut_same_length:
                                    max_len = len(fea_real_ls)
                                    step_len = min(
                                        len(fea_generated_ls), len(fea_real_ls)
                                    )
                            else:
                                max_len = N
                                step_len = 1
                            j = 0
                            while j < max_len:
                                if use_all_references:
                                    if cut_same_length:
                                        fea_test_ori = fea_test_ori[:step_len]
                                    fea_real_ori = fea_real_ls[j : j + step_len]
                                    fea_generated_ori = fea_generated_ls[
                                        j : j + step_len
                                    ]
                                    final_x = net(fea_test_ori)
                                else:
                                    if cut_same_length:
                                        fea_real_ori = fea_real_ls[
                                            torch.randperm(len(fea_real_ls))[:min_len]
                                        ]
                                        fea_generated_ori = fea_generated_ls[
                                            torch.randperm(len(fea_generated_ls))[
                                                :min_len
                                            ]
                                        ]
                                    else:
                                        fea_real_ori = fea_real_ls[
                                            torch.randperm(len(fea_real_ls))
                                        ]
                                        fea_generated_ori = fea_generated_ls[
                                            torch.randperm(len(fea_generated_ls))
                                        ]

                                final_y = net(fea_real_ori)
                                final_z = net(fea_generated_ori)

                                if use_meta_model:
                                    h_u, p_value, t, *rest = TST_MMD(
                                        final_x,
                                        final_y,
                                        final_z,
                                        fea_test_ori.view(fea_test_ori.shape[0], -1),
                                        fea_real_ori.view(fea_real_ori.shape[0], -1),
                                        fea_generated_ori.view(
                                            fea_generated_ori.shape[0], -1
                                        ),
                                        **TST_MMD_kwargs,
                                    )
                                else:
                                    h_u, p_value, t, *rest = TST_MMD(
                                        fea_test_ori,
                                        fea_real_ori,
                                        fea_generated_ori,
                                        **TST_MMD_kwargs,
                                    )
                                if p_value < lowest_p_value:
                                    lowest_p_value = p_value
                                    lowest_h_value = h_u
                                    lowest_t_value = t
                                # temp_p_value_ls.append(p_value)
                                # temp_t_ls.append(t)
                                # temp_test_power_ls.append(h_u)
                                if use_all_references:
                                    ## If the remaining data is less than the step length, then set the index to the maximum length minus the step length times 2, to ensure the remaining data is enough for 1 step length
                                    if (j + step_len * 2 > max_len) and (
                                        j + step_len < max_len
                                    ):
                                        j = max_len - step_len * 2
                                j += step_len
                            p_value_ls.append(lowest_p_value)
                            t_ls.append(lowest_t_value)
                            test_power_ls.append(lowest_h_value)

                            # ## Take the minimum p-value and the corresponding t and test power
                            # temp_p_value_ls = [min(max(p, 1e-08), 1 - 1e-08) for p in temp_p_value_ls]
                            # _, p_value = combine_pvalues(temp_p_value_ls, method='stouffer')
                            # if p_value < lowest_p_value:
                            #     lowest_p_value = p_value
                            #     p_value_ls = temp_p_value_ls
                            #     t_ls = temp_t_ls
                            #     test_power_ls = temp_test_power_ls

                    if rest:
                        return test_power_ls, p_value_ls, t_ls, rest
                    return test_power_ls, p_value_ls, t_ls

                def mmd_three_sample(fea_test_ls, fea_real_ls, fea_generated_ls, N=10):
                    TST_MMD_kwargs = {
                        "sigma": sigma,
                        "sigma0": sigma0_u,
                        "epsilon": ep,
                        "alpha": args.relative_test_alpha,
                        "is_smooth": True,
                    }
                    return mmd_three_sample_base(
                        fea_test_ls,
                        fea_real_ls,
                        fea_generated_ls,
                        use_all_references=(args.relative_test_reference_mode == "all"),
                        N=N,
                        TST_MMD=TST_MMD_u_3S,
                        TST_MMD_kwargs=TST_MMD_kwargs,
                        use_meta_model=True,
                    )

                def mmd_three_sample_permutation(
                    fea_test_ls, fea_real_ls, fea_generated_ls, N=10
                ):
                    TST_MMD_kwargs = {
                        "sigma": sigma,
                        "sigma0": sigma0_u,
                        "epsilon": ep,
                        "alpha": args.relative_test_alpha,
                        "num_permutations": 50,
                        "is_smooth": True,
                    }
                    return mmd_three_sample_base(
                        fea_test_ls,
                        fea_real_ls,
                        fea_generated_ls,
                        use_all_references=True,
                        N=N,
                        TST_MMD=TST_MMD_u_3S_Permutation_Kernel,
                        TST_MMD_kwargs=TST_MMD_kwargs,
                        use_meta_model=True,
                        cut_same_length=False,
                    )

                def mmd_three_sample_baseline(
                    fea_test_ls,
                    fea_real_ls,
                    fea_generated_ls,
                    no_median_heuristic=False,
                    N=10,
                ):
                    TST_MMD_kwargs = {
                        # 'sigma': 1.0,
                        "alpha": args.relative_test_alpha,
                        "return_sigma": (not test_flag),
                        "sigma_path": model_path
                        + "/"
                        + meta_save_model_flag
                        + "relative_test_baseline_sigma.npy",
                        "no_median_heuristic": no_median_heuristic,
                    }
                    return mmd_three_sample_base(
                        fea_test_ls,
                        fea_real_ls,
                        fea_generated_ls,
                        use_all_references=(args.relative_test_reference_mode == "all"),
                        N=N,
                        TST_MMD=TST_MMD_u_3S_Baseline,
                        TST_MMD_kwargs=TST_MMD_kwargs,
                    )

                p_value_ls = None
                if relative_test_baseline and not test_flag:
                    test_power_ls, p_value_ls, t_ls, *relative_test_baseline_sigma = (
                        mmd_three_sample_baseline(
                            fea_test_ls, fea_real_ls, fea_generated_ls, N=N
                        )
                    )
                if baseline:
                    if args.relative_test_two_sample_mode == "normal":
                        test_power_ls, p_value_ls, t_ls = mmd_two_sample_baseline(
                            fea_test_ls, fea_real_ls, fea_generated_ls, N=N
                        )
                    elif args.relative_test_two_sample_mode == "permutation":
                        test_power_ls, p_value_ls, t_ls = (
                            mmd_two_sample_baseline_permutation(
                                fea_test_ls, fea_real_ls, fea_generated_ls, N=N
                            )
                        )
                elif relative_test_baseline and test_flag:
                    test_power_ls, p_value_ls, t_ls = mmd_three_sample_baseline(
                        fea_test_ls, fea_real_ls, fea_generated_ls, N=N
                    )
                elif relative_test_baseline_simple:
                    test_power_ls, p_value_ls, t_ls = mmd_three_sample_baseline(
                        fea_test_ls,
                        fea_real_ls,
                        fea_generated_ls,
                        N=N,
                        no_median_heuristic=True,
                    )
                elif relative_test_permutation:
                    ## Get the p_value of the relative test
                    test_power_ls, p_value_ls, t_ls = mmd_three_sample_permutation(
                        fea_test_ls, fea_real_ls, fea_generated_ls, N=N
                    )
                elif relative_test:
                    ## Get the p_value of the relative test
                    test_power_ls, p_value_ls, t_ls = mmd_three_sample(
                        fea_test_ls, fea_real_ls, fea_generated_ls, N=N
                    )
                else:
                    ## Get the test power of the real and generated data pairs
                    test_power_ls, t_ls = mmd_two_sample(
                        fea_real_ls, fea_generated_ls, N=N
                    )

                ## Calculate p_value and/or avg_t
                if p_value_ls:
                    ## Avoid p_value_ls to be 1 or 0
                    p_value_ls_temp = [
                        min(max(p, 1e-08), 1 - 1e-08) for p in p_value_ls
                    ]
                    _, p_value = combine_pvalues(p_value_ls_temp, method="stouffer")
                avg_t = sum(t_ls) / len(t_ls)

                ## Print the p_value and/or avg_t or mmd
                if baseline:
                    ## For two sample baseline, print p_value and mmd
                    print(f"p_value: {np.round(p_value, 6)}")
                    print(f"mmd: {np.round(avg_t, 6)}")
                elif p_value_ls:
                    ## For relative tests, print p_value and avg_t
                    print(f"p_value: {np.round(p_value, 6)}")
                    print(f"t: {np.round(avg_t, 6)}")
                else:
                    ## For normal two sample test, print mmd
                    print(f"mmd: {np.round(avg_t, 6)}")

                ## Calculate the average test power
                power = sum(test_power_ls) / len(test_power_ls)
                print(f"power: {np.round(power, 6)}")

                state = {
                    "net": net.state_dict(),
                    # 'epsilonOPT': epsilonOPT,
                    # 'sigmaOPT': sigmaOPT,
                    # 'sigma0OPT': sigma0OPT,
                    "sigma": sigma,
                    "sigma0_u": sigma0_u,
                    "ep": ep,
                }

                ## Save the model if the test power is the best and the test flag is not set
                if not test_flag:
                    if not os.path.isdir(model_path):
                        os.makedirs(model_path, exist_ok=True)
                    # if (epoch+1)%100==0:
                    # 	torch.save(state, model_path + '/'+ str(epoch) +'_ckpt.pth')
                    if power > power_best:
                        power_best = power
                        if relative_test_baseline:
                            np.save(
                                model_path
                                + "/"
                                + meta_save_model_flag
                                + "relative_test_baseline_sigma.npy",
                                relative_test_baseline_sigma,
                            )
                            print(
                                "Save the best relative test baseline sigma: power_best=",
                                power_best,
                            )
                        torch.save(
                            state,
                            model_path + "/" + meta_save_model_flag + "best_ckpt.pth",
                        )
                        print("Save the best model: power_best=", power_best)
            # torch.save(state, model_path + '/'+ meta_save_model_flag +'last_ckpt.pth')
            if relative_test and return_p_and_stat:
                return power, p_value_ls, avg_t
            if (
                baseline or relative_test_baseline or relative_test_baseline_simple
            ) and return_p_and_stat:
                return power, p_value_ls, avg_t
            return power

        _, token_num, hidden_size = fea_train_real0.size()
        fea_dim = token_num * hidden_size
        Config = namedtuple(
            "Config", ["in_dim", "hid_dim", "dropout", "out_dim", "token_num"]
        )
        config = Config(
            in_dim=hidden_size,
            token_num=token_num,
            hid_dim=512,
            dropout=0.2,
            out_dim=300,
        )

        # train
        device = torch.device("cuda:0")
        maml = Meta(args, config).to(device)

        id = args.id

        ## Initialize the model with the given configuration
        net = mmdPreModel(
            config=config,
            num_mlp=args.num_mlp,
            transformer_flag=args.transformer_flag,
            num_hidden_layers=args.num_hidden_layers,
        ).cuda()
        if args.half:
            net = net.half()
        # print('==> loading meta_model from checkpoint..')
        # model_path = f'./net_D/resnet101/{id}'
        # print("No meta learing!")
        sigma, sigma0_u, ep = maml.sigmaOPT**2, maml.sigma0OPT**2, maml.epsilonOPT**2
        ## Code for different metrics
        if args.metric == "auroc" and args.MMDO_flag:
            ep = torch.ones(1).to("cuda", torch.float)
        if not args.test_flag:
            print("==> testing from the loaded checkpoint..")
            num_target = len(fea_real) // test_lenth

            power_ls = []
            ## Get the test power of the real and generated data pairs
            for i in range(num_target):
                power = two_sample_test(
                    0,
                    fea_real_ls=fea_real[i * test_lenth : (i + 1) * test_lenth],
                    fea_generated_ls=fea_generated[
                        i * test_lenth : (i + 1) * test_lenth
                    ],
                    test_flag=True,
                    N=10,
                )
                power_ls.append(power)
            print("average power_value:", sum(power_ls) / len(power_ls))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Initialize parameters
        epsilonOPT = torch.from_numpy(
            np.ones(1) * np.sqrt(ep.detach().cpu().numpy())
        ).to(device, torch.float)
        epsilonOPT.requires_grad = True
        sigmaOPT = torch.from_numpy(
            np.ones(1) * np.sqrt(sigma.detach().cpu().numpy())
        ).to(device, torch.float)
        sigmaOPT.requires_grad = True
        sigma0OPT = torch.from_numpy(
            np.ones(1) * np.sqrt(sigma0_u.detach().cpu().numpy())
        ).to(device, torch.float)
        sigma0OPT.requires_grad = True

        sigma, sigma0_u, ep = None, None, None
        optimizer = torch.optim.Adam(
            list(net.parameters()) + [epsilonOPT] + [sigmaOPT] + [sigma0OPT], lr=args.lr
        )
        epochs = args.epochs
        train_batch_size = args.train_batch_size

        ## Function to train the model using two sample test
        def train(epoch):
            print("\nEpoch: %d" % epoch)

            ## Set the model to training mode
            net.train()
            for batch in tqdm.tqdm(
                range(len(fea_train_generated) // train_batch_size),
                desc="Training for deep MMD",
            ):
                ## Slice the real and generated data into batches
                inputs = fea_train_real[
                    batch * train_batch_size : (batch + 1) * train_batch_size
                ]
                x_adv = fea_train_generated[
                    batch * train_batch_size : (batch + 1) * train_batch_size
                ]
                ## Check if the length of the real and generated data is the same
                if inputs.shape[0] != x_adv.shape[0]:
                    break
                ## Move the real and generated data to the GPU
                inputs = inputs.cuda(non_blocking=True)
                x_adv = x_adv.cuda(non_blocking=True)
                ## Check if the length of the real and generated data is the same again
                assert inputs.shape[0] == x_adv.shape[0]

                ## Concatenate the real and generated data
                X = torch.cat([inputs, x_adv], dim=0)

                optimizer.zero_grad()
                outputs = net(X)

                ep = epsilonOPT**2
                sigma = sigmaOPT**2
                sigma0_u = sigma0OPT**2
                # Compute Compute J (STAT_u)
                TEMP = MMDu(
                    outputs,
                    inputs.shape[0],
                    X.view(X.shape[0], -1),
                    sigma,
                    sigma0_u,
                    ep,
                    coeff_xy=args.coeff_xy,
                    is_yy_zero=args.is_yy_zero,
                    is_xx_zero=args.is_xx_zero,
                )
                mmd_value_temp = -1 * (TEMP[0])
                mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
                STAT_u = torch.div(mmd_value_temp, mmd_std_temp)

                # Compute gradient
                STAT_u.backward()

                # Update weights using gradient descent
                optimizer.step()

            print(
                f"epoch:{epoch}, mmd_value_temp:{mmd_value_temp.item()}, STAT_u:{STAT_u.item()}"
            )
            return sigma, sigma0_u, ep

        id = args.id
        start_epoch = 0
        auroc_value_best_epoch = 0
        ## train/test the model
        if not args.test_flag:
            ## If the test flag is not set, train the model and test it
            for epoch in range(start_epoch, start_epoch + epochs):
                time0 = time.time()

                ## Shuffle the real and generated data
                fea_train_real0 = fea_train_real0[
                    np.random.permutation(fea_train_real0.shape[0])
                ]
                fea_train_generated0 = fea_train_generated0[
                    np.random.permutation(fea_train_generated0.shape[0])
                ]
                if len(fea_train_real0) >= len(fea_train_generated0):
                    for i in range(len(fea_train_real0) // len(fea_train_generated0)):
                        ## If the length of the real data is larger than the generated data, slice the real data into batches to match the length of the generated data
                        ## In order to balance the contribution of the real and generated data to the training
                        fea_train_real = fea_train_real0[
                            fea_train_generated0.shape[0]
                            * i : fea_train_generated0.shape[0]
                            * (i + 1)
                        ]
                        fea_train_generated = fea_train_generated0
                        sigma, sigma0_u, ep = train(epoch)
                else:
                    ## If the length of the generated data is larger than the real data, slice the generated data into batches to match the length of the real data
                    ## In order to balance the contribution of the real and generated data to the training
                    for i in range(len(fea_train_generated0) // len(fea_train_real0)):
                        fea_train_generated = fea_train_generated0[
                            len(fea_train_real0) * i : len(fea_train_real0) * (i + 1)
                        ]
                        fea_train_real = fea_train_real0
                        sigma, sigma0_u, ep = train(epoch)
                ## Print the training time
                print("train time:", time.time() - time0)
                time0 = time.time()
                ## Test the model
                if (epoch + 1) % 1 == 0:
                    ## Code for different metrics
                    if args.metric == "power":
                        ## in this case, we're using the two sample test, so set the test flag of single instance test to True, so we're not saving the model according to the single instance test
                        power = two_sample_test(
                            epoch,
                            fea_real_ls=val_real,
                            fea_generated_ls=val_generated,
                            fea_test_ls=val_generated,
                            N=10,
                            test_flag=False,
                            relative_test_baseline=args.relative_test_baseline,
                        )
                        auroc_value_epoch = single_instance_test(
                            epoch,
                            fea_real=val_sing_real,
                            fea_generated=val_sing_generated,
                            test_flag=True,
                        )
                    else:
                        power = two_sample_test(
                            epoch,
                            fea_real_ls=val_real,
                            fea_generated_ls=val_generated,
                            fea_test_ls=val_generated,
                            N=10,
                            test_flag=True,
                        )
                        auroc_value_epoch = single_instance_test(
                            epoch,
                            fea_real=val_sing_real,
                            fea_generated=val_sing_generated,
                            test_flag=False,
                        )
                    if auroc_value_epoch > auroc_value_best_epoch:
                        auroc_value_best_epoch = auroc_value_epoch
                print("test time:", time.time() - time0)

            ## After training, test the model again
            print("==> loading meta_best_model from checkpoint..")
            model_path = f"./{PATH_exper}/HC3-{args.base_model_name}/{id}"
            assert os.path.isdir(model_path), "Error: no checkpoint directory found!"
            checkpoint = torch.load(model_path + "/" + "best_ckpt.pth")
            net.load_state_dict(checkpoint["net"])
            sigma, sigma0_u, ep = (
                checkpoint["sigma"],
                checkpoint["sigma0_u"],
                checkpoint["ep"],
            )
            print("==> testing from the loaded checkpoint..")
            power = two_sample_test(epoch, test_flag=True)
            auroc_value = single_instance_test(
                epoch,
                fea_real=val_sing_real,
                fea_generated=val_sing_generated,
                test_flag=True,
            )
            print("==> testing each model..")
            for i in range(num_target):
                ## Code for different metrics
                if args.metric == "power":
                    two_sample_test(
                        0,
                        fea_real_ls=fea_real[i * test_lenth : (i + 1) * test_lenth],
                        fea_generated_ls=fea_generated[
                            i * test_lenth : (i + 1) * test_lenth
                        ],
                        fea_test_ls=fea_generated[
                            i * test_lenth : (i + 1) * test_lenth
                        ],
                        test_flag=True,
                        N=10,
                    )
                else:
                    single_instance_test(
                        0,
                        fea_real=val_sing_real_ls[i][:1000],
                        fea_generated=val_sing_generated_ls[i][:1000],
                        test_flag=True,
                    )

            ## Print the best power and auroc value of the model
            print(f"{id}'s best power is {power}!")
            print(f"and the corresponding auroc is {auroc_value}!")
            print(f"but the best auroc is {auroc_value_best_epoch}!")

        else:
            ## If the test flag is set, test the model only without training
            epoch = 99
            print("==> testing from checkpoint..")
            model_path = f"./{PATH_exper}/HC3-{args.base_model_name}/{id}"
            if not os.path.isdir(model_path):
                model_path = f"./{PATH_exper}/HC3-gpt2/999"
                print(f"Note you are loading {model_path}")
            assert os.path.isdir(model_path), "Error: no checkpoint directory found!"
            checkpoint = torch.load(model_path + "/" + "best_ckpt.pth")
            net.load_state_dict(checkpoint["net"])
            sigma, sigma0_u, ep = (
                checkpoint["sigma"],
                checkpoint["sigma0_u"],
                checkpoint["ep"],
            )

            # test(epoch)
            def print_auroc_power(
                all_power_list=all_power_list, all_auroc_list=all_auroc_list
            ):
                print(f"The best power list is {all_power_list}!")
                print(f"The best auroc list is {all_auroc_list}!")
                print(
                    f"avg_power: {avg_value(all_power_list)[0]} and std_power: {avg_value(all_power_list)[1]}"
                )
                print(
                    f"avg_auroc: {avg_value(all_auroc_list)[0]} and std_auroc: {avg_value(all_auroc_list)[1]}"
                )

            if (
                args.test_text is None
                and args.test_text_file is None
                and not args.relative_test
            ):
                if args.relative_test:
                    power, p_value, t = two_sample_test(
                        epoch, test_flag=True, return_p_and_stat=True
                    )
                    relative_test_p_value_list.append(np.round(p_value, 6))
                    relative_test_t_stat_list.append(np.round(t, 6))
                else:
                    power = two_sample_test(epoch, test_flag=True)
                auroc_value = single_instance_test(
                    epoch,
                    fea_real=val_sing_real,
                    fea_generated=val_sing_generated,
                    test_flag=True,
                )
                all_power_list.append(np.round(power, 6))
                all_auroc_list.append(np.round(auroc_value, 6))
                ## Print the best power and auroc value of the model and the average and standard deviation of the power and auroc value if we're in the last trial
                if current_trial == args.trial_num:
                    print_auroc_power()
            else:
                ## Assume test text is generated
                power_genenrated = two_sample_test(
                    epoch,
                    fea_generated_ls=fea_test,
                    test_flag=True,
                    relative_test=False,
                )
                auroc_value_generated = single_instance_test(
                    epoch,
                    fea_real=val_sing_real,
                    fea_generated=fea_test_single,
                    test_flag=True,
                )
                test_power_list_generated.append(np.round(power_genenrated, 6))
                test_auroc_list_generated.append(np.round(auroc_value_generated, 6))
                ## Assume test text is real
                power_real = two_sample_test(
                    epoch, fea_real_ls=fea_test, test_flag=True, relative_test=False
                )
                auroc_value_real = single_instance_test(
                    epoch,
                    fea_real=fea_test_single,
                    fea_generated=val_sing_generated,
                    test_flag=True,
                )
                test_power_list_real.append(np.round(power_real, 6))
                test_auroc_list_real.append(np.round(auroc_value_real, 6))
                if args.relative_test:
                    ## First, run two sample test baseline
                    start_time = time.time()
                    power_baseline, p_value_baseline, mmd_baseline = two_sample_test(
                        epoch,
                        fea_test_ls=fea_test,
                        test_flag=True,
                        baseline=True,
                        return_p_and_stat=True,
                        relative_test=False,
                    )
                    end_time = time.time()
                    baseline_time_list.append(np.round(end_time - start_time, 6))
                    baseline_power_list.append(np.round(power_baseline, 6))
                    baseline_p_value_list.append(np.round(p_value_baseline, 6))
                    baseline_mmd_list.append(np.round(mmd_baseline, 6))

                    if args.relative_test_baseline_simple:
                        ## Then, run simple relative test baseline
                        start_time = time.time()
                        (
                            power_simple_relative_baseline,
                            p_value_simple_relative_baseline,
                            t_simple_relative_baseline,
                        ) = two_sample_test(
                            epoch,
                            fea_test_ls=fea_test,
                            test_flag=True,
                            return_p_and_stat=True,
                            relative_test=False,
                            relative_test_baseline_simple=True,
                        )
                        end_time = time.time()
                        simple_relative_baseline_time_list.append(
                            np.round(end_time - start_time, 6)
                        )
                        simple_relative_baseline_p_value_list.append(
                            np.round(p_value_simple_relative_baseline, 6)
                        )
                        simple_relative_baseline_t_stat_list.append(
                            np.round(t_simple_relative_baseline, 6)
                        )
                        simple_relative_baseline_power_list.append(
                            np.round(power_simple_relative_baseline, 6)
                        )
                    if args.relative_test_baseline:
                        ## Then, run relative test baseline
                        start_time = time.time()
                        (
                            power_relative_baseline,
                            p_value_relative_baseline,
                            t_relative_baseline,
                        ) = two_sample_test(
                            epoch,
                            fea_test_ls=fea_test,
                            test_flag=True,
                            relative_test_baseline=True,
                            return_p_and_stat=True,
                            relative_test=False,
                        )
                        end_time = time.time()
                        relative_baseline_time_list.append(
                            np.round(end_time - start_time, 6)
                        )
                        relative_baseline_p_value_list.append(
                            np.round(p_value_relative_baseline, 6)
                        )
                        relative_baseline_t_stat_list.append(
                            np.round(t_relative_baseline, 6)
                        )
                        relative_baseline_power_list.append(
                            np.round(power_relative_baseline, 6)
                        )
                    if args.binoculars or args.output_test_text_file:
                        fea_test_ori_list_temp = []
                        for i in range(args.test_text_n_sample_rounds):
                            if args.test_text_split:
                                fea_test_ori_ls_temp = random.sample(
                                    fea_test_ori_ls, len(fea_test_ori_ls)
                                )
                            else:
                                fea_test_ori_ls_temp = random.sample(
                                    fea_test_ori_ls[0], len(fea_test_ori_ls[0])
                                )
                            fea_test_ori = " ".join(
                                paragraph for paragraph in fea_test_ori_ls_temp
                            )
                            fea_test_ori_list_temp.append(fea_test_ori)
                        fea_test_ori_list.append(fea_test_ori_list_temp)

                    ## Get the relative test p_value
                    start_time = time.time()
                    power, p_value, t = two_sample_test(
                        epoch,
                        fea_test_ls=fea_test,
                        test_flag=True,
                        return_p_and_stat=True,
                        relative_test_permutation=(
                            args.relative_test_mode == "permutation"
                        ),
                    )
                    end_time = time.time()
                    relative_test_time_list.append(np.round(end_time - start_time, 6))
                    relative_test_p_value_list.append(np.round(p_value, 6))
                    relative_test_t_stat_list.append(np.round(t, 6))
                    relative_test_power_list.append(np.round(power, 6))
                ## Print the best power and auroc value of the model and the average and standard deviation of the power and auroc value if we're in the last trial
                if current_trial == args.trial_num:
                    print(
                        "When assuming the test text is generated (comparing test text to ground truth real data):"
                    )
                    print_auroc_power(
                        test_power_list_generated, test_auroc_list_generated
                    )

                    print()

                    print(
                        "When assuming the test text is real (comparing test text to ground truth generated data):"
                    )
                    print_auroc_power(test_power_list_real, test_auroc_list_real)

                    if args.relative_test:
                        if len(set(answer_labels)) > 1:
                            print()
                            print("Ground-truth answer Labels:", answer_labels)

                        print()

                        ## First, print the two sample test baseline test results
                        # print(f"The mmd-mp baseline p_value list is {baseline_p_value_list}!")
                        print(f"The mmd-mp baseline mmd list is {baseline_mmd_list}!")
                        print(
                            f"The mmd-mp baseline power list is {baseline_power_list}!"
                        )
                        if len(set(answer_labels)) > 1:
                            baseline_power_hwt_list = [
                                baseline_power_list[i]
                                for i in range(len(baseline_power_list))
                                if answer_labels[i] == 0
                            ]
                            baseline_power_mgt_list = [
                                baseline_power_list[i]
                                for i in range(len(baseline_power_list))
                                if answer_labels[i] == 1
                            ]
                            print(
                                f"The mmd-mp baseline power list for HWT is {baseline_power_hwt_list}!"
                            )
                            print(
                                f"The mmd-mp baseline power list for MGT is {baseline_power_mgt_list}!"
                            )
                            print(
                                f"The mmd-mp baseline avg_power for HWT: {avg_value(baseline_power_hwt_list)[0]} and std_power: {avg_value(baseline_power_hwt_list)[1]}"
                            )
                            print(
                                f"The mmd-mp baseline avg_power for MGT: {avg_value(baseline_power_mgt_list)[0]} and std_power: {avg_value(baseline_power_mgt_list)[1]}"
                            )
                            print(f"The mmd-mp baseline auroc stats:")
                            process_p_values_and_labels_odd(
                                answer_labels, baseline_p_value_list
                            )
                            (
                                baseline_auroc_list,
                                baseline_fpr_list,
                                baseline_tpr_list,
                            ) = process_p_values_and_labels(
                                answer_labels, baseline_p_value_list
                            )
                            print(f"AUROC_LIST: {baseline_auroc_list}!")
                            # print(f"FPR_LIST: {baseline_fpr_list}!")
                            # print(f"TPR_LIST: {baseline_tpr_list}!")
                        else:
                            print(
                                f"The mmd-mp baseline avg_power: {avg_value(baseline_power_list)[0]} and std_power: {avg_value(baseline_power_list)[1]}"
                            )
                        print(f"The mmd-mp baseline time list is {baseline_time_list}!")
                        print(
                            f"The mmd-mp baseline total time is {np.sum(baseline_time_list)}!"
                        )

                        if args.relative_test_baseline_simple:
                            print()

                            ## Then, print the relative baseline test results
                            # print(f"The simple relative test baseline p_value list is {simple_relative_baseline_p_value_list}!")
                            print(
                                f"The simple relative test baseline t_stat list is {simple_relative_baseline_t_stat_list}!"
                            )
                            print(
                                f"The simple relative test baseline power list is {simple_relative_baseline_power_list}!"
                            )
                            if len(set(answer_labels)) > 1:
                                simple_relative_baseline_power_hwt_list = [
                                    simple_relative_baseline_power_list[i]
                                    for i in range(
                                        len(simple_relative_baseline_power_list)
                                    )
                                    if answer_labels[i] == 0
                                ]
                                simple_relative_baseline_power_mgt_list = [
                                    simple_relative_baseline_power_list[i]
                                    for i in range(
                                        len(simple_relative_baseline_power_list)
                                    )
                                    if answer_labels[i] == 1
                                ]
                                print(
                                    f"The simple relative test baseline power list for HWT is {simple_relative_baseline_power_hwt_list}!"
                                )
                                print(
                                    f"The simple relative test baseline power list for MGT is {simple_relative_baseline_power_mgt_list}!"
                                )
                                print(
                                    f"The simple relative test baseline avg_power for HWT: {avg_value(simple_relative_baseline_power_hwt_list)[0]} and std_power: {avg_value(simple_relative_baseline_power_hwt_list)[1]}"
                                )
                                print(
                                    f"The simple relative test baseline avg_power for MGT: {avg_value(simple_relative_baseline_power_mgt_list)[0]} and std_power: {avg_value(simple_relative_baseline_power_mgt_list)[1]}"
                                )
                                print("The simple relative test baseline auroc stats:")
                                process_p_values_and_labels_odd(
                                    answer_labels, simple_relative_baseline_p_value_list
                                )
                                (
                                    simple_relative_baseline_auroc_list,
                                    simple_relative_baseline_fpr_list,
                                    simple_relative_baseline_tpr_list,
                                ) = process_p_values_and_labels(
                                    answer_labels, simple_relative_baseline_p_value_list
                                )
                                print(
                                    f"AUROC_LIST: {simple_relative_baseline_auroc_list}!"
                                )
                                # print(f"FPR_LIST: {simple_relative_baseline_fpr_list}!")
                                # print(f"TPR_LIST: {simple_relative_baseline_tpr_list}!")
                            else:
                                print(
                                    f"The simple relative test baseline avg_power: {avg_value(simple_relative_baseline_power_list)[0]} and std_power: {avg_value(simple_relative_baseline_power_list)[1]}"
                                )
                            print(
                                f"The simple relative test baseline time list is {simple_relative_baseline_time_list}!"
                            )
                            print(
                                f"The simple relative test baseline total time is {np.sum(simple_relative_baseline_time_list)}!"
                            )

                        if args.relative_test_baseline:
                            print()

                            ## Then, print the relative baseline test results
                            # print(f"The relative test baseline p_value list is {relative_baseline_p_value_list}!")
                            print(
                                f"The relative test baseline t_stat list is {relative_baseline_t_stat_list}!"
                            )
                            print(
                                f"The relative test baseline power list is {relative_baseline_power_list}!"
                            )
                            if len(set(answer_labels)) > 1:
                                relative_baseline_power_hwt_list = [
                                    relative_baseline_power_list[i]
                                    for i in range(len(relative_baseline_power_list))
                                    if answer_labels[i] == 0
                                ]
                                relative_baseline_power_mgt_list = [
                                    relative_baseline_power_list[i]
                                    for i in range(len(relative_baseline_power_list))
                                    if answer_labels[i] == 1
                                ]
                                print(
                                    f"The relative test baseline power list for HWT is {relative_baseline_power_hwt_list}!"
                                )
                                print(
                                    f"The relative test baseline power list for MGT is {relative_baseline_power_mgt_list}!"
                                )
                                print(
                                    f"The relative test baseline avg_power for HWT: {avg_value(relative_baseline_power_hwt_list)[0]} and std_power: {avg_value(relative_baseline_power_hwt_list)[1]}"
                                )
                                print(
                                    f"The relative test baseline avg_power for MGT: {avg_value(relative_baseline_power_mgt_list)[0]} and std_power: {avg_value(relative_baseline_power_mgt_list)[1]}"
                                )
                                print("The relative test baseline auroc stats:")
                                process_p_values_and_labels_odd(
                                    answer_labels, relative_baseline_p_value_list
                                )
                                (
                                    relative_baseline_auroc_list,
                                    relative_baseline_fpr_list,
                                    relative_baseline_tpr_list,
                                ) = process_p_values_and_labels(
                                    answer_labels, relative_baseline_p_value_list
                                )
                                print(f"AUROC_LIST: {relative_baseline_auroc_list}!")
                                # print(f"FPR_LIST: {relative_baseline_fpr_list}!")
                                # print(f"TPR_LIST: {relative_baseline_tpr_list}!")
                            else:
                                print(
                                    f"The relative test baseline avg_power: {avg_value(relative_baseline_power_list)[0]} and std_power: {avg_value(relative_baseline_power_list)[1]}"
                                )
                            print(
                                f"The relative test baseline time list is {relative_baseline_time_list}!"
                            )
                            print(
                                f"The relative test baseline total time is {np.sum(relative_baseline_time_list)}!"
                            )

                        if fea_test_ori_list:
                            ## Output the test_ori_list into a txt file for manual checking
                            with open(
                                f"./{PATH_exper}/HC3-{args.base_model_name}/{id}/test_data.txt",
                                "w",
                                encoding="utf-8",
                            ) as file:
                                for i, lst in enumerate(fea_test_ori_list):
                                    file.write(
                                        f"=== Trial {i + 1} {'('+ ('MGT' if answer_labels[i] else 'HWT') +')' if answer_labels else args.test_dataset_answer} ===\n"
                                    )
                                    for j, paragraph in enumerate(lst):
                                        file.write(f"--- Shuffle {j + 1} ---\n")
                                        file.write(repr(paragraph) + "\n")
                                        file.write("\n")
                                    file.write("\n")
                                    file.write("=" * 30 + "\n\n")
                            ## Also output the test_ori_list into a csv file for other analysis
                            with open(
                                f"./{PATH_exper}/HC3-{args.base_model_name}/{id}/test_data.csv",
                                "w",
                                encoding="utf-8",
                            ) as file:
                                writer = csv.writer(file)
                                writer.writerow(["trial", "shuffle", "label", "text"])
                                for i, lst in enumerate(fea_test_ori_list):
                                    for j, paragraph in enumerate(lst):
                                        writer.writerow(
                                            [
                                                i,
                                                j,
                                                "MGT" if answer_labels[i] else "HWT",
                                                repr(paragraph),
                                            ]
                                        )

                        if args.binoculars:
                            print()

                            ## Import Binoculars if the binoculars flag is set
                            from Binoculars.binoculars import Binoculars

                            bino = Binoculars()
                            ## Then, run binoculars
                            for i in range(len(fea_test_ori_list)):
                                fea_test_ori_list_temp = fea_test_ori_list[i]
                                binoculars_p_value_list_temp = []
                                binoculars_result_list_temp = []
                                binoculars_time_list_temp = []
                                for j in range(len(fea_test_ori_list_temp)):
                                    fea_test_ori = fea_test_ori_list_temp[j]
                                    start_time = time.time()
                                    score = bino.compute_score(fea_test_ori)
                                    end_time = time.time()
                                    result = (
                                        "HWT"
                                        if "human" in bino.predict(fea_test_ori)
                                        else "MGT"
                                    )
                                    binoculars_p_value_list_temp.append(score)
                                    binoculars_result_list_temp.append(result)
                                    binoculars_time_list_temp.append(
                                        end_time - start_time
                                    )
                                binoculars_time_list.append(
                                    np.round(np.sum(binoculars_time_list_temp), 6)
                                )
                                binoculars_power_list.append(
                                    round(
                                        np.mean(
                                            [p for p in binoculars_p_value_list_temp]
                                        ),
                                        6,
                                    )
                                )
                                binoculars_p_value_list.append(
                                    binoculars_p_value_list_temp
                                )
                                ## Take the most frequent result as the final result
                                binoculars_result_list.append(
                                    max(
                                        set(binoculars_result_list_temp),
                                        key=binoculars_result_list_temp.count,
                                    )
                                )
                            ## Print the binoculars test results
                            print(
                                f"The binoculars score list is {binoculars_power_list}!"
                            )
                            print(
                                f"The binoculars result list is {binoculars_result_list}!"
                            )
                            if len(set(answer_labels)) > 1:
                                binoculars_power_hwt_list = [
                                    binoculars_power_list[i]
                                    for i in range(len(binoculars_power_list))
                                    if answer_labels[i] == 0
                                ]
                                binoculars_power_mgt_list = [
                                    binoculars_power_list[i]
                                    for i in range(len(binoculars_power_list))
                                    if answer_labels[i] == 1
                                ]
                                print(
                                    f"The binoculars score list for HWT is {binoculars_power_hwt_list}!"
                                )
                                print(
                                    f"The binoculars score list for MGT is {binoculars_power_mgt_list}!"
                                )
                                print(
                                    f"The binoculars avg_score for HWT: {avg_value(binoculars_power_hwt_list)[0]} and std_score: {avg_value(binoculars_power_hwt_list)[1]}"
                                )
                                print(
                                    f"The binoculars avg_score for MGT: {avg_value(binoculars_power_mgt_list)[0]} and std_score: {avg_value(binoculars_power_mgt_list)[1]}"
                                )
                                process_p_values_and_labels_odd(
                                    answer_labels, binoculars_power_list
                                )
                                (
                                    binoculars_auroc_list,
                                    binoculars_fpr_list,
                                    binoculars_tpr_list,
                                ) = process_p_values_and_labels(
                                    answer_labels, binoculars_power_list
                                )
                                print(f"AUROC_LIST: {binoculars_auroc_list}!")
                                # print(f"FPR_LIST: {binoculars_fpr_list}!")
                                # print(f"TPR_LIST: {binoculars_tpr_list}!")
                            else:
                                print(
                                    f"The binoculars avg_score: {avg_value(binoculars_power_list)[0]} and std_score: {avg_value(binoculars_power_list)[1]}"
                                )
                            print(
                                f"The binoculars time list is {binoculars_time_list}!"
                            )
                            print(
                                f"The binoculars total time is {np.sum(binoculars_time_list)}!"
                            )

                        print()

                        ## Print the relative test p_value and lists
                        # print(f"The relative test p_value list is {relative_test_p_value_list}!")
                        print(
                            f"The relative test t_stat list is {relative_test_t_stat_list}!"
                        )
                        print(
                            f"The relative test power list is {relative_test_power_list}!"
                        )
                        if len(set(answer_labels)) > 1:
                            relative_test_power_hwt_list = [
                                relative_test_power_list[i]
                                for i in range(len(relative_test_power_list))
                                if answer_labels[i] == 0
                            ]
                            relative_test_power_mgt_list = [
                                relative_test_power_list[i]
                                for i in range(len(relative_test_power_list))
                                if answer_labels[i] == 1
                            ]
                            print(
                                f"The relative test power list for HWT is {relative_test_power_hwt_list}!"
                            )
                            print(
                                f"The relative test power list for MGT is {relative_test_power_mgt_list}!"
                            )
                            print(
                                f"The relative test avg_power for HWT: {avg_value(relative_test_power_hwt_list)[0]} and std_power: {avg_value(relative_test_power_hwt_list)[1]}"
                            )
                            print(
                                f"The relative test avg_power for MGT: {avg_value(relative_test_power_mgt_list)[0]} and std_power: {avg_value(relative_test_power_mgt_list)[1]}"
                            )
                            print("The relative test auroc stats:")
                            process_p_values_and_labels_odd(
                                answer_labels, relative_test_p_value_list
                            )
                            (
                                relative_test_auroc_list,
                                relative_test_fpr_list,
                                relative_test_tpr_list,
                            ) = process_p_values_and_labels(
                                answer_labels, relative_test_p_value_list
                            )
                            print(f"AUROC_LIST: {relative_test_auroc_list}!")
                            # print(f"FPR_LIST: {relative_test_fpr_list}!")
                            # print(f"TPR_LIST: {relative_test_tpr_list}!")
                        else:
                            print(
                                f"The relative test avg_power: {avg_value(relative_test_power_list)[0]} and std_power: {avg_value(relative_test_power_list)[1]}"
                            )
                        print(
                            f"The relative test time list is {relative_test_time_list}!"
                        )
                        print(
                            f"The relative test total time is {np.sum(relative_test_time_list)}!"
                        )

    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("==========", "Script Ended:", current_time_str, "==========")
