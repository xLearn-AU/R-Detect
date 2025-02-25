#!/bin/bash

# If you are in China, you can use set the following mirror link to download the model and dataset
# export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download --resume-download openai-community/roberta-base-openai-detector --local-dir llm-models/roberta-base
huggingface-cli download --repo-type dataset --resume-download Hello-SimpleAI/HC3 --local-dir datasets/HC3

