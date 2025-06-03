# %%
# # R-Detect Main Script
# This script is designed to run R-Detect system. You can run each block interactively in Colab.
import sys
import argparse
from relative_tester import RelativeTester
from utils import init_random_seeds, config, get_device


if __name__ == "__main__":
    init_random_seeds()
    parser = argparse.ArgumentParser(description="R-Detect the file content")
    parser.add_argument(
        "--test_file",
        type=str,
        help="The file path of the test file. Default is demo_text_gpt.txt",
        default="./demo_text_gpt.txt",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU or not.",
    )
    parser.add_argument(
        "--local_model",
        type=str,
        help="Use local model or not, you need to download the model first, and set the path. Script will use remote if this param is empty.",
        default="",
    )
    parser.add_argument(
        "--feature_ref_HWT",
        type=str,
        help="The feature ref path of HWT. Script will use remote if this param is empty.",
        default="",
    )
    parser.add_argument(
        "--feature_ref_MGT",
        type=str,
        help="The feature ref path of MGT. Default is Empty",
        default="",
    )
    args = parser.parse_args()
    config["test_file"] = args.test_file
    config["use_gpu"] = args.use_gpu
    config["local_model"] = args.local_model
    config["feature_ref_HWT"] = args.feature_ref_HWT
    config["feature_ref_MGT"] = args.feature_ref_MGT
    print(f"Running on device", get_device())
    with open(config["test_file"], "r") as file:
        content = file.read()
        relative_tester = RelativeTester()
        print(relative_tester.test(content))
    # print(content)
