import argparse


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--config",
        dest="config",
        type=str,
        default="configs/bisenetv2_city.py",
    )
    parse.add_argument(
        "--finetune-from",
        type=str,
        default=None,
    )
    return parse.parse_args()
