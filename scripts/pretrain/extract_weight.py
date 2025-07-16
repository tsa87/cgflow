import argparse

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=str)
    parser.add_argument("-o", "--out_path", type=str, required=True)
    args = parser.parse_args()

    state_dict = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    extract_state_dict = {
        "hyper_parameters": state_dict["hyper_parameters"],
        "state_dict": state_dict["state_dict"],
    }
    torch.save(extract_state_dict, args.out_path)
