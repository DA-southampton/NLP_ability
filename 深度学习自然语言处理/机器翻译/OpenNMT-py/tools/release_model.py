#!/usr/bin/env python
import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Removes the optim data of PyTorch models")
    parser.add_argument("--model", "-m",
                        help="The model filename (*.pt)", required=True)
    parser.add_argument("--output", "-o",
                        help="The output filename (*.pt)", required=True)
    opt = parser.parse_args()

    model = torch.load(opt.model)
    model['optim'] = None
    torch.save(model, opt.output)
