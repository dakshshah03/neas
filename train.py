"""
NeAS Training Script

Neural Attenuation Fields for Sparse-View CBCT Reconstruction
"""
import os
import torch
import argparse

from src.config import load_config
from src.trainer import Trainer


def config_parser():
    parser = argparse.ArgumentParser(description="Train NeAS model")
    parser.add_argument("--config", default="./config/foot_50.yaml",
                        help="Path to config file")
    return parser


if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Start] exp: {cfg['exp']['expname']}")
    print(f"Using device: {device}")

    # Create trainer and start training
    trainer = Trainer(cfg, device)
    trainer.start()
