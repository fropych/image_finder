from omegaconf import DictConfig
from src.data.components import imgflip_parser, process_images

import hydra
import pyrootutils
from hydra import compose, initialize
from pathlib import Path

root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

with initialize(
    version_base="1.2", config_path="../../configs", job_name="make_dataset"
):
    cfg = compose(config_name="make_dataset.yaml")


def main():
    imgflip_parser.main(cfg)
    process_images.main(cfg)


if __name__ == "__main__":
    main()
