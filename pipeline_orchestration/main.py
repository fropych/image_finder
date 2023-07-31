import os

import hydra
import pyrootutils
from prefect import flow, task

from imgfinder import train
from parser import parse, process_images

root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
os.environ["PROJECT_ROOT"] = str(root.absolute())


@task(retries=3)
def parse_data(cfg):
    return cfg
    parse(cfg)


@task
def process_data(cfg):
    return cfg
    process_images(cfg)


@task
def train_model(cfg):
    train(cfg)


@hydra.main(version_base="1.1", config_path="../configs", config_name="train.yaml")
@flow(name="Train Model", log_prints=True)
def main(cfg):
    cfg = parse_data(cfg)
    cfg = process_data(cfg)
    train_model(cfg)


if __name__ == "__main__":
    main()
