import hydra
import pyrootutils
from omegaconf import DictConfig
from src.imgsim import train
from src.imgsim.data import get_transforms
import os

os.environ["HYDRA_FULL_ERROR"] = "1"
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@hydra.main(version_base="1.2", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig):
    train.main(cfg)


if __name__ == "__main__":
    print(get_transforms.test())
    main()
