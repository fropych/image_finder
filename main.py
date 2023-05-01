import hydra
from lightning import LightningDataModule, LightningModule
import pyrootutils
from omegaconf import DictConfig

from src.data.datamodule import DataModule
from src.models.lightning_module import LitModule
from src import train

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@hydra.main(version_base="1.2", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig):
    train.main(cfg)

if __name__ == '__main__':
    main()
