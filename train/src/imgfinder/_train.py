from typing import Tuple

import hydra
import lightning as L
import wandb
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from . import utils


log = utils.get_pylogger("imgfinder")


def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    # TODO: add confusion matrix https://www.ravirajag.dev/blog/mlops-wandb-integration
    wandb_logger = WandbLogger(
        project="Image Finder",
    )

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    cfg.model.num_classes = datamodule.num_classes
    cfg.model.criterion.num_classes = datamodule.num_classes

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)
    trainer = trainer(logger=wandb_logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "trainer": trainer,
    }

    if cfg.train:
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.test:
        ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    test_metrics = trainer.callback_metrics
    metric_dict = {**train_metrics, **test_metrics}

    if cfg.save_model_wandb:
        model_art = wandb.Artifact(f"vit-emb", type="model", metadata=metric_dict)
        model_art.add_file(trainer.checkpoint_callback.best_model_path, "model.pt")

        try:
            artifact = wandb.use_artifact(
                "mr-misister/model-registry/Image Embedder:best", type="model"
            )
            best_score = artifact.metadata["val_f1_best"]
        except wandb.CommError as exception:
            best_score = -1
            log.info(
                f"There is no saved model on wandb. This model will be saved as best"
            )

        aliases = ["latest"]
        wandb.log_artifact(model_art, aliases=aliases)

        if metric_dict["val_f1_best"] > best_score:
            aliases.append("best")

            wandb.run.link_artifact(
                model_art, "mr-misister/model-registry/Image Embedder", aliases=aliases
            )

    return metric_dict, object_dict
