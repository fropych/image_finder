from typing import Any
from itertools import chain

import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from lightning import LightningModule
from torchmetrics import Accuracy, F1Score, MaxMetric, MeanMetric, Recall


class LitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        scheduler: torch.optim.lr_scheduler,
        num_classes: int,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.net = net
        self.criterion = criterion
        self.num_classes = num_classes

        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_acc_best = MaxMetric()
        self.val_f1_best = MaxMetric()

        self.knn = None

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def model_step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        return loss, logits, y

    def predict_step(self, batch: Any, batch_idx: int) -> np.ndarray:
        if self.knn is None:
            raise RuntimeError("Not trained")
        embeds = self.forward(batch)
        preds = self.knn.predict(embeds.cpu().tolist())

        return preds

    def on_train_start(self):
        self.val_loss.reset()

        self.val_acc.reset()
        self.val_acc_best.reset()

        self.val_f1.reset()
        self.val_f1_best.reset()

    def training_step(self, batch: Any, batch_idx: int):
        loss, _, _ = self.model_step(batch)

        self.train_loss(loss)
        self.log(
            "train_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )

        return loss

    @torch.no_grad()
    def on_train_epoch_end(self):
        self.eval()
        embeds = [[], []]
        for batch in self.trainer.train_dataloader:
            x, y = batch
            logits = self.forward(x.to(self.device))
            embeds[0].extend(logits.tolist())
            embeds[1].extend(y.tolist())

        x, y = np.array(embeds[0]), np.array(embeds[1])
        self.knn = KNeighborsClassifier(metric="cosine", n_jobs=-1)
        self.knn = self.knn.fit(x, y)

    def validation_step(self, batch: Any, batch_idx: int):
        loss, embeds, targets = self.model_step(batch)
        if self.knn is None:
            preds = torch.zeros(
                (batch[0].shape[0], self.num_classes), device=self.device
            )
        else:
            preds = self.knn.predict(embeds.cpu().tolist())
            preds = torch.tensor(preds, device=targets.device)

        self.val_loss(loss)
        self.val_f1(preds, targets)
        self.val_acc(preds, targets)

        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        f1 = self.val_f1.compute()
        self.val_acc_best(acc)
        self.val_f1_best(f1)
        self.log("val_acc_best", self.val_acc_best.compute(), prog_bar=True)
        self.log("val_f1_best", self.val_f1_best.compute(), prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(
            params=chain(self.parameters(), self.hparams.criterion.parameters())
        )
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
