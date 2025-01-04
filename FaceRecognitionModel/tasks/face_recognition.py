import hydra
import lightning as L
import torch
import torchmetrics
from omegaconf import DictConfig


class FaceRecognitionModule(L.LightningModule):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = hydra.utils.instantiate(hparams.model)
        self.loss = hydra.utils.instantiate(hparams.loss)


    def forward(self, images: torch.Tensor, *args, **kwargs):
        return self.model(images)


    def training_step(
        self, batch: torch.Tensor, batch_idx: int, *args, **kwargs
    ) -> torch.Tensor:
        # get the training images and identities
        imgs, idtys = batch

        # predict the face embeddings from each image
        img_embs = self.forward(imgs)
        
        # calculate the loss
        loss = self.loss(img_embs, idtys)

        # log the loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss


    def validation_step(
        self, batch: torch.Tensor, batch_idx: int, *args, **kwargs
    ) -> torch.Tensor:
        imgs, idtys = batch

        # predict the face embeddings from each image
        img_embs = self.forward(imgs)
        
        # calculate the loss
        loss = self.loss(img_embs, idtys)

        # log the loss
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return img_embs


    def test_step(
        self, batch: torch.Tensor, batch_idx: int, *args, **kwargs
    ) -> torch.Tensor:
        imgs, idtys = batch

        # predict the face embeddings from each image
        img_embs = self.forward(imgs)
        
        # calculate the loss
        loss = self.loss(img_embs, idtys)

        # log the loss
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return img_embs
    

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer, list(self.model.parameters())
        )
        lr_scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer)
        return [optimizer], [lr_scheduler]
