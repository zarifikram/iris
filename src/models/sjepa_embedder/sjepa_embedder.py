from dataclasses import dataclass
from typing import Any, Optional, Tuple

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sjepa import SJEPA


class SJEPA_Embedder(nn.Module):
    def __init__(
        self,
        encoder: SJEPA,
        lr: float,
        weight_decay: float,
        target_aspect_ratio: Tuple[float, float],
        target_scale: Tuple[float, float],
        context_aspect_ratio: float,
        context_scale: Tuple[float, float],
        m: float,
        m_start_end: Tuple[float, float],
    ) -> None:
        super().__init__()

        # define models
        self.model = encoder
        # define hyperparameters
        self.M = encoder.M
        self.lr = lr
        self.weight_decay = weight_decay
        self.m = m
        self.target_aspect_ratio = target_aspect_ratio
        self.target_scale = target_scale
        self.context_aspect_ratio = context_aspect_ratio
        self.context_scale = context_scale
        self.embed_dim = encoder.embed_dim
        self.num_tokens = encoder.sequence_length
        self.m_start_end = m_start_end

        # define loss
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    # def forward(self, x, target_aspect_ratio, target_scale, context_aspect_ratio, context_scale):
    #     return self.model(x, target_aspect_ratio, target_scale, context_aspect_ratio, context_scale)

    # '''Update momentum for teacher encoder'''
    # def update_momentum(self, m):
    #     student_model = self.model.student_encoder.eval()
    #     teacher_model = self.model.teacher_encoder.eval()
    #     with torch.no_grad():
    #         for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
    #             teacher_param.data.mul_(other=m).add_(other=student_param.data, alpha=1 - m)

    # def training_step(self, batch, batch_idx):
    #     x = batch
    #     #generate random target and context aspect ratio and scale
    #     target_aspect_ratio = np.random.uniform(self.target_aspect_ratio[0], self.target_aspect_ratio[1])
    #     target_scale = np.random.uniform(self.target_scale[0], self.target_scale[1])
    #     context_aspect_ratio = self.context_aspect_ratio
    #     context_scale = np.random.uniform(self.context_scale[0], self.context_scale[1])

    #     y_student, y_teacher = self(x, target_aspect_ratio, target_scale, context_aspect_ratio, context_scale)
    #     loss = self.criterion(y_student, y_teacher)
    #     print('train_loss', loss)

    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     x = batch
    #     target_aspect_ratio = np.random.uniform(self.target_aspect_ratio[0], self.target_aspect_ratio[1])
    #     target_scale = np.random.uniform(self.target_scale[0], self.target_scale[1])
    #     context_aspect_ratio = self.context_aspect_ratio
    #     context_scale = np.random.uniform(self.context_scale[0], self.context_scale[1])

    #     y_student, y_teacher = self(x, target_aspect_ratio, target_scale, context_aspect_ratio, context_scale)
    #     loss = self.criterion(y_student, y_teacher)
    #     print('val_loss', loss)

    #     return loss

    # def predict_step(self, batch, batch_idx, dataloader_idx):
    #     target_aspect_ratio = np.random.uniform(self.target_aspect_ratio[0], self.target_aspect_ratio[1])
    #     target_scale = np.random.uniform(self.target_scale[0], self.target_scale[1])
    #     context_aspect_ratio = self.context_aspect_ratio
    #     context_scale = 1
    #     self.model.mode = "test"

    #     return self(batch, target_aspect_ratio, target_scale, context_aspect_ratio, context_scale) #just get teacher embedding

    # def on_after_backward(self):
    #     self.update_momentum(self.m)
    #     self.m += (self.m_start_end[1] - self.m_start_end[0]) / self.trainer.estimated_stepping_batches

    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    #     scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #         optimizer,
    #         max_lr=self.lr,
    #         total_steps=self.trainer.estimated_stepping_batches,
    #     )
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": scheduler,
    #             "interval": "step",
    #         },
    #     }


# if __name__ == '__main__':
#     dataset = D2VDataModule(dataset_path='data')

#     model = SJEPA(img_size=224, patch_size=16, in_chans=3, embed_dim=64, enc_heads=8, enc_depth=8, decoder_depth=6, lr=1e-3)

#     lr_monitor = LearningRateMonitor(logging_interval="step")
#     model_summary = ModelSummary(max_depth=2)

#     trainer = pl.Trainer(
#         accelerator='gpu',
#         devices=1,
#         precision=16,
#         max_epochs=10,
#         callbacks=[lr_monitor, model_summary],
#         gradient_clip_val=.1,
#     )

#     trainer.fit(model, dataset)
