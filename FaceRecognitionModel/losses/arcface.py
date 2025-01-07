import math

import torch
from torch import nn
import torch.nn.functional as F


class ArcFace(nn.Module):
    """
    ArcFace loss function.
    """

    def __init__(
        self,
        num_classes: int = 360232,
        dim: int = 512,
        scale: float = 64.0,
        margin: float = 0.5,
    ):
        super(ArcFace, self).__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embs: torch.tensor, idtys: torch.tensor) -> torch.tensor:
        idtys = idtys.unsqueeze(1)
        # get the cosine of the angles between the embeddings and the weights
        # size (batch_size, num_classes)
        cosine = F.linear(F.normalize(embs), F.normalize(self.weight))

        # get the sine of the angles
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

        # calculate the marginal penalty term cos(angle + margin)
        phi = cosine * math.cos(self.margin) - sine * math.sin(self.margin)
        phi = torch.where(cosine > 0, phi, cosine)
        
        # create one hot encoding
        one_hot = F.one_hot(idtys, num_classes=self.num_classes)

        # get the logits output
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # scale the logits and calculate the loss
        loss = F.cross_entropy(self.scale * logits, idtys)

        return loss
