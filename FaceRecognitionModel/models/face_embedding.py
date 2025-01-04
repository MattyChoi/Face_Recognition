import os
import sys

# Import modules from base directory
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
from backbones.resnet import resnet18


class FaceEmbeddingModel(nn.Module):
    """
    Constructs a Face embedding model using a given classification backbone model

    Args:
        backbone (nn.Module): Backbone model to be used for the face embedding model. Defaults to resnet18().
        embedding_dimension (int): Required dimension of the resulting embedding layer that is outputted by the model.
            using triplet loss. Defaults to 512.
    """

    def __init__(self, backbone=resnet18(), embedding_dimension=512):
        super().__init__()
        
        self.backbone = backbone

        # Output embedding
        input_features_fc_layer = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(
            input_features_fc_layer, embedding_dimension, bias=False
        )

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        embedding = self.backbone(images)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding
