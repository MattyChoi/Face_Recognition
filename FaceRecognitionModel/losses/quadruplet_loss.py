import torch
from torch import nn

from losses.triplet_loss import (
    _get_anchor_positive_mask,
    _get_anchor_negative_mask,
    _get_triplet_mask,
    _pairwise_distances,
    normalize,
)


def _get_quadruplet_mask(idtys: torch.tensor):
    """
    Compute a 4D mask where mask[a, p, n1, n2] is True iff the quadruplet (a, p, n1, n2) is valid.
    A quadruple (i, j, k, l) is valid if:
        - i, j, k, l are distinct
        - idtys[i] == idtys[j], idtys[i] != idtys[k], idtys[i] != idtys[l], idtys[k] != idtys[l]

    Args:
        idtys (torch.long): tensor of shape (batch_size, ) containing the identity of the embeddings.

    Returns:
        triplet_mask (torch.tensor): tensor of shape (batch_size, batch_size, batch_size) where triplet_mask[a, p, n] is True
    """
    # Get the batch size
    batch_size = idtys.size(0)

    # Create a tensor where i, j, k are distinct first
    indices_equal = torch.eye(batch_size).bool()  # 2D identity matrix
    indices_unequal = ~indices_equal

    # Only need to check that i != j, other cases are covered by second condition
    distinct_indices = indices_unequal[(...,) + (None,) * 2]

    # Create a tensor where idtys follow i=j, k!=l, i!=l, i!=k
    idtys_equal = idtys.unsqueeze(0) == idtys.unsqueeze(1)
    i_same_j = idtys_equal[(...,) + (None,) * 2]
    j_same_k = idtys_equal[(None,) + (...,) + (None,)]
    k_same_l = idtys_equal[(None,) * 2]
    expand_shape = (slice(None),) + (None,) * 2 + (slice(None),)
    i_same_l = idtys_equal[expand_shape]

    valid_idtys = i_same_j & ~j_same_k & ~k_same_l & ~i_same_l

    quadruplet_mask = distinct_indices.type_as(valid_idtys) & valid_idtys

    return quadruplet_mask


class BatchHardQuadrupletLoss(nn.Module):
    """
    Batch hard quadruplet loss function.
    """

    def __init__(
        self,
        margin_triple: float = 0.2,
        margin_inter: float = 0.1,
        squared: bool = False,
        normalize_features: bool = False,
    ):
        super(BatchHardQuadrupletLoss, self).__init__()
        self.margin_triple = margin_triple
        self.margin_inter = margin_inter
        self.squared = squared
        self.normalize_features = normalize_features
        """
        margin_triple (float): margin for triplet loss term
        margin_inter (float): margin for inter-class loss term
        squared (bool): if True, output is the pairwise squared euclidean distance matrix,
        normalize_feature (bool): if True, normalize the feature embeddings before computing the distance matrix
        """

    def forward(self, embs: torch.tensor, idtys: torch.tensor) -> torch.tensor:
        """
        Calculate the quadruplet loss over a batch of embeddings. For each anchor, we find the hardest positive and hardest
        negative to form a triplet.

        Args:
            embs (torch.tensor): tensor of shape (batch_size, emb_dim) containing the embeddings.
            idtys (torch.long): tensor of shape (batch_size, ) containing the identity of the embeddings.

        Returns:
            triplet_loss (torch.tensor): scalar tensor containing the triplet loss.
        """
        # Get the batch size
        batch_size = idtys.size(0)

        if self.normalize_features:
            embs = normalize(embs, axis=-1)

        # Compute the pairwise distance matrix
        pairwise_distances = _pairwise_distances(embs, squared=self.squared)

        # Get the hardest positive
        mask_anchor_positive = _get_anchor_positive_mask(idtys).float()
        anchor_positive_dist = mask_anchor_positive * pairwise_distances
        hardest_positive_dist, hardest_pos_ind = anchor_positive_dist.max(
            1, keepdim=True
        )

        # Get the hardest negative
        mask_anchor_negative = _get_anchor_negative_mask(idtys).float()
        max_anchor_dist, _ = pairwise_distances.max(1, keepdim=True)
        anchor_negative_dist = pairwise_distances + max_anchor_dist * (
            1.0 - mask_anchor_negative
        )
        hardest_negative_dist, hardest_neg_ind = anchor_negative_dist.min(
            1, keepdim=True
        )  # use min because the hardest negative is the smallest distance

        # Calculate the triplet loss using the largest d(a, p) and smallest d(a, n)
        triplet_loss = torch.relu(
            hardest_positive_dist - hardest_negative_dist + self.margin_triple
        )

        # Get all d(a, p) and d(n1, n2) for inter-class loss
        anchor_pos_dist = pairwise_distances[(...,) + (None,) * 2]
        neg1_neg2_dist = pairwise_distances[(None,) * 2]

        # Calculate inter-class loss, 4D tensor of shape (batch_size, batch_size, batch_size, batch_size)
        mask = _get_quadruplet_mask(idtys).float()
        inter_class_loss = (
            torch.relu(anchor_pos_dist - neg1_neg2_dist + self.margin_inter) * mask
        )

        # Get the hardest batch triplets
        inter_class_loss = inter_class_loss[
            torch.arange(batch_size), hardest_pos_ind, hardest_neg_ind
        ]

        # Get the largest inter-class losses to get the batch hard inter-class losses
        inter_class_loss, _ = inter_class_loss.max(1, keepdim=True)

        # Calulate the final batch hard quadruplet loss
        batch_hard_quadruplet_loss = (triplet_loss + inter_class_loss).mean()

        return batch_hard_quadruplet_loss


class BatchAllQuadrupletLoss(nn.Module):
    """
    Batch all quadruplet loss function.
    """

    def __init__(
        self,
        margin_triple: float = 0.2,
        margin_inter: float = 0.1,
        squared: bool = False,
        normalize_features: bool = False,
    ):
        super(BatchAllQuadrupletLoss, self).__init__()
        self.margin_triple = margin_triple
        self.margin_inter = margin_inter
        self.squared = squared
        self.normalize_features = normalize_features
        """
        margin_triple (float): margin for triplet loss term
        margin_inter (float): margin for inter-class loss term
        squared (bool): if True, output is the pairwise squared euclidean distance matrix,
        normalize_feature (bool): if True, normalize the feature embeddings before computing the distance matrix
        """

    def forward(self, embs: torch.tensor, idtys: torch.tensor) -> torch.tensor:
        """
        Calculate the quadruplet loss over a batch of embeddings. For each anchor, we find the hardest positive and hardest
        negative to form a triplet.

        Args:
            embs (torch.tensor): tensor of shape (batch_size, emb_dim) containing the embeddings.
            idtys (torch.long): tensor of shape (batch_size, ) containing the identity of the embeddings.

        Returns:
            triplet_loss (torch.tensor): scalar tensor containing the triplet loss.
        """
        # Get the batch size
        batch_size = idtys.size(0)

        if self.normalize_features:
            embs = normalize(embs, axis=-1)

        # Compute the pairwise distance matrix
        pairwise_distances = _pairwise_distances(embs, squared=self.squared)

        # Calculate all distances
        anchor_pos_dist = pairwise_distances[(...,) + (None,) * 2]
        expand_shape = (slice(None),) + (None,) + (slice(None),) + (None,)
        anchor_neg1_dist = pairwise_distances[expand_shape]
        neg1_neg2_dist = pairwise_distances[(None,) * 2]

        # Calculate the triplet loss
        mask = _get_quadruplet_mask(idtys).float()
        triplet_loss = (
            torch.relu(anchor_pos_dist - anchor_neg1_dist + self.margin_triple) * mask
        )

        # Compute the inter-class loss term
        inter_class_loss = (
            torch.relu(anchor_pos_dist - neg1_neg2_dist + self.margin_inter) * mask
        )

        # Calculate the quadruplet loss
        quadruplet_loss = triplet_loss + inter_class_loss

        valid_quadruplets = quadruplet_loss[quadruplet_loss > 1e-16]
        num_positive_quadruplets = valid_quadruplets.size(0)

        return quadruplet_loss.sum() / (num_positive_quadruplets + 1e-16)
