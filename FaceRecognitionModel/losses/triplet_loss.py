import torch
import torch.nn as nn


def normalize(x: torch.tensor, axis=-1):
    x = 1.0 * x / (torch.norm(x, p=2, dim=axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def _pairwise_distances(embs: torch.tensor, squared: bool = False):
    """
    Compute the 2D matrix of L2 norm distances between all the embeddings.

    Args:
        embs (torch.tensor): tensor of shape (batch_size, emb_dim)
        squared (bool): if True, output is the pairwise squared euclidean distance matrix,
        if False, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: return a torch tensor matrix of shape (batch_size, batch_size) representing
        the pairwise euclidean distances between all the embeddings.
        A_ij represents dot product between embeddings at index i and j.
    """
    # Compute the dot product between all embeddings, matrix of shape (batch_size, batch_size)
    dot_prod = embs @ embs.t()

    # Calculate the squared L2 norm of each embedding, matrix of shape (batch_size, )
    square_norm = torch.diag(dot_prod)

    # Compute the pairwse distance matrix ||a-b||^2 = ||a||^2 - 2<a, b> + ||b||^2
    # shape (batch_size, batch_size)
    pairwise_distances = (
        square_norm.unsqueeze(0) - 2.0 * dot_prod + square_norm.unsqueeze(1)
    )

    # Ensure no negative values due to floating point errors
    pairwise_distances[pairwise_distances < 0] = 0

    if not squared:
        mask = pairwise_distances.eq(0).float()
        pairwise_distances = pairwise_distances + mask * 1e-16
        pairwise_distances = (1.0 - mask) * torch.sqrt(pairwise_distances)

    return pairwise_distances


def _get_triplet_mask(idtys: torch.tensor):
    """
    Compute a 3D mask where mask[a, p, n] is True iff the triple (a, p, n) is valid.
    A triple (i, j, k) is valid if:
        - i, j, k are distinct
        - idtys[i] == idtys[j] and idtys[i] != idtys[k]

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
    i_neq_j = indices_unequal.unsqueeze(2)
    i_neq_k = indices_unequal.unsqueeze(1)
    j_neq_k = indices_unequal.unsqueeze(0)

    distinct_indices = i_neq_j & i_neq_k & j_neq_k

    # Create a tensor where idtys[i] == idtys[j] and idtys[i] != idtys[k]
    idtys_equal = idtys.unsqueeze(0) == idtys.unsqueeze(1)
    i_same_j = idtys_equal.unsqueeze(2)
    i_same_k = idtys_equal.unsqueeze(1)

    valid_idtys = i_same_j & ~i_same_k

    triplet_mask = distinct_indices.type_as(valid_idtys) & valid_idtys

    return triplet_mask


def _get_anchor_positive_mask(idtys: torch.tensor):
    """
    Return a 2D mask where mask[a, p] is True iff a and p are distinct and have the same identity.

    Args:
        idtys (torch.long): tensor of shape (batch_size, ) containing the identity of the embeddings.

    Returns:
        anchor_positive_mask (torch.tensor): tensor of shape (batch_size, batch_size) where anchor_positive_mask[a, p] is True
    """
    # Get the batch size
    batch_size = idtys.size(0)

    # Create a tensor where i, j are distinct first
    indices_equal = torch.eye(batch_size).bool()  # 2D identity matrix
    indices_unequal = ~indices_equal

    # Create a tensor where idtys[i] == idtys[j]
    idtys_equal = idtys.unsqueeze(0) == idtys.unsqueeze(1)

    anchor_positive_mask = indices_unequal.type_as(idtys_equal) & idtys_equal

    return anchor_positive_mask


def _get_anchor_negative_mask(idtys: torch.tensor):
    """
    Return a 2D mask where mask[a, n] is True iff a and n have distinct identities.

    Args:
        idtys (torch.long): tensor of shape (batch_size, ) containing the identity of the embeddings.

    Returns:
        anchor_negative_mask (torch.tensor): tensor of shape (batch_size, batch_size) where anchor_negative_mask[a, n] is True
    """
    # Create a tensor where idtys[i] != idtys[j]
    anchor_negative_mask = idtys.unsqueeze(0) != idtys.unsqueeze(1)

    return anchor_negative_mask


class BatchHardTripletLoss(nn.Module):
    def __init__(
        self, margin: float, squared: bool = False, normalize_features: bool = False
    ):
        super(BatchHardTripletLoss, self).__init__()
        """
        margin (float): margin for triplet loss
        squared (bool): if True, output is the pairwise squared euclidean distance matrix,
        normalize_feature (bool): if True, normalize the feature embeddings before computing the distance matrix
        """

        self.margin = margin
        self.squared = squared
        self.normalize_features = normalize_features

    def forward(self, embs: torch.tensor, idtys: torch.tensor) -> torch.tensor:
        """
        Calculate the triplet loss over a batch of embeddings. For each anchor, we find the hardest positive and hardest
        negative to form a triplet.

        Args:
            embs (torch.tensor): tensor of shape (batch_size, emb_dim) containing the embeddings.
            idtys (torch.long): tensor of shape (batch_size, ) containing the identity of the embeddings.

        Returns:
            triplet_loss (torch.tensor): scalar tensor containing the triplet loss.
        """
        if self.normalize_features:
            embs = normalize(embs, axis=-1)

        # Compute the pairwise distance matrix
        pairwise_distances = _pairwise_distances(embs, squared=self.squared)

        # Get the hardest positive
        mask_anchor_positive = _get_anchor_positive_mask(idtys).float()
        anchor_positive_dist = mask_anchor_positive * pairwise_distances
        hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

        # Get the hardest negative
        mask_anchor_negative = _get_anchor_negative_mask(idtys).float()
        max_anchor_dist, _ = pairwise_distances.max(1, keepdim=True)
        anchor_negative_dist = pairwise_distances + max_anchor_dist * (
            1.0 - mask_anchor_negative
        )
        hardest_negative_dist, _ = anchor_negative_dist.min(
            1, keepdim=True
        )  # use min because the hardest negative is the smallest distance

        # Calculate the triplet loss
        hard_triplet_loss = torch.relu(
            hardest_positive_dist - hardest_negative_dist + self.margin
        )
        batch_hard_triplet_loss = hard_triplet_loss.mean()

        return batch_hard_triplet_loss


class BatchAllTripletLoss(nn.Module):
    def __init__(
        self, margin: float, squared: bool = False, normalize_features: bool = False
    ):
        super(BatchAllTripletLoss, self).__init__()
        """
        margin (float): margin for triplet loss
        squared (bool): if True, output is the pairwise squared euclidean distance matrix,
        normalize_feature (bool): if True, normalize the feature embeddings before computing the distance matrix
        """

        self.margin = margin
        self.squared = squared
        self.normalize_features = normalize_features

    def forward(self, embs: torch.tensor, idtys: torch.tensor) -> torch.tensor:
        """
        Calculate the triplet loss over a batch of embeddings. For each anchor, we find the hardest positive and hardest
        negative to form a triplet.

        Args:
            embs (torch.tensor): tensor of shape (batch_size, emb_dim) containing the embeddings.
            idtys (torch.long): tensor of shape (batch_size, ) containing the identity of the embeddings.

        Returns:
            triplet_loss (torch.tensor): scalar tensor containing the triplet loss.
        """
        if self.normalize_features:
            embs = normalize(embs, axis=-1)

        # Compute the pairwise distance matrix
        pairwise_distances = _pairwise_distances(embs, squared=self.squared)

        # Get the positive and negative distances
        anchor_positive_dist = pairwise_distances.unsqueeze(2)
        anchor_negative_dist = pairwise_distances.unsqueeze(1)

        # Compute the triplet loss
        all_triplet_loss = torch.relu(
            anchor_positive_dist - anchor_negative_dist + self.margin
        )

        # Put the invalid triplets to zero
        triplet_mask = _get_triplet_mask(idtys).float()
        all_triplet_loss = all_triplet_loss * triplet_mask

        # Count the number of valid triplets with a positive loss
        num_pos_triplets = all_triplet_loss.gt(1e-16).float().sum()
        batch_all_triplet_loss = all_triplet_loss.sum() / (num_pos_triplets + 1e-16)

        return batch_all_triplet_loss
