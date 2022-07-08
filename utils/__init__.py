import torch


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def torch_choice(pop_size: int, num_samples: int, device: str = "cuda"):
    """Generate a random torch.Tensor (GPU) and sort it to generate indices."""
    return torch.argsort(torch.rand(pop_size, device="cuda")).int()[:num_samples]


def generate_batch_mask(batch_size: int, masking_prob: float = 0.5):
    # This is 1 where we should mask
    return torch.rand(batch_size) < masking_prob


def mask_batch_with_mask_token(
    batch: torch.Tensor, mask_token: torch.Tensor, batch_mask_indices: torch.Tensor
):
    # batch_mask_indices is assumed to be a boolean tensor indicating where the masking should
    # happen. On those places, we replace the batch data with the mask_token data.
    new_batch = torch.zeros_like(batch, device=batch.device)
    batch_mask_indices = batch_mask_indices.to(batch.device)
    new_batch[batch_mask_indices] = mask_token
    new_batch[~batch_mask_indices] = batch[~batch_mask_indices]
    return new_batch


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))
