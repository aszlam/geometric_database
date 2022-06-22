import torch


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def torch_choice(pop_size: int, num_samples: int, device:str = "cuda"):
    """Generate a random torch.Tensor (GPU) and sort it to generate indices."""
    return torch.argsort(torch.rand(pop_size, device="cuda")).int()[:num_samples]
