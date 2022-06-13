from models.view_encoders.timm_view_encoder import TimmViewEncoder
import torch


def test_timm_view_encoder():
    """
    Test the input and the output shapes of the TimmViewEncoeer
    """
    view_shape = (224, 224)
    semantic_classes = 10
    BATCH_SIZE = 10
    REP_LEN = 256
    device = "cuda"

    view_encoder = TimmViewEncoder(
        view_shape=view_shape,
        representation_length=REP_LEN,
        timm_class="resnet18",
        semantic_embedding_len=16,
        num_semantic_classes=semantic_classes,
        device=device,
    )

    depth = torch.randn((BATCH_SIZE,) + view_shape)
    rgba = torch.randn((BATCH_SIZE, 4) + view_shape)
    semantic_segmentation = torch.randint_like(depth, high=semantic_classes).long()

    sample_batch = {
        "rgb": rgba,
        "truth": semantic_segmentation,
        "depth": depth,
    }

    results = view_encoder({k: v.to(device) for k, v in sample_batch.items()})
    assert results.shape[0] == BATCH_SIZE
    assert results.shape[-1] == REP_LEN
