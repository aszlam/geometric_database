import matplotlib.pyplot as plt
from dataloaders.habitat_loaders import HabitatViewDataset, HabitatLocationDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # Replace with the path to your scene file
    SCENE_FILEPATH = [
        "/checkpoint/notmahi/data/replica_dataset/room_0/habitat/mesh_semantic.ply",
    ]
    BATCH_SIZE = 6

    # Run basic sanity test on the dataloader.
    dataset = HabitatViewDataset(
        habitat_scenes=SCENE_FILEPATH,
        pose_extractor_grid_size=5,
        height_levels=3,
    )

    # Create a Dataloader to batch and shuffle our data
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    def show_batch(sample_batch):
        def show_row(imgs, batch_size, img_type):
            plt.figure(figsize=(12, 8))
            for i, img in enumerate(imgs):
                ax = plt.subplot(1, batch_size, i + 1)
                ax.axis("off")
                if img_type == "rgb":
                    plt.imshow(img.numpy().transpose(1, 2, 0))
                elif img_type in ("depth", "truth"):
                    plt.imshow(img.numpy())

            # plt.savefig(f"{k}_fig.png", dpi=200)

        batch_size = len(sample_batch["rgb"])
        for k in sample_batch.keys():
            show_row(sample_batch[k], batch_size, k)

    # _, sample_batch = next(enumerate(dataloader))
    # show_batch(sample_batch)

    location_dataset = HabitatLocationDataset(habitat_view_ds=dataset)
    # Create a Dataloader to batch and shuffle our data
    xyz_dataloader = DataLoader(location_dataset, batch_size=BATCH_SIZE, shuffle=True)
    _, sample_batch = next(enumerate(xyz_dataloader))
    print(sample_batch)
