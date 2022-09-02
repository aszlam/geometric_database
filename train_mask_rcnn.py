from typing import Dict, List, Optional
import numpy as np
import torch
import hydra

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from dataloaders.habitat_loaders import HabitatViewDataset

import einops
from torch.utils.data import Dataset, random_split, DataLoader

import segmentation_model.transforms as T
from segmentation_model.engine import train_one_epoch, evaluate
import segmentation_model.utils as utils


class HabitatSegmentationDataset(Dataset):
    def __init__(
        self,
        habitat_dataset: HabitatViewDataset,
        transforms,
        num_inst_segmented_images: int = 5,
        valid_classes: Optional[List] = None,
    ):
        self.habitat_dataset = habitat_dataset
        self.chosen_indices, self.chosen_classes = self.choose_images_to_inst_segment(
            habitat_dataset, num_inst_segmented_images
        )
        self.id_to_name: Dict[int, str] = habitat_dataset.dataset.id_to_name
        self.crowd_classes = {"wall", "floor", "ceiling"}
        self.transforms = transforms
        if valid_classes is None:
            self.valid_classes = self.chosen_classes
        else:
            self.valid_classes = valid_classes

    def choose_images_to_inst_segment(
        self, habitat_view_dataset, num_segmented_image=5
    ):
        best_set_so_far = set()
        chosen_images_so_far = set()
        num_chosen_images = num_segmented_image
        dataset = habitat_view_dataset
        unique_sets = [
            set(torch.unique(dataset[idx]["instance_segmentation"]).tolist())
            for idx in range(len(dataset))
        ]
        for _ in range(num_chosen_images):
            best_set_index = -1
            best_set_score = -1
            for idx in range(len(dataset)):
                if idx in chosen_images_so_far:
                    # Only consider new images.
                    continue
                current_new_set_under_consideration = unique_sets[idx]
                current_set_score = len(
                    current_new_set_under_consideration - best_set_so_far
                )
                if current_set_score > best_set_score:
                    best_set_score = current_set_score
                    best_set_index = idx

            chosen_images_so_far.add(best_set_index)
            best_set_so_far = best_set_so_far | unique_sets[best_set_index]

        chosen_images_so_far = list(chosen_images_so_far)
        best_set_so_far = [x + 1 for x in best_set_so_far]
        return chosen_images_so_far, best_set_so_far

    def get_chosen_classes(self):
        return self.chosen_classes

    def __len__(self):
        return len(self.chosen_indices)

    def __getitem__(self, idx):
        habitat_image_index = self.chosen_indices[idx]
        habitat_data_dict = self.habitat_dataset[habitat_image_index]

        # load images and masks
        img = einops.rearrange(habitat_data_dict["rgb"][..., :3], "h w c -> c h w")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = habitat_data_dict["instance_segmentation"].numpy()
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        # obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        final_masks = []
        final_labels = []

        for i in range(num_objs):
            if (obj_ids[i] + 1) not in self.valid_classes:
                continue
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if xmin < xmax and ymin < ymax:
                boxes.append([xmin, ymin, xmax, ymax])
                final_masks.append(masks[i])
                final_labels.append(obj_ids[i] + 1)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        # Increase class label by 1 since we have no background
        labels = torch.as_tensor(final_labels)
        masks = torch.as_tensor(np.stack(final_masks, axis=0), dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        # Figure out the crowd instance

        iscrowd = []
        for inst_id in final_labels:
            inst_name = self.id_to_name[inst_id.item() - 1]
            inst_name = (
                inst_name.replace("-", " ").replace("_", "").lower().lstrip().rstrip()
            )
            if inst_name in self.crowd_classes:
                iscrowd.append(1)
            else:
                iscrowd.append(0)

        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model


def get_transform(train):
    transforms = []
    # transforms.append(T.ToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


@hydra.main(version_base="1.2", config_path="configs", config_name="train.yaml")
def main(cfg):
    # train on the GPU or on the CPU, if a GPU is not available
    device = (
        torch.device(cfg.device) if torch.cuda.is_available() else torch.device("cpu")
    )

    # our dataset has two classes only - background and person
    # use our dataset and defined transformations
    base_filename = cfg.scene.base
    dataset = HabitatViewDataset(
        habitat_scenes=[base_filename],
        pose_extractor_grid_size=cfg.scene.grid_size,
        height_levels=0,
        canonical_object_ids=True,
        image_size=(cfg.scene.image_size, cfg.scene.image_size),
    )

    num_classes = (
        len(dataset.id_to_name) + 1
    )  # Since it expects a background class but we have none.
    train_split_size = len(dataset) // 2
    view_train_dataset, view_test_dataset = random_split(
        dataset,
        lengths=[train_split_size, len(dataset) - train_split_size],
    )
    dataset_train = HabitatSegmentationDataset(
        view_train_dataset,
        get_transform(train=True),
        num_inst_segmented_images=train_split_size,
    )
    dataset_test = HabitatSegmentationDataset(
        view_test_dataset,
        get_transform(train=False),
        num_inst_segmented_images=train_split_size,
        valid_classes=dataset_train.valid_classes,
    )

    # define training and validation data loaders
    data_loader = DataLoader(
        dataset_train,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        collate_fn=utils.collate_fn,
    )

    data_loader_test = DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=utils.collate_fn,
    )

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = cfg.epochs
    eval_every = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        if epoch % eval_every == 0 or epoch == (num_epochs - 1):
            evaluate(model, data_loader_test, device=device)

    print("That's it!")


if __name__ == "__main__":
    main()
