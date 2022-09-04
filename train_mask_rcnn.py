from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import hydra
import wandb
from omegaconf import OmegaConf

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
        num_segmented_images: int = 5,
        valid_classes: Optional[List] = None,
        mode: str = "instance_segmentation",
    ):
        self.habitat_dataset = habitat_dataset
        self.crowd_classes = {"wall", "floor", "ceiling", "background"}
        self.transforms = transforms

        assert mode in ["instance_segmentation", "semantic_segmentation"]
        self.mode = mode

        if "instance" in self.mode:
            self.id_to_name: Dict[int, str] = habitat_dataset.dataset.id_to_name
            choice_fn = self.choose_images_to_inst_segment
        else:
            self.id_to_name: Dict[int, str] = habitat_dataset.dataset._id_to_name
            choice_fn = self.get_best_sem_segmented_images

        self.chosen_indices, self.chosen_classes = choice_fn(
            habitat_dataset, num_segmented_images
        )
        if valid_classes is None:
            self.valid_classes = self.chosen_classes
        else:
            self.valid_classes = valid_classes

    def get_best_sem_segmented_images(
        self, habitat_view_dataset, num_segmented_images=50
    ) -> Tuple[List[int], List[int]]:
        dataset = habitat_view_dataset
        # Using depth as a proxy for object diversity in a scene.
        num_objects_and_images = [
            (dataset[idx]["depth"].max() - dataset[idx]["depth"].min(), idx)
            for idx in range(len(dataset))
        ]
        sorted_num_object_and_img = sorted(
            num_objects_and_images, key=lambda x: x[0], reverse=True
        )
        # All classes are valid for semantic segmentation.
        return [x[1] for x in sorted_num_object_and_img[:num_segmented_images]], list(
            self.id_to_name.keys()
        )

    def choose_images_to_inst_segment(
        self, habitat_view_dataset, num_segmented_image=5
    ) -> Tuple[List[int], List[int]]:
        best_set_so_far = set()
        chosen_images_so_far = set()
        num_chosen_images = num_segmented_image
        dataset = habitat_view_dataset
        unique_sets = [
            set(torch.unique(dataset[idx][self.mode]).tolist())
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
        mask = habitat_data_dict[self.mode].numpy()
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
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if xmin < xmax and ymin < ymax:
                final_masks.append(masks[i])
                boxes.append([xmin, ymin, xmax, ymax])
                if (obj_ids[i] + 1) not in self.valid_classes:
                    final_labels.append(0)  # Treat as background class.
                else:
                    final_labels.append(obj_ids[i] + 1)
        if len(final_masks) == 0:
            # There were no objects in the scene, pure noise?
            final_masks.append(np.ones_like(masks[0]))
            final_labels.append(0)
            # Add the whole image as background.
            boxes.append([0, 0, final_masks[-1].shape[0], final_masks[-1].shape[1]])

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
        for inst_id in labels:
            inst_name = self.id_to_name.get(inst_id.item() - 1, "background")
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


MODEL_DICT = {
    "FCN_ResNet50": torchvision.models.segmentation.fcn_resnet50,
    "FCN_ResNet101": torchvision.models.segmentation.fcn_resnet101,
    "Faster_RCNN_ResNet50_FPN": torchvision.models.detection.maskrcnn_resnet50_fpn,
    "DeepLabV3_ResNet50": torchvision.models.segmentation.deeplabv3_resnet50,
    "DeepLabV3_ResNet101": torchvision.models.segmentation.deeplabv3_resnet101,
}


def get_model_segmentation(model_name, num_classes):
    # load an instance segmentation model pre-trained on COCO
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    model_class = MODEL_DICT.get(
        model_name, torchvision.models.detection.maskrcnn_resnet50_fpn
    )
    model = model_class(pretrained=True)

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
    base_filename = cfg.scene.filepath
    dataset = HabitatViewDataset(
        habitat_scenes=base_filename,
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
    num_train_images = (
        cfg.num_inst_segmented_images
        if "inst" in cfg.two_dim_models.mode
        else cfg.num_sem_segmented_images
    )
    dataset_train = HabitatSegmentationDataset(
        view_train_dataset,
        get_transform(train=True),
        num_segmented_images=num_train_images,
        mode=cfg.two_dim_models.mode,
    )
    dataset_test = HabitatSegmentationDataset(
        view_test_dataset,
        get_transform(train=False),
        num_segmented_images=train_split_size,
        valid_classes=dataset_train.valid_classes,
        mode=cfg.two_dim_models.mode,
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
    if cfg.deterministic_id:
        run_id = f"2d_{cfg.scene.base}_{cfg.two_dim_models.model_class}_{cfg.scene.image_size}i{cfg.num_inst_segmented_images}s{cfg.num_sem_segmented_images}w{cfg.num_web_segmented_images}"
    else:
        run_id = wandb.util.generate_id()
    wandb.init(
        project=cfg.two_dim_models.project,
        id=run_id,
        tags=[
            f"model/{cfg.two_dim_models.model_class}",
            f"scene/{cfg.scene.base}",
        ],
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # get the model using our helper function
    model = get_model_segmentation(cfg.two_dim_models.model_class, num_classes)

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
        to_log = {}
        # train for one epoch, printing every 10 iterations
        metric_logger = train_one_epoch(
            model, optimizer, data_loader, device, epoch, print_freq=10
        )
        for name, meter in metric_logger.meters.items():
            to_log[f"{name}_median"] = meter.median
            to_log[f"{name}_avg"] = meter.global_avg
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        if epoch % eval_every == 0 or epoch == (num_epochs - 1):
            _, summary_results = evaluate(model, data_loader_test, device=device)
            # Log summary results to WandB.
            # We only care about the segmentation results, so we are going for that
            segm_results = summary_results[1]
            avg_precision = segm_results[0]
            avg_recall = segm_results[6]
            to_log.update(
                {
                    "test/avg_precision": avg_precision,
                    "test/avg_recall": avg_recall,
                }
            )
        wandb.log(to_log)
    print("That's it!")


if __name__ == "__main__":
    main()
