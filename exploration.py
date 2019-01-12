import os
from albumentations import (HorizontalFlip, ShiftScaleRotate, OneOf, Compose, RandomBrightnessContrast, RandomCrop)
import cv2
import matplotlib.pyplot as plt
import numpy as np

PATH = 'E:/datasets/signate'
TRAIN_PATH = os.path.join(PATH, 'seg_train_images')
ANNO_PATH = os.path.join(PATH, 'seg_train_annotations')
CLASS_COLOR = {
    'Car': [0, 0, 255],
    'Bus': [193, 214, 0],
    'Truck': [180, 0, 129],
    'SVehicle': [255, 121, 166],
    'Pedestrian': [255, 0, 0],
    'Motorbike': [65, 166, 1],
    'Bicycle': [208, 149, 1],
    'Signal': [255, 255, 0],
    'Signs': [255, 134, 0],
    'Sky': [0, 152, 225],
    'Building': [0, 203, 151],
    'Natural': [85, 255, 50],
    'Wall': [92, 136, 125],
    'Lane': [69, 47, 142],
    'Ground': [136, 45, 66],
    'Sidewalk': [0, 255, 255],
    'RoadShoulder': [215, 0, 255],
    'Obstacle': [180, 131, 135],
    'others': [81, 99, 0],
    'own': [86, 62, 67]
}


def strong_aug(p=0.5):
    return Compose([
        OneOf([
            ShiftScaleRotate(p=0.5, rotate_limit=10, interpolation=cv2.INTER_CUBIC, scale_limit=0),
            HorizontalFlip(p=0.5)
        ]),
        RandomBrightnessContrast(p=0.5),
        RandomCrop(p=1, height=512, width=512)
    ], p=p)


if __name__ == '__main__':
    train_list = os.listdir(TRAIN_PATH)
    test_label = train_list[0]

    image = cv2.imread(os.path.join(TRAIN_PATH, test_label), cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(os.path.join(ANNO_PATH, test_label.replace('.jpg', '.png')), cv2.IMREAD_UNCHANGED)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    # mask_in_range = cv2.inRange(mask, np.asarray(CLASS_COLOR['own']), np.asarray(CLASS_COLOR['own']))

    augmentation = strong_aug(p=1)
    data = {"image": image, "mask": mask}
    augmented = augmentation(**data)
    image_a, mask_a = augmented["image"], augmented["mask"]

    fig, axes = plt.subplots(2, 2)
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('image')
    axes[1, 0].imshow(image_a)
    axes[1, 0].set_title('aug image')
    axes[0, 1].imshow(mask)
    axes[0, 1].set_title('mask')
    axes[1, 1].imshow(mask_a)
    axes[1, 1].set_title('aug mask')
    # axes[1].imshow(mask_in_range)

    plt.show()
