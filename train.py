import keras.backend as K
from segmentation_models import FPN
import os
from albumentations import (HorizontalFlip, ShiftScaleRotate, OneOf, Compose, RandomBrightnessContrast, RandomCrop)
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv2D,Input
from sklearn.model_selection import train_test_split
from keras.utils import Sequence

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
smooth = 1e-10
SEED = 42


class SegDataGenerator(Sequence):
    ''' Data generator class for segmentation
        Note:
            Used as data generator in fit_generator from keras.
            Includes support for augmentations via passing prepocessing function
            as preprocessing_function parameter. For example interface of preprocessing function
            see example_preprocess function.
        Args:
            input_directory (str): path to the folder where the input images are stored
            mask_directory (str): path to the folder where the masks are stored
            image_extention (str): extention of the input images files
            mask_extention (str): extention of the input masks files
            image_shape (tuple/list): target shape of the input images
            mask_shape (tuple/list): target shape of the masks
            batch_size (int): batch size
            preload_dataset (bool): if True input images and masks will be loaded to RAM (should be set to False if dataset if larger than available RAM)
            prob_aug (float): probability of getting augmented image
            preprocessing_function (func): function that performs preprocessing and augmentation (if needed) (see example_preprocess function)
            classes (int): number of segmented classes. Only 1 for RAM usage, more for live load
            classes_colors (dict): list of class colors
        Attributes:
            no public attributes
        '''
    def __init__(self,
                 input_directory, mask_directory,
                 image_extention='.jpg', mask_extention='.png',
                 image_shape=(256, 256, 3), mask_shape=(256, 256, 1),
                 batch_size=1, preload_dataset=False, prob_aug=0.5,
                 preprocessing_function=None, classes=1, classes_colors=None):

        self._dir = input_directory
        self._mask_dir = mask_directory
        self._img_shape = image_shape
        self._mask_shape = mask_shape
        self._iext = image_extention
        self._mext = mask_extention
        self._batch_size = batch_size
        self._in_files = list(filter(lambda x: x.endswith(self._iext), os.listdir(self._dir)))
        self._in_files.sort()
        self._mask_files = list(filter(lambda x: x.endswith(self._mext), os.listdir(self._mask_dir)))
        self._mask_files.sort()
        self._preload = preload_dataset
        self._prob_aug = prob_aug
        self._data = None
        self._masks = None
        self._h = 0
        self._w = 1
        self._c = 2
        self._classes = classes
        self._colors = classes_colors

        if (preprocessing_function is not None) and callable(preprocessing_function):
            self._preprocess = preprocessing_function
        else:
            self._preprocess = self._def_preprocess

        if self._preload:
            self._data = list()
            self._masks = list()
            for i, name in enumerate(self._in_files):
                img = cv2.imread(self._dir + name, cv2.IMREAD_UNCHANGED)
                mask = cv2.imread(self._mask_dir + self._mask_files[i], cv2.IMREAD_UNCHANGED)
                self._data.append(img)
                self._masks.append(mask)

    def __len__(self):
        return int(np.ceil(len(self._in_files) / float(self._batch_size)))

    def __getitem__(self, idx):

        batch_x = np.empty((self._batch_size, self._img_shape[self._h], self._img_shape[self._w], self._img_shape[self._c]), dtype='float32')
        batch_y = np.empty((self._batch_size, self._mask_shape[self._h], self._mask_shape[self._w], self._classes), dtype='float32')

        if self._preload:

            for i, img in enumerate(self._data[idx * self._batch_size:(idx + 1) * self._batch_size]):

                batch_img, batch_mask = self.__filter__(img, self._masks[idx])
                batch_x[i] = batch_img
                batch_y[i] = batch_mask

        else:

            for i, name in enumerate(self._in_files[idx * self._batch_size:(idx + 1) * self._batch_size]):

                img = cv2.imread(os.path.join(self._dir, name), cv2.IMREAD_UNCHANGED)
                mask = cv2.imread(os.path.join(self._mask_dir, self._mask_files[idx]), cv2.IMREAD_UNCHANGED)

                batch_img, batch_mask = self.__filter__(img, mask)
                batch_x[i] = batch_img
                batch_y[i] = batch_mask

        return batch_x, batch_y

    def __filter__(self, img, mask):
        inter = cv2.INTER_AREA

        if (img.shape[self._w] < self._img_shape[self._w]) or (img.shape[self._h] < self._img_shape[self._h]):
            inter = cv2.INTER_CUBIC

        batch_img = cv2.resize(img, dsize=(self._img_shape[self._w], self._img_shape[self._h]), interpolation=inter)
        batch_mask = cv2.resize(mask, dsize=(self._mask_shape[self._w], self._mask_shape[self._h]),
                                interpolation=inter)

        batch_img_a, batch_mask_a = self._preprocess(batch_img, batch_mask, self._prob_aug)

        if self._classes > 1:
            dashed_mask = np.empty((self._mask_shape[self._h], self._mask_shape[self._w], self._classes), dtype='float32')

            for color in self._colors:
                one_mask = cv2.inRange(batch_mask_a, color, color)
                np.append(dashed_mask, one_mask)

            return batch_img_a, batch_mask_a
        else:
            return batch_img_a, batch_mask_a

    @staticmethod
    def _def_preprocess(img, mask, prob_aug):
        return img, mask


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def jaccard_coef(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def dice_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


def strong_aug(p=0.5):
    return Compose([
        OneOf([
            ShiftScaleRotate(p=0.5, rotate_limit=10, scale_limit=0),
            HorizontalFlip(p=0.5)
        ]),
        RandomBrightnessContrast(p=0.5),
        RandomCrop(p=1, height=512, width=512)
    ], p=p)


def make_aug(image, mask, p):
    augmentation = strong_aug(p=p)
    data = {'image': image, 'mask': mask}
    augmented = augmentation(**data)
    return augmented['image'], augmented['mask']


def IOU(true, pred):
    iou_over_all = 0
    num_images = len(true)
    images_intersection = set(true).intersection(set(pred))
    for image_intersection in images_intersection:
        iou_per_image = 0
        y_true_categories = true[image_intersection]
        num_categories = len(y_true_categories)
        y_pred_categories = pred[image_intersection]
        categories_intersection = set(y_true_categories).intersection(set(y_pred_categories))
        iou_per_category = 0
        for category in categories_intersection:
            y_pred = y_pred_categories[category]
            y_true = y_true_categories[category]
            iou_per_category += compute_iou_pix(y_pred, y_true)
        iou_per_image += iou_per_category
        iou_per_image/=num_categories
        iou_over_all += iou_per_image
    return iou_over_all/num_images


def compute_area(data):
    """
    data: dict
    {x_1,segments_1, x_2:segments_2}
    """
    area = 0
    for segments in data.values():
        for segment in segments:
            area += max(segment)-min(segment)+1
    return area


def compute_iou_pix(y_pred, y_true):
    """
    y_pred, y_true: dict
    {x_1:segments_1, x_2:segments_2,...}
    """
    area_true = compute_area(y_true)
    area_pred = compute_area(y_pred)
    x_intersection = set(y_true).intersection(set(y_pred))
    area_intersection = 0
    for x in x_intersection:
        segments_true = y_true[x]
        segments_pred = y_pred[x]
        for segment_true in segments_true:
            max_segment_true = max(segment_true)
            min_segment_true = min(segment_true)
            del_list = []
            for segment_pred in segments_pred:
                max_segment_pred = max(segment_pred)
                min_segment_pred = min(segment_pred)
                if max_segment_true>=min_segment_pred and min_segment_true<=max_segment_pred:
                    seg_intersection = min(max_segment_true, max_segment_pred)-max(min_segment_true, min_segment_pred)+1
                    area_intersection += seg_intersection
                    if max_segment_true>=max_segment_pred and min_segment_true<=min_segment_pred:
                        del_list.append(segment_pred)
            for l in del_list:
                segments_pred.remove(l)

    area_union = area_true + area_pred - area_intersection

    return area_intersection/area_union


if __name__ == '__main__':

    generator = SegDataGenerator(TRAIN_PATH, ANNO_PATH, image_shape=(1216, 1936, 3), mask_shape=(1216, 1936, 3), preprocessing_function=make_aug, classes=len(CLASS_COLOR), classes_colors=CLASS_COLOR, prob_aug=1)
    input_layer = Input(shape=(256, 256, 3))
    model = FPN(
        backbone_name='resnet34',
        input_tensor=input_layer,
        encoder_weights='imagenet',
        classes=len(CLASS_COLOR),
        use_batchnorm=True,
        dropout=0.25,
        activation='softmax'
    )

    save_name = 'resnet34_test'
    callbacks_list = [
        ModelCheckpoint(
            save_name,
            verbose=1,
            save_best_only=True,
            mode='min',
            save_weights_only=True),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-5)
        ]

    model.compile(optimizer=Adam(1e-4), loss=dice_loss, metrics=[dice_coef, jaccard_coef, IOU])

    history = model.fit_generator(generator,
                                  steps_per_epoch=3000,
                                  epochs=10,
                                  verbose=1)

    model_json = model.to_json()
    json_file = open('model.json', 'w')
    json_file.write(model_json)
    json_file.close()
    print('Model saved!')

    K.clear_session()
    print('Cache cleared')
