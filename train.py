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


class DataLoader:
    def __init__(self, shape):
        self.seed = SEED
        self.shape = shape

    def generator(self, x_train, y_train, batch_size):
        data_generator = ImageDataGenerator(
            width_shift_range=0.25,
            height_shift_range=0.25,
            horizontal_flip=True,
            rotation_range=10,
            rescale=1. / 255).flow(x_train, x_train, batch_size, seed=self.seed)
        mask_generator = ImageDataGenerator(
            width_shift_range=0.25,
            height_shift_range=0.25,
            horizontal_flip=True,
            rotation_range=10,
            rescale=1. / 255).flow(y_train, y_train, batch_size, seed=self.seed)
        while True:
            x_batch, _ = data_generator.next()
            y_batch, _ = mask_generator.next()
            yield x_batch, y_batch

    def val_generator(self, x_train, y_train, batch_size=1):
        data_generator = ImageDataGenerator(
            rescale=1. / 255).flow(x_train, x_train, batch_size, seed=self.seed)
        mask_generator = ImageDataGenerator(
            rescale=1. / 255).flow(y_train, y_train, batch_size, seed=self.seed)
        while True:
            x_batch, _ = data_generator.next()
            y_batch, _ = mask_generator.next()
            yield x_batch, y_batch

    def prepare(self):
        augmentation = self.augs(p=1)

        # get names of jpg files inside folder and create a list
        train_images = list(filter(lambda x: x.endswith('.jpg'), os.listdir(TRAIN_PATH)))

        # input data array
        x_data = np.empty((len(train_images[:10]) * 2, self.shape[0], self.shape[1], self.shape[2]), dtype='uint8')
        y_data = np.empty((len(train_images[:10]) * 2, self.shape[0], self.shape[1], self.shape[2]), dtype='uint8')

        for i, file_name in enumerate(train_images[:9]):
            image = cv2.imread(os.path.join(TRAIN_PATH, file_name), cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(os.path.join(ANNO_PATH, file_name.replace('.jpg', '.png')), cv2.IMREAD_UNCHANGED)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

            for j in range(2):
                data = {'image': image, 'mask': mask}
                augmented = augmentation(**data)
                image_a, mask_a = augmented['image'], augmented['mask']

                print('making ' + str(i+j) + ' image..')
                x_data[i+j] = image_a
                y_data[i+j] = mask_a

        return x_data, y_data

    def augs(self, p=0.5):
        # return Compose([
        #     OneOf([
        #         ShiftScaleRotate(p=0.5, rotate_limit=10, interpolation=cv2.INTER_CUBIC, scale_limit=0),
        #         HorizontalFlip(p=0.5)
        #     ]),
        #     RandomBrightnessContrast(p=0.5),
        #     RandomCrop(p=1, height=512, width=512)
        # ], p=p)
        return RandomCrop(p=p, height=self.shape[0], width=self.shape[1])


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


if __name__ == '__main__':

    loader = DataLoader(shape=(256, 256, 3))

    print('preparing images..')

    train, masks = loader.prepare()

    print('splitting train/validation..')

    x_train, x_val, y_train, y_val = train_test_split(train, masks, test_size=0.2, random_state=SEED)

    print('making model')

    input_layer = Input(shape=(256, 256, 3))
    fpn = FPN(
        backbone_name='resnet34',
        input_tensor=input_layer,
        encoder_weights='imagenet',
        classes=len(CLASS_COLOR),
        use_batchnorm=True,
        dropout=0.25,
        activation='softmax'
    )

    x = fpn.layers[-1].output
    x = Conv2D(3, (1, 1), activation='softmax')(x)
    model = Model(input=input_layer, output=x)

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

    model.compile(optimizer=Adam(1e-4), loss=dice_loss, metrics=[dice_coef, jaccard_coef])

    history = model.fit_generator(loader.generator(x_train, y_train, 2),
                                  steps_per_epoch=len(x_train),
                                  validation_data=loader.val_generator(x_val, y_val),
                                  validation_steps=len(x_val),
                                  epochs=10,
                                  verbose=1)

    model_json = model.to_json()
    json_file = open('../models/some.json', 'w')
    json_file.write(model_json)
    json_file.close()
    print('Model saved!')

    K.clear_session()
    print('Cache cleared')
