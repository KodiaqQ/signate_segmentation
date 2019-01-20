import keras.backend as K
from segmentation_models import FPN
import os
from albumentations import (HorizontalFlip, ShiftScaleRotate, OneOf, Compose, RandomBrightnessContrast, RandomCrop)
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input
import sys
import argparse
from losses import dice_coef, dice_loss, jaccard_coef

parser = argparse.ArgumentParser()
parser.add_argument('--path', help='Path to dataset', default='data')
parser.add_argument('--backbone', help='Backbone name')
args = parser.parse_args()

path = os.path.abspath(__file__)
pathes = path.split('\\')
sys.path.append(os.path.abspath('\\'.join(pathes[:-2])))
from DLUtils.seg_data_generator import SegDataGenerator

TRAIN_PATH = os.path.join(args.path, 'seg_train_images')
ANNO_PATH = os.path.join(args.path, 'seg_train_annotations')
CLASS_COLOR = {
    'Car': [0, 0, 255],
    'Pedestrian': [255, 0, 0],
    'Sky': [0, 152, 225],
    'Natural': [85, 255, 50],
    'Lane': [69, 47, 142],
}
SEED = 42
IMG_HEIGHT, IMG_WIDTH = 256, 256


def strong_aug(p=0.5):
    return Compose([
        OneOf([
            ShiftScaleRotate(p=0.5, rotate_limit=10, scale_limit=0),
            HorizontalFlip(p=0.5)
        ]),
        RandomBrightnessContrast(p=0.5),
        RandomCrop(p=1, height=IMG_HEIGHT, width=IMG_WIDTH)
    ], p=p)


def make_aug(image, mask, p):
    augmentation = strong_aug(p=p)
    data = {'image': image, 'mask': mask}
    augmented = augmentation(**data)
    return augmented['image'], augmented['mask']


if __name__ == '__main__':
    generator = SegDataGenerator(TRAIN_PATH, ANNO_PATH, batch_size=2, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                                 mask_shape=(IMG_HEIGHT, IMG_WIDTH, 3), preprocessing_function=make_aug,
                                 classes=len(CLASS_COLOR),
                                 classes_colors=CLASS_COLOR, prob_aug=1)
    input_layer = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    model = FPN(
        backbone_name=args.name,
        input_tensor=input_layer,
        encoder_weights='imagenet',
        classes=len(CLASS_COLOR),
        use_batchnorm=True,
        dropout=0.25,
        activation='softmax'
    )

    save_name = 'weights/' + args.name + '.h5'
    callbacks_list = [
        ModelCheckpoint(
            save_name,
            monitor='loss',
            verbose=1,
            save_best_only=True,
            mode='min',
            save_weights_only=True),
        ReduceLROnPlateau(
            monitor='loss',
            factor=0.2,
            patience=2,
            min_lr=1e-5)
    ]

    model.compile(optimizer=Adam(1e-4), loss=dice_loss, metrics=[dice_coef, jaccard_coef])

    history = model.fit_generator(generator,
                                  steps_per_epoch=500,
                                  epochs=2,
                                  verbose=1,
                                  callbacks=callbacks_list)

    model_json = model.to_json()
    json_file = open('models/' + args.name + '.json', 'w')
    json_file.write(model_json)
    json_file.close()
    print('Model saved!')

    K.clear_session()
    print('Cache cleared')
