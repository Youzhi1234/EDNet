import numpy as np
import tensorflow as tf
import cv2
import glob
from classification_models.tfkeras import Classifiers

stride=64 #Stride of the sliding window
batch_size=32
nPosSoftSize=2

def data_preparation(stride=stride):
    img_train = sorted(glob.glob(r'image/train/img/*'))
    label_train = sorted(glob.glob(r'image/train/label/*'))
    img_validation = sorted(glob.glob(r'image/validation/img/*'))
    label_validation = sorted(glob.glob(r'image/validation/label/*'))

    def read_img(path):
        img = cv2.imread(path)[:, :, 0]
        return img

    def load(name_path):
        name = []
        for path in name_path:
            name.append(read_img(path))
        name = np.array(name)
        return name

    def split(stride, before):
        H = before.shape[1]
        W = before.shape[2]
        h = 128
        w = 128
        after = []
        for n in range(before.shape[0]):
            for k1 in range(int((H - h) / stride) + 1):
                for k2 in range(int((W - w) / stride) + 1):
                    after.append(before[n, k1 * stride:k1 * stride + h, k2 * stride:k2 * stride + w])
        after = np.array(after)
        return after

    img_train = load(img_train)
    label_train = load(label_train)
    img_validation = load(img_validation)
    label_validation = load(label_validation)
    # split image
    img_train = split(stride, img_train) / 255
    img_train = np.expand_dims(img_train, axis=3)
    label_train = split(stride, label_train) / 255
    label_train = np.expand_dims(label_train, axis=3)
    img_validation = split(stride, img_validation) / 255
    img_validation = np.expand_dims(img_validation, axis=3)

    label_validation_ = label_validation.copy() / 255
    label_validation_ = np.expand_dims(label_validation_, axis=3)
    label_validation = split(stride, label_validation) / 255
    label_validation = np.expand_dims(label_validation, axis=3)

    img_train = np.append(np.append(img_train, img_train, axis=3), img_train, axis=3)
    img_validation = np.append(np.append(img_validation, img_validation, axis=3), img_validation, axis=3)
    return img_train, label_train, img_validation, label_validation, label_validation_

#Build decoder network
def decoder():
    def down(filters, size, apply_bn=True, apply_ac=True):  #
        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', use_bias=False)
        )
        if apply_bn:
            model.add(tf.keras.layers.BatchNormalization())
        if apply_ac:
            model.add(tf.keras.layers.LeakyReLU())
        return model


    def up(filters, size, apply_drop=False):  #
        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same', use_bias=False)
        )
        model.add(tf.keras.layers.BatchNormalization())
        if apply_drop:
            model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.LeakyReLU())
        return model
    
    inputs = tf.keras.layers.Input(shape=(128, 128, 1))
    down_stack = [
        down(16, 5),  # 64*64*128
        down(32, 5),  # 32*32*256
        down(32, 5),  # 16*16*512
        down(32, 5, apply_ac=False),  # 8*8*512
    ]

    up_stack = [
        up(32, 5),  # 16*16*512
        up(32, 5),  # 32*32*256
        up(16, 5),  # 64*64*128
    ]

    x = inputs

    for d in down_stack:
        x = d(x)

    for u in up_stack:
        x = u(x)

    x = tf.keras.layers.Conv2DTranspose(1, 5, strides=2,
                                        padding='same',
                                        activation='sigmoid')(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

#Build Encoder network
def encoder():
    ResNet34, _ = Classifiers.get('resnet34')
    encoder = ResNet34(input_shape=(128, 128, 3), weights='imagenet')
    encoder = tf.keras.Model(inputs=encoder.input, outputs=encoder.get_layer('stage4_unit1_relu1').output)
    x = encoder.output
    x = tf.keras.layers.Conv2D(256, 1, strides=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(512, 3, strides=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(256, 1, strides=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(512, 3, strides=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(256, 1, strides=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(512, 3, strides=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(32, 1, strides=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.Model(inputs=encoder.input, outputs=x)
