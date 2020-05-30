import h5py
import keras.backend as K
import numpy as np
import os
import os.path
import tensorflow as tf
import threading
from PIL import Image
from keras import backend as K
from keras import losses
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Input, MaxPooling2D, Lambda
from keras.layers import concatenate, Conv2D, Conv2DTranspose, Dropout, ReLU, BatchNormalization, Activation
from keras.layers.merge import add, multiply
from keras.models import Model
from keras.optimizers import Adam
from numpy import random
from random import randint
from utils import data_augmentation, prepare_dataset


def get_unet(minimum_kernel=32, do=0, activation=ReLU, iteration=1):
    inputs = Input((None, None, 3))
    conv1 = Dropout(do)(activation()(Conv2D(minimum_kernel, (3, 3), padding='same')(inputs)))
    conv1 = Dropout(do)(activation()(Conv2D(minimum_kernel, (3, 3), padding='same')(conv1)))
    a = conv1
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Dropout(do)(activation()(Conv2D(minimum_kernel * 2, (3, 3), padding='same')(pool1)))
    conv2 = Dropout(do)(activation()(Conv2D(minimum_kernel * 2, (3, 3), padding='same')(conv2)))
    b = conv2
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Dropout(do)(activation()(Conv2D(minimum_kernel * 4, (3, 3), padding='same')(pool2)))
    conv3 = Dropout(do)(activation()(Conv2D(minimum_kernel * 4, (3, 3), padding='same')(conv3)))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Dropout(do)(activation()(Conv2D(minimum_kernel * 8, (3, 3), padding='same')(pool3)))
    conv4 = Dropout(do)(activation()(Conv2D(minimum_kernel * 8, (3, 3), padding='same')(conv4)))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Dropout(do)(activation()(Conv2D(minimum_kernel * 16, (3, 3), padding='same')(pool4)))
    conv5 = Dropout(do)(activation()(Conv2D(minimum_kernel * 16, (3, 3), padding='same')(conv5)))

    up6 = concatenate([Conv2DTranspose(minimum_kernel * 8, (2, 2), strides=(2, 2), padding='same')(conv5), conv4],
                      axis=3)
    conv6 = Dropout(do)(activation()(Conv2D(minimum_kernel * 8, (3, 3), padding='same')(up6)))
    conv6 = Dropout(do)(activation()(Conv2D(minimum_kernel * 8, (3, 3), padding='same')(conv6)))

    up7 = concatenate([Conv2DTranspose(minimum_kernel * 4, (2, 2), strides=(2, 2), padding='same')(conv6), conv3],
                      axis=3)
    conv7 = Dropout(do)(activation()(Conv2D(minimum_kernel * 4, (3, 3), padding='same')(up7)))
    conv7 = Dropout(do)(activation()(Conv2D(minimum_kernel * 4, (3, 3), padding='same')(conv7)))

    up8 = concatenate([Conv2DTranspose(minimum_kernel * 2, (2, 2), strides=(2, 2), padding='same')(conv7), conv2],
                      axis=3)
    conv8 = Dropout(do)(activation()(Conv2D(minimum_kernel * 2, (3, 3), padding='same')(up8)))
    conv8 = Dropout(do)(activation()(Conv2D(minimum_kernel * 2, (3, 3), padding='same')(conv8)))

    up9 = concatenate([Conv2DTranspose(minimum_kernel, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Dropout(do)(activation()(Conv2D(minimum_kernel, (3, 3), padding='same')(up9)))
    conv9 = Dropout(do)(activation()(Conv2D(minimum_kernel, (3, 3), padding='same')(conv9)))

    pt_conv1a = Conv2D(minimum_kernel, (3, 3), padding='same')
    pt_activation1a = activation()
    pt_dropout1a = Dropout(do)
    pt_conv1b = Conv2D(minimum_kernel, (3, 3), padding='same')
    pt_activation1b = activation()
    pt_dropout1b = Dropout(do)
    pt_pooling1 = MaxPooling2D(pool_size=(2, 2))

    pt_conv2a = Conv2D(minimum_kernel * 2, (3, 3), padding='same')
    pt_activation2a = activation()
    pt_dropout2a = Dropout(do)
    pt_conv2b = Conv2D(minimum_kernel * 2, (3, 3), padding='same')
    pt_activation2b = activation()
    pt_dropout2b = Dropout(do)
    pt_pooling2 = MaxPooling2D(pool_size=(2, 2))

    pt_conv3a = Conv2D(minimum_kernel * 4, (3, 3), padding='same')
    pt_activation3a = activation()
    pt_dropout3a = Dropout(do)
    pt_conv3b = Conv2D(minimum_kernel * 4, (3, 3), padding='same')
    pt_activation3b = activation()
    pt_dropout3b = Dropout(do)

    pt_tranconv8 = Conv2DTranspose(minimum_kernel * 2, (2, 2), strides=(2, 2), padding='same')
    pt_conv8a = Conv2D(minimum_kernel * 2, (3, 3), padding='same')
    pt_activation8a = activation()
    pt_dropout8a = Dropout(do)
    pt_conv8b = Conv2D(minimum_kernel * 2, (3, 3), padding='same')
    pt_activation8b = activation()
    pt_dropout8b = Dropout(do)

    pt_tranconv9 = Conv2DTranspose(minimum_kernel, (2, 2), strides=(2, 2), padding='same')
    pt_conv9a = Conv2D(minimum_kernel, (3, 3), padding='same')
    pt_activation9a = activation()
    pt_dropout9a = Dropout(do)
    pt_conv9b = Conv2D(minimum_kernel, (3, 3), padding='same')
    pt_activation9b = activation()
    pt_dropout9b = Dropout(do)

    conv9s = [conv9]
    outs = []
    a_layers = [a]
    for iteration_id in range(iteration):
        out = Conv2D(1, (1, 1), activation='sigmoid', name=f'out1{iteration_id + 1}')(conv9s[-1])
        outs.append(out)

        conv1 = pt_dropout1a(pt_activation1a(pt_conv1a(conv9s[-1])))
        conv1 = pt_dropout1b(pt_activation1b(pt_conv1b(conv1)))
        a_layers.append(conv1)
        conv1 = concatenate(a_layers, axis=3)
        conv1 = Conv2D(minimum_kernel, (1, 1), padding='same')(conv1)
        pool1 = pt_pooling1(conv1)

        conv2 = pt_dropout2a(pt_activation2a(pt_conv2a(pool1)))
        conv2 = pt_dropout2b(pt_activation2b(pt_conv2b(conv2)))
        pool2 = pt_pooling2(conv2)

        conv3 = pt_dropout3a(pt_activation3a(pt_conv3a(pool2)))
        conv3 = pt_dropout3b(pt_activation3b(pt_conv3b(conv3)))

        up8 = concatenate([pt_tranconv8(conv3), conv2], axis=3)
        conv8 = pt_dropout8a(pt_activation8a(pt_conv8a(up8)))
        conv8 = pt_dropout8b(pt_activation8b(pt_conv8b(conv8)))

        up9 = concatenate([pt_tranconv9(conv8), conv1], axis=3)
        conv9 = pt_dropout9a(pt_activation9a(pt_conv9a(up9)))
        conv9 = pt_dropout9b(pt_activation9b(pt_conv9b(conv9)))

        conv9s.append(conv9)

    seg_final_out = Conv2D(1, (1, 1), activation='sigmoid', name='seg_final_out')(conv9)
    outs.append(seg_final_out)

    # to cls
    def masked_input(args):
        x, inputs = args
        return x * inputs
    cls_in = Lambda(masked_input)([seg_final_out, inputs])
    # cls_in = concatenate([cls_in, crossing_final_out], axis=3)
    cls_in = Lambda(lambda x: K.stop_gradient(x))(cls_in)

    # to cls (artery)
    conv1 = Dropout(do)(activation()(Conv2D(minimum_kernel, (3, 3), padding='same')(cls_in)))
    conv1 = Dropout(do)(activation()(Conv2D(minimum_kernel, (3, 3), padding='same')(conv1)))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Dropout(do)(activation()(Conv2D(minimum_kernel * 2, (3, 3), padding='same')(pool1)))
    conv2 = Dropout(do)(activation()(Conv2D(minimum_kernel * 2, (3, 3), padding='same')(conv2)))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Dropout(do)(activation()(Conv2D(minimum_kernel * 4, (3, 3), padding='same')(pool2)))
    conv3 = Dropout(do)(activation()(Conv2D(minimum_kernel * 4, (3, 3), padding='same')(conv3)))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Dropout(do)(activation()(Conv2D(minimum_kernel * 8, (3, 3), padding='same')(pool3)))
    conv4 = Dropout(do)(activation()(Conv2D(minimum_kernel * 8, (3, 3), padding='same')(conv4)))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Dropout(do)(activation()(Conv2D(minimum_kernel * 16, (3, 3), padding='same')(pool4)))
    conv5 = Dropout(do)(activation()(Conv2D(minimum_kernel * 16, (3, 3), padding='same')(conv5)))

    up6 = concatenate([Conv2DTranspose(minimum_kernel * 8, (2, 2), strides=(2, 2), padding='same')(conv5), conv4],
                      axis=3)
    conv6 = Dropout(do)(activation()(Conv2D(minimum_kernel * 8, (3, 3), padding='same')(up6)))
    conv6 = Dropout(do)(activation()(Conv2D(minimum_kernel * 8, (3, 3), padding='same')(conv6)))

    up7 = concatenate([Conv2DTranspose(minimum_kernel * 4, (2, 2), strides=(2, 2), padding='same')(conv6), conv3],
                      axis=3)
    conv7 = Dropout(do)(activation()(Conv2D(minimum_kernel * 4, (3, 3), padding='same')(up7)))
    conv7 = Dropout(do)(activation()(Conv2D(minimum_kernel * 4, (3, 3), padding='same')(conv7)))

    up8 = concatenate([Conv2DTranspose(minimum_kernel * 2, (2, 2), strides=(2, 2), padding='same')(conv7), conv2],
                      axis=3)
    conv8 = Dropout(do)(activation()(Conv2D(minimum_kernel * 2, (3, 3), padding='same')(up8)))
    conv8 = Dropout(do)(activation()(Conv2D(minimum_kernel * 2, (3, 3), padding='same')(conv8)))

    up9 = concatenate([Conv2DTranspose(minimum_kernel, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Dropout(do)(activation()(Conv2D(minimum_kernel, (3, 3), padding='same')(up9)))
    conv9 = Dropout(do)(activation()(Conv2D(minimum_kernel, (3, 3), padding='same')(conv9)))


    pt_cls_art_conv1a = Conv2D(minimum_kernel, (3, 3), padding='same')
    pt_cls_art_activation1a = activation()
    pt_cls_art_dropout1a = Dropout(do)
    pt_cls_art_conv1b = Conv2D(minimum_kernel, (3, 3), padding='same')
    pt_cls_art_activation1b = activation()
    pt_cls_art_dropout1b = Dropout(do)
    pt_cls_art_pooling1 = MaxPooling2D(pool_size=(2, 2))

    pt_cls_art_conv2a = Conv2D(minimum_kernel * 2, (3, 3), padding='same')
    pt_cls_art_activation2a = activation()
    pt_cls_art_dropout2a = Dropout(do)
    pt_cls_art_conv2b = Conv2D(minimum_kernel * 2, (3, 3), padding='same')
    pt_cls_art_activation2b = activation()
    pt_cls_art_dropout2b = Dropout(do)
    pt_cls_art_pooling2 = MaxPooling2D(pool_size=(2, 2))

    pt_cls_art_conv3a = Conv2D(minimum_kernel * 4, (3, 3), padding='same')
    pt_cls_art_activation3a = activation()
    pt_cls_art_dropout3a = Dropout(do)
    pt_cls_art_conv3b = Conv2D(minimum_kernel * 4, (3, 3), padding='same')
    pt_cls_art_activation3b = activation()
    pt_cls_art_dropout3b = Dropout(do)

    pt_cls_art_tranconv8 = Conv2DTranspose(minimum_kernel * 2, (2, 2), strides=(2, 2), padding='same')
    pt_cls_art_conv8a = Conv2D(minimum_kernel * 2, (3, 3), padding='same')
    pt_cls_art_activation8a = activation()
    pt_cls_art_dropout8a = Dropout(do)
    pt_cls_art_conv8b = Conv2D(minimum_kernel * 2, (3, 3), padding='same')
    pt_cls_art_activation8b = activation()
    pt_cls_art_dropout8b = Dropout(do)

    pt_cls_art_tranconv9 = Conv2DTranspose(minimum_kernel, (2, 2), strides=(2, 2), padding='same')
    pt_cls_art_conv9a = Conv2D(minimum_kernel, (3, 3), padding='same')
    pt_cls_art_activation9a = activation()
    pt_cls_art_dropout9a = Dropout(do)
    pt_cls_art_conv9b = Conv2D(minimum_kernel, (3, 3), padding='same')
    pt_cls_art_activation9b = activation()
    pt_cls_art_dropout9b = Dropout(do)

    conv9s_cls_art = [conv9]
    a_layers = [a]
    for iteration_id in range(iteration):
        out = Conv2D(1, (1, 1), activation='sigmoid', name=f'out1_cls_art{iteration_id + 1}')(conv9s_cls_art[-1])
        outs.append(out)

        conv1 = pt_cls_art_dropout1a(pt_cls_art_activation1a(pt_cls_art_conv1a(conv9s_cls_art[-1])))
        conv1 = pt_cls_art_dropout1b(pt_cls_art_activation1b(pt_cls_art_conv1b(conv1)))
        a_layers.append(conv1)
        conv1 = concatenate(a_layers, axis=3)
        conv1 = Conv2D(minimum_kernel, (1, 1), padding='same')(conv1)
        pool1 = pt_cls_art_pooling1(conv1)

        conv2 = pt_cls_art_dropout2a(pt_cls_art_activation2a(pt_cls_art_conv2a(pool1)))
        conv2 = pt_cls_art_dropout2b(pt_cls_art_activation2b(pt_cls_art_conv2b(conv2)))
        pool2 = pt_cls_art_pooling2(conv2)

        conv3 = pt_cls_art_dropout3a(pt_cls_art_activation3a(pt_cls_art_conv3a(pool2)))
        conv3 = pt_cls_art_dropout3b(pt_cls_art_activation3b(pt_cls_art_conv3b(conv3)))

        up8 = concatenate([pt_cls_art_tranconv8(conv3), conv2], axis=3)
        conv8 = pt_cls_art_dropout8a(pt_cls_art_activation8a(pt_cls_art_conv8a(up8)))
        conv8 = pt_cls_art_dropout8b(pt_cls_art_activation8b(pt_cls_art_conv8b(conv8)))

        up9 = concatenate([pt_cls_art_tranconv9(conv8), conv1], axis=3)
        conv9 = pt_cls_art_dropout9a(pt_cls_art_activation9a(pt_cls_art_conv9a(up9)))
        conv9 = pt_cls_art_dropout9b(pt_cls_art_activation9b(pt_cls_art_conv9b(conv9)))

        conv9s_cls_art.append(conv9)


    cls_art_final_out = Conv2D(1, (1, 1), activation='sigmoid', name='cls_art_final_out')(conv9)

    outs.append(cls_art_final_out)


    # to cls (vein)
    conv1 = Dropout(do)(activation()(Conv2D(minimum_kernel, (3, 3), padding='same')(cls_in)))
    conv1 = Dropout(do)(activation()(Conv2D(minimum_kernel, (3, 3), padding='same')(conv1)))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Dropout(do)(activation()(Conv2D(minimum_kernel * 2, (3, 3), padding='same')(pool1)))
    conv2 = Dropout(do)(activation()(Conv2D(minimum_kernel * 2, (3, 3), padding='same')(conv2)))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Dropout(do)(activation()(Conv2D(minimum_kernel * 4, (3, 3), padding='same')(pool2)))
    conv3 = Dropout(do)(activation()(Conv2D(minimum_kernel * 4, (3, 3), padding='same')(conv3)))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Dropout(do)(activation()(Conv2D(minimum_kernel * 8, (3, 3), padding='same')(pool3)))
    conv4 = Dropout(do)(activation()(Conv2D(minimum_kernel * 8, (3, 3), padding='same')(conv4)))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Dropout(do)(activation()(Conv2D(minimum_kernel * 16, (3, 3), padding='same')(pool4)))
    conv5 = Dropout(do)(activation()(Conv2D(minimum_kernel * 16, (3, 3), padding='same')(conv5)))

    up6 = concatenate([Conv2DTranspose(minimum_kernel * 8, (2, 2), strides=(2, 2), padding='same')(conv5), conv4],
                      axis=3)
    conv6 = Dropout(do)(activation()(Conv2D(minimum_kernel * 8, (3, 3), padding='same')(up6)))
    conv6 = Dropout(do)(activation()(Conv2D(minimum_kernel * 8, (3, 3), padding='same')(conv6)))

    up7 = concatenate([Conv2DTranspose(minimum_kernel * 4, (2, 2), strides=(2, 2), padding='same')(conv6), conv3],
                      axis=3)
    conv7 = Dropout(do)(activation()(Conv2D(minimum_kernel * 4, (3, 3), padding='same')(up7)))
    conv7 = Dropout(do)(activation()(Conv2D(minimum_kernel * 4, (3, 3), padding='same')(conv7)))

    up8 = concatenate([Conv2DTranspose(minimum_kernel * 2, (2, 2), strides=(2, 2), padding='same')(conv7), conv2],
                      axis=3)
    conv8 = Dropout(do)(activation()(Conv2D(minimum_kernel * 2, (3, 3), padding='same')(up8)))
    conv8 = Dropout(do)(activation()(Conv2D(minimum_kernel * 2, (3, 3), padding='same')(conv8)))

    up9 = concatenate([Conv2DTranspose(minimum_kernel, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Dropout(do)(activation()(Conv2D(minimum_kernel, (3, 3), padding='same')(up9)))
    conv9 = Dropout(do)(activation()(Conv2D(minimum_kernel, (3, 3), padding='same')(conv9)))


    pt_cls_vei_conv1a = Conv2D(minimum_kernel, (3, 3), padding='same')
    pt_cls_vei_activation1a = activation()
    pt_cls_vei_dropout1a = Dropout(do)
    pt_cls_vei_conv1b = Conv2D(minimum_kernel, (3, 3), padding='same')
    pt_cls_vei_activation1b = activation()
    pt_cls_vei_dropout1b = Dropout(do)
    pt_cls_vei_pooling1 = MaxPooling2D(pool_size=(2, 2))

    pt_cls_vei_conv2a = Conv2D(minimum_kernel * 2, (3, 3), padding='same')
    pt_cls_vei_activation2a = activation()
    pt_cls_vei_dropout2a = Dropout(do)
    pt_cls_vei_conv2b = Conv2D(minimum_kernel * 2, (3, 3), padding='same')
    pt_cls_vei_activation2b = activation()
    pt_cls_vei_dropout2b = Dropout(do)
    pt_cls_vei_pooling2 = MaxPooling2D(pool_size=(2, 2))

    pt_cls_vei_conv3a = Conv2D(minimum_kernel * 4, (3, 3), padding='same')
    pt_cls_vei_activation3a = activation()
    pt_cls_vei_dropout3a = Dropout(do)
    pt_cls_vei_conv3b = Conv2D(minimum_kernel * 4, (3, 3), padding='same')
    pt_cls_vei_activation3b = activation()
    pt_cls_vei_dropout3b = Dropout(do)

    pt_cls_vei_tranconv8 = Conv2DTranspose(minimum_kernel * 2, (2, 2), strides=(2, 2), padding='same')
    pt_cls_vei_conv8a = Conv2D(minimum_kernel * 2, (3, 3), padding='same')
    pt_cls_vei_activation8a = activation()
    pt_cls_vei_dropout8a = Dropout(do)
    pt_cls_vei_conv8b = Conv2D(minimum_kernel * 2, (3, 3), padding='same')
    pt_cls_vei_activation8b = activation()
    pt_cls_vei_dropout8b = Dropout(do)

    pt_cls_vei_tranconv9 = Conv2DTranspose(minimum_kernel, (2, 2), strides=(2, 2), padding='same')
    pt_cls_vei_conv9a = Conv2D(minimum_kernel, (3, 3), padding='same')
    pt_cls_vei_activation9a = activation()
    pt_cls_vei_dropout9a = Dropout(do)
    pt_cls_vei_conv9b = Conv2D(minimum_kernel, (3, 3), padding='same')
    pt_cls_vei_activation9b = activation()
    pt_cls_vei_dropout9b = Dropout(do)

    conv9s_cls_vei = [conv9]
    a_layers = [a]
    for iteration_id in range(iteration):
        out = Conv2D(1, (1, 1), activation='sigmoid', name=f'out1_cls_vei{iteration_id + 1}')(conv9s_cls_vei[-1])
        outs.append(out)

        conv1 = pt_cls_vei_dropout1a(pt_cls_vei_activation1a(pt_cls_vei_conv1a(conv9s_cls_vei[-1])))
        conv1 = pt_cls_vei_dropout1b(pt_cls_vei_activation1b(pt_cls_vei_conv1b(conv1)))
        a_layers.append(conv1)
        conv1 = concatenate(a_layers, axis=3)
        conv1 = Conv2D(minimum_kernel, (1, 1), padding='same')(conv1)
        pool1 = pt_cls_vei_pooling1(conv1)

        conv2 = pt_cls_vei_dropout2a(pt_cls_vei_activation2a(pt_cls_vei_conv2a(pool1)))
        conv2 = pt_cls_vei_dropout2b(pt_cls_vei_activation2b(pt_cls_vei_conv2b(conv2)))
        pool2 = pt_cls_vei_pooling2(conv2)

        conv3 = pt_cls_vei_dropout3a(pt_cls_vei_activation3a(pt_cls_vei_conv3a(pool2)))
        conv3 = pt_cls_vei_dropout3b(pt_cls_vei_activation3b(pt_cls_vei_conv3b(conv3)))

        up8 = concatenate([pt_cls_vei_tranconv8(conv3), conv2], axis=3)
        conv8 = pt_cls_vei_dropout8a(pt_cls_vei_activation8a(pt_cls_vei_conv8a(up8)))
        conv8 = pt_cls_vei_dropout8b(pt_cls_vei_activation8b(pt_cls_vei_conv8b(conv8)))

        up9 = concatenate([pt_cls_vei_tranconv9(conv8), conv1], axis=3)
        conv9 = pt_cls_vei_dropout9a(pt_cls_vei_activation9a(pt_cls_vei_conv9a(up9)))
        conv9 = pt_cls_vei_dropout9b(pt_cls_vei_activation9b(pt_cls_vei_conv9b(conv9)))

        conv9s_cls_vei.append(conv9)


    cls_vei_final_out = Conv2D(1, (1, 1), activation='sigmoid', name='cls_vei_final_out')(conv9)

    outs.append(cls_vei_final_out)


    model = Model(inputs=[inputs], outputs=outs)


    loss_funcs = {}
    for iteration_id in range(iteration):
        loss_funcs.update({f'out1{iteration_id + 1}': losses.binary_crossentropy})
    loss_funcs.update({'seg_final_out': losses.binary_crossentropy})
    loss_funcs.update({'cls_art_final_out': losses.binary_crossentropy})
    loss_funcs.update({'cls_vei_final_out': losses.binary_crossentropy})
    for iteration_id in range(iteration):
        loss_funcs.update({f'out1_cls_art{iteration_id + 1}': losses.binary_crossentropy})
    for iteration_id in range(iteration):
        loss_funcs.update({f'out1_cls_vei{iteration_id + 1}': losses.binary_crossentropy})
  
    metrics = {
        "seg_final_out": ['accuracy'],
        "cls_art_final_out": ['accuracy'],
        "cls_vei_final_out": ['accuracy'],
    }

    model.compile(optimizer=Adam(lr=1e-3), loss=loss_funcs, metrics=metrics)

    return model


def random_crop(img, mask, mask_onehot, crop_size):
    imgheight = img.shape[0]
    imgwidth = img.shape[1]

    i = randint(0, imgheight - crop_size)
    j = randint(0, imgwidth - crop_size)

    return img[i:(i + crop_size), j:(j + crop_size), :]\
            , np.array(mask)[:, i:(i + crop_size), j:(j + crop_size)]\
            , np.array(mask_onehot)[:, i:(i + crop_size), j:(j + crop_size)]


class Generator():
    def __init__(self, batch_size, repeat, dataset):
        self.lock = threading.Lock()
        self.dataset = dataset
        with self.lock:
            self.list_images_all = prepare_dataset.getTrainingData(0, self.dataset)
            self.list_gt_all = prepare_dataset.getTrainingData(1, self.dataset)
            self.list_gt_all_onehot = prepare_dataset.getTrainingData(1, self.dataset, need_one_hot=True)
        self.n = len(self.list_images_all)
        self.index = 0
        self.repeat = repeat
        self.batch_size = batch_size
        self.step = self.batch_size // self.repeat

        if self.repeat >= self.batch_size:
            self.repeat = self.batch_size
            self.step = 1

    def gen(self, au=True, crop_size=48, iteration=None):

        while True:
            data_yield = [self.index % self.n,
                          (self.index + self.step) % self.n if (self.index + self.step) < self.n else self.n]
            self.index = (self.index + self.step) % self.n

            list_images_base = self.list_images_all[data_yield[0]:data_yield[1]]
            list_gt_base = self.list_gt_all[data_yield[0]:data_yield[1]]
            list_gt_onehot_base = self.list_gt_all_onehot[data_yield[0]:data_yield[1]]

            list_images_aug = []
            list_gt_aug = []
            list_gt_onehot_aug = []
            image_id = -1
            for image, gt in zip(list_images_base, list_gt_base):
                image_id += 1
                gt2 = list_gt_onehot_base[image_id]
                if au:
                    if crop_size == prepare_dataset.DESIRED_DATA_SHAPE[0]:
                        for _ in range(self.repeat):
                            image, gt, gt2 = data_augmentation.random_augmentation(image, gt, gt2)
                            list_images_aug.append(image)
                            list_gt_aug.append(gt)
                            list_gt_onehot_aug.append(gt2)
                    else:
                        image, gt, gt2 = data_augmentation.random_augmentation(image, gt, gt2)
                        list_images_aug.append(image)
                        list_gt_aug.append(gt)
                        list_gt_onehot_aug.append(gt2)
                else:
                    list_images_aug.append(image)
                    list_gt_aug.append(gt)
                    list_gt_onehot_aug.append(gt2)

            list_images = []
            list_gt = []
            list_gt_onehot = []
            image_id = -1
            if crop_size == prepare_dataset.DESIRED_DATA_SHAPE[0]:
                list_images = list_images_aug
                list_gt = list_gt_aug
                list_gt_onehot = list_gt_onehot_aug
            else:
                for image, gt in zip(list_images_aug, list_gt_aug):
                    image_id += 1
                    for _ in range(self.repeat):
                        image_, gt_, gt_onehot_ = random_crop(image, gt, list_gt_onehot_aug[image_id], crop_size)

                        list_images.append(image_)
                        list_gt.append(gt_)
                        list_gt_onehot.append(gt_onehot_)

            outs = {}
            for iteration_id in range(iteration):
                outs.update({f'out1{iteration_id + 1}': np.array(list_gt)[:,0]})
            outs.update({'seg_final_out': np.array(list_gt)[:,0]})
            # outs.update({'crossing_final_out': np.array(list_gt)[:,1]})
            outs.update({'cls_art_final_out': np.array(list_gt)[:,2]})
            outs.update({'cls_vei_final_out': np.array(list_gt)[:,3]})
            for iteration_id in range(iteration):
                outs.update({f'out1_cls_art{iteration_id + 1}': np.array(list_gt)[:,2]})
            for iteration_id in range(iteration):
                outs.update({f'out1_cls_vei{iteration_id + 1}': np.array(list_gt)[:,3]})
            yield np.array(list_images), outs