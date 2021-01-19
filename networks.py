'''
Model architectures of the residual block, upsampling block, generator and discriminator
The original codes are from https://github.com/deepak112/Keras-SRGAN/blob/master/Network.py

'''

from __future__ import print_function, division
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import PReLU, LeakyReLU, ReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def res_block_gen(model, kernal_size, filters, strides):
    gen = model
    model = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(model)
    model = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)
    model = Add()([gen, model])

    return model


def up_sampling_block(model, kernal_size, filters, strides, up_row=3, up_col=2):
    model = UpSampling2D(size=(up_row, up_col))(model)
    model = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)
    model = ReLU()(model)

    return model


def discriminator_block(model, filters, kernel_size, strides):
    model = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.25)(model)

    return model


def Generator(noise_shape, gf, n_residual_blocks, channels, no_up, up_row_list, up_col_list):
    gen_input = Input(shape=noise_shape)

    model = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(gen_input)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(
        model)

    gen_model = model

    for index in range(n_residual_blocks):
        model = res_block_gen(model, 3, gf, 1)

    model = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)
    model = Add()([gen_model, model])

    for index in range(no_up):
        model = up_sampling_block(model, 3, 128, 1, up_row=up_row_list[index], up_col=up_col_list[index])

    model = Conv2D(filters=channels, kernel_size=3, strides=1, padding="same")(model)
    model = Activation('sigmoid')(model)

    generator_model = Model(inputs=gen_input, outputs=model)
    generator_model.summary()
    return generator_model


def Discriminator(input_shape):
    dis_input = Input(shape=input_shape)
    model = ZeroPadding2D(padding=((0, 0), (2, 2)))(dis_input)
    model = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.25)(model)

    model = discriminator_block(model, 128, 3, 2)
    model = discriminator_block(model, 256, 3, 1)
    model = Flatten()(model)

    model = Dense(1)(model)
    model = Activation('sigmoid')(model)

    discriminator_model = Model(inputs=dis_input, outputs=model)
    discriminator_model.summary()
    return discriminator_model


def resolution_model_params(input_resolution):
    # input_resolution of 1800 represents 30-minute resolution, 3600 represents hourly resolution
    if input_resolution == 1800:
        no_up, up_row_list, up_col_list, lr_height, lr_width = 1, [3], [2], 8, 6
    elif input_resolution == 3600:
        no_up, up_row_list, up_col_list, lr_height, lr_width = 2, [2, 2], [3, 1], 6, 4
    return no_up, up_row_list, up_col_list, lr_height, lr_width
