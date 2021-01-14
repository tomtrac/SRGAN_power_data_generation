from SRGAN_train import SRGAN
import os
import numpy as np
from datasets import get_evaluation_data
from networks import resolution_model_params
data_dir = 'data/'
evaluation_data_file_name = 'test'
train_data_file_name = 'train'


def model_inference(data_type, input_resolution, output_resolution, con_epoch, evaluation_data_file_name):
    no_up, up_row_list, up_col_list, lr_height, lr_width = resolution_model_params(input_resolution)
    gan_model = SRGAN(data_type, model_dir, no_up, up_row_list, up_col_list, train_data_file_name, continue_train=True,
                input_resolution=input_resolution,
                output_resolution=output_resolution, lr_height=lr_height, lr_width=lr_width, con_epoch=con_epoch)
    X_test_lr = get_evaluation_data(data_type, input_resolution, evaluation_data_file_name)
    X_test_lr = np.reshape(X_test_lr, (X_test_lr.shape[0], gan_model.lr_height, gan_model.lr_width, gan_model.channels))
    gen_data = gan_model.generator.predict(X_test_lr)
    gen_data = gen_data.reshape(-1, 288)
    return gen_data


if __name__ == '__main__':
    data_type = 'pv'
    input_resolution = 1800
    output_resolution = 300
    batch_size = 128
    con_epoch = 50000
    gan_model_name = 'srgan_%s_%s/' % (data_type, input_resolution)
    clear_folder = False
    model_dir = data_dir + gan_model_name
    gen_data = model_inference(data_type, input_resolution, output_resolution, con_epoch, evaluation_data_file_name)
