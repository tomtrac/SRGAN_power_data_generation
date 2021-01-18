from __future__ import print_function, division
from keras.layers import Input
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from networks import Discriminator, Generator, resolution_model_params
import numpy as np
import os
from datasets import get_training_data
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

KTF.set_session(sess)
data_dir = 'data/'
train_data_file_name = 'train'


def get_gan_network(discriminator, shape, generator, optimizer):
    discriminator.trainable = False
    gan_input = Input(shape=shape)
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=[x, gan_output])
    gan.compile(loss=["mse", "binary_crossentropy"],
                loss_weights=[1., 1e-3],
                optimizer=optimizer)

    return gan


class SRGAN():
    def __init__(self, data_type, model_dir, no_up, up_row_list, up_col_list, train_data_file_name,
                 continue_train=False, con_epoch=None, input_resolution=1800, output_resolution=300,
                 lr_height=8, lr_width=6):
        self.train_data_file_name = train_data_file_name
        # Input shape
        self.data_type = data_type
        self.channels = 1
        self.lr_height = lr_height  # Low resolution height
        self.lr_width = lr_width  # Low resolution width
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        self.hr_height = 24  # High resolution height
        self.hr_width = 12  # High resolution width
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.no_up = no_up
        self.up_row_list = up_row_list
        self.up_col_list = up_col_list
        self.con_epoch = con_epoch
        # Number of residual blocks in the generator
        self.n_residual_blocks = 5
        optimizer = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        if continue_train is False:
            self.gf = 64
            self.df = 64

            self.discriminator = Discriminator(self.hr_shape)

            self.generator = Generator(self.lr_shape, self.gf, self.n_residual_blocks, self.channels, self.no_up,
                                       self.up_row_list, self.up_col_list)

            self.generator.compile(loss='mse', optimizer=optimizer)
            self.discriminator.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])

            self.combined = get_gan_network(self.discriminator, self.lr_shape, self.generator, optimizer)
        else:
            if con_epoch is None:
                discriminator_model = model_dir + "gan_discriminator.json"
                discriminator_weights = model_dir + "gan_discriminator_weights.hdf5"
                generator_model = model_dir + "gan_generator.json"
                generator_weights = model_dir + "gan_generator_weights.hdf5"
            else:
                discriminator_model = model_dir + "gan_discriminator_%s.json" % con_epoch
                discriminator_weights = model_dir + "gan_discriminator_%s_weights.hdf5" % con_epoch
                generator_model = model_dir + "gan_generator_%s.json" % con_epoch
                generator_weights = model_dir + "gan_generator_%s_weights.hdf5" % con_epoch
            discriminator_file = open(discriminator_model, 'r')
            loaded_discriminator_file = discriminator_file.read()
            discriminator_file.close()
            self.discriminator = model_from_json(loaded_discriminator_file)
            self.discriminator.load_weights(discriminator_weights)

            generator_file = open(generator_model, 'r')
            loaded_generator_file = generator_file.read()
            generator_file.close()
            self.generator = model_from_json(loaded_generator_file)
            self.generator.load_weights(generator_weights)
            # The generator takes noise as input and generates imgs
            self.generator.compile(loss='mse', optimizer=optimizer)
            self.discriminator.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])

            self.combined = get_gan_network(self.discriminator, self.lr_shape, self.generator, optimizer)


    def train(self, epochs, batch_size=64, print_interval=50, data_type='pv'):

        X_train_hr = get_training_data(data_type, self.output_resolution, self.train_data_file_name)
        X_train_lr = get_training_data(data_type, self.input_resolution, self.train_data_file_name)
        if self.con_epoch is None:
            epoch_range = range(epochs)
        else:
            epoch_range = range(self.con_epoch + 1, self.con_epoch + epochs)
        for epoch in epoch_range:

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train_hr.shape[0], batch_size)
            batch_hr = X_train_hr[idx]
            batch_hr = np.reshape(batch_hr, (batch_size, self.hr_height, self.hr_width, self.channels))
            batch_lr = X_train_lr[idx]
            batch_lr = np.reshape(batch_lr, (batch_size, self.lr_height, self.lr_width, self.channels))


            # Generate a batch of new images
            gen_data = self.generator.predict(batch_lr)
            real_data_Y = np.ones((batch_size, 1))
            fake_data_Y = np.zeros((batch_size, 1))

            self.discriminator.trainable = True

            d_loss_real = self.discriminator.train_on_batch(batch_hr, real_data_Y)
            d_loss_fake = self.discriminator.train_on_batch(gen_data, fake_data_Y)
            d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

            idx = np.random.randint(0, X_train_hr.shape[0], batch_size)
            batch_hr = X_train_hr[idx]
            batch_hr = np.reshape(batch_hr, (batch_size, self.hr_height, self.hr_width, self.channels))
            batch_lr = X_train_lr[idx]
            batch_lr = np.reshape(batch_lr, (batch_size, self.lr_height, self.lr_width, self.channels))

            gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
            self.discriminator.trainable = False
            g_loss = self.combined.train_on_batch(batch_lr, [batch_hr, gan_Y])


            if epoch % print_interval == 0:
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %s][D loss real: %s] [D loss fake: %s]" %
                      (epoch, d_loss[0], 100 * d_loss[1], g_loss, d_loss_real, d_loss_fake))
                self.save_model(epoch)


    def save_model(self, epoch):

        def save(model, model_name):
            model_path = model_dir + "%s.json" % model_name
            weights_path = model_dir + "%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                       "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "gan_generator")
        save(self.discriminator, "gan_discriminator")
        save(self.generator, "gan_generator_%s" % epoch)
        save(self.discriminator, "gan_discriminator_%s" % epoch)




if __name__ == '__main__':
    data_type = 'pv'
    input_resolution = 1800
    output_resolution = 300
    batch_size = 128
    gan_model_name = 'srgan_%s_%s/' % (data_type, input_resolution)
    clear_folder = False
    model_dir = data_dir + gan_model_name
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if clear_folder is True:
        filelist = [f for f in os.listdir(model_dir) if f.endswith(".hdf5") or f.endswith(".json")]
        for f in filelist:
            os.remove(os.path.join(model_dir, f))
    no_up, up_row_list, up_col_list, lr_height, lr_width = resolution_model_params(input_resolution)
    gan = SRGAN(data_type, model_dir, no_up, up_row_list, up_col_list, train_data_file_name, continue_train=False, input_resolution=input_resolution,
                output_resolution=output_resolution, lr_height=lr_height, lr_width=lr_width)
    gan.train(epochs=20000, batch_size=batch_size, print_interval=5000, data_type=data_type)
