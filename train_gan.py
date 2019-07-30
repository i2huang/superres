import random
import glob
import subprocess
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import PReLU, LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.applications import VGG19
import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbCallback

run = wandb.init(project='superres')
config = run.config

config.num_epochs = 10000
config.batch_size = 1
config.input_height = 32
config.input_width = 32
config.output_height = 256
config.output_width = 256

val_dir = 'data/test'
train_dir = 'data/train'

# automatically get the data if it doesn't exist
if not os.path.exists("data"):
    print("Downloading flower dataset...")
    subprocess.check_output(
        "mkdir data && curl https://storage.googleapis.com/wandb/flower-enhance.tar.gz | tar xzf - -C data", shell=True)

config.steps_per_epoch = len(
    glob.glob(train_dir + "/*-in.jpg")) // config.batch_size
config.val_steps_per_epoch = len(
    glob.glob(val_dir + "/*-in.jpg")) // config.batch_size


def image_generator(batch_size, img_dir):
    """A generator that returns small images and large images.  DO NOT ALTER the validation set"""
    input_filenames = glob.glob(img_dir + "/*-in.jpg")
    counter = 0
    random.shuffle(input_filenames)
    while True:
        small_images = np.zeros(
            (batch_size, config.input_width, config.input_height, 3))
        large_images = np.zeros(
            (batch_size, config.output_width, config.output_height, 3))
        if counter+batch_size >= len(input_filenames):
            counter = 0
        for i in range(batch_size):
            img = input_filenames[counter + i]
            small_images[i] = np.array(Image.open(img)) / 255.0
            large_images[i] = np.array(
                Image.open(img.replace("-in.jpg", "-out.jpg"))) / 255.0
        yield (small_images, large_images)
        counter += batch_size


def perceptual_distance(y_true, y_pred):
    """Calculate perceptual distance, DO NOT ALTER"""
    y_true *= 255
    y_pred *= 255
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]

    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))

val_generator = image_generator(config.batch_size, val_dir)
in_sample_images, out_sample_images = next(val_generator)

class ImageLogger(Callback):
    def on_epoch_end(self, epoch, logs):
        preds = self.generator.predict(in_sample_images)
        in_resized = []
        for arr in in_sample_images:
            # Simple upsampling
            in_resized.append(arr.repeat(8, axis=0).repeat(8, axis=1))
        wandb.log({
            "examples": [wandb.Image(np.concatenate([in_resized[i] * 255, o * 255, out_sample_images[i] * 255], axis=1)) for i, o in enumerate(preds)]
        }, commit=False)


class GAN():
    def __init__(self):
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.img_lr_shape = (32, 32, 3)

        self.sample_generator = image_generator(2, train_dir)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer, 
                               metrics=[perceptual_distance])

        self.vgg = self.build_vgg()
        self.vgg.trainable = False
        self.vgg.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        # The generator takes noise as input and generated imgs
        img_hr = Input(shape=self.img_shape)
        img_lr = Input(shape=self.img_lr_shape)

        fake_hr = self.generator(img_lr)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(fake_hr)

        # Extract image features of the generated img
        fake_features = self.vgg(fake_hr)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model([img_lr, img_hr], [valid, fake_features])
        self.combined.compile(loss=['binary_crossentropy', 'mse'], optimizer=optimizer, 
                              loss_weights=[1e-3, 1])

    def build_vgg(self):
        """
        Builds a pre-trained VGG19 model that outputs image features extracted at the
        third block of the model
        """
        vgg = VGG19(weights="imagenet", input_shape=self.img_shape, include_top=False)
        # Set outputs to outputs of last conv. layer in block 3
        # See architecture at: https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
        vgg.outputs = [vgg.layers[9].output]

        img = Input(shape=self.img_shape)

        #res = img[:,16:240,16:240,:]
        res = img * 2 - 1

        # Extract image features
        img_features = vgg(res)

        return Model(img, img_features)

    def build_generator(self):

        def residual_block(layer_input, filters):
            #"""Residual block described in paper"""
            d = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
            d = layers.Activation('relu')(d)
            d = layers.BatchNormalization(momentum=0.8)(d)
            d = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
            d = layers.BatchNormalization(momentum=0.8)(d)
            d = layers.Add()([d, layer_input])
            return d

        def deconv2d(layer_input):
            #"""Layers used during upsampling"""
            u = layers.UpSampling2D(size=2)(layer_input)
            u = layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
            u = layers.Activation('relu')(u)
            return u

        img_lr_shape = (config.input_width, config.input_height, 3)

        model = Sequential()
        img_lr = Input(shape=img_lr_shape) 

        c1 = layers.Conv2D(64, kernel_size=9, strides=1, padding='same')(img_lr * 2 - 1)
        c1 = layers.Activation('relu')(c1)

        # Local variables
        self.gf = 64
        self.n_residual_blocks = 16

        # Propogate through residual blocks
        r = residual_block(c1, self.gf)
        for _ in range(self.n_residual_blocks - 1):
            r = residual_block(r, self.gf)

        # Post-residual block
        c2 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
        c2 = layers.BatchNormalization(momentum=0.8)(c2)
        c2 = layers.Add()([c2, c1])

        # Upsampling
        u1 = deconv2d(c2)
        u2 = deconv2d(u1)
        u3 = deconv2d(u2)

        # Generate high resolution output
        gen_hr = layers.Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(u3)
        gen_hr = gen_hr * 0.5 + 0.5

        return Model(img_lr, gen_hr)

    def build_discriminator(self):
        def d_block(layer_input, filters, strides=1, bn=True):
            """Discriminator layer"""
            d = layers.Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d = layers.LeakyReLU(alpha=0.2)(d)
            if bn:
                d = layers.BatchNormalization(momentum=0.8)(d)
            return d

        # local variables
        self.df = 64

        # Input img
        img = Input(shape=self.img_shape)
        res = img * 2 - 1

        #print("Ishape:" + str(res.shape))

        d1 = d_block(res, self.df, bn=False)
        d2 = d_block(d1, self.df, strides=2)
        d3 = d_block(d2, self.df*2)
        d4 = d_block(d3, self.df*2, strides=2)
        d5 = d_block(d4, self.df*4)
        d6 = d_block(d5, self.df*4, strides=2)
        d7 = d_block(d6, self.df*8)
        d8 = d_block(d7, self.df*8, strides=2)

        d9 = Dense(self.df*16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)
        d11 = Flatten()(d10)
        validity = Dense(1, activation='sigmoid')(d11)

        #print("Oshape:" + str(validity.shape))

        return Model(img, validity)

    def train(self, epochs, batch_size=1, save_interval=50):

        # Load the dataset
        train_generator = image_generator(batch_size, train_dir)
        #half_batch = int(batch_size / 2)

        # The generator wants the discriminator to label the generated samples
        # as valid (ones)
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size,1))

        for epoch in range(epochs):

            img_lr, img_hr = next(train_generator)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(img_lr)

            # Train the discriminator
            self.discriminator.trainable=True
            d_loss_real = self.discriminator.train_on_batch(img_hr, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            # Pick another sample image
            img_lr, img_hr = next(train_generator)

            # Extract ground truth image features using pre-trained VGG19 model
            image_features = self.vgg.predict(img_hr)

            self.discriminator.trainable=False

            # Train the generator
            g_loss = self.combined.train_on_batch([img_lr, img_hr], [valid, image_features])

            # Plot the progress
            #print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            print ("%d [D loss: %f][G loss: %f]" % (epoch, d_loss[0], g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                img_lr, img_hr = next(val_generator)
                self.save_imgs(epoch, img_lr, img_hr)
                val_loss = self.generator.evaluate(
                                                   x=img_lr, 
                                                   y=img_hr, 
                                                   batch_size=batch_size, 
                                                   verbose=1, 
                                                   sample_weight=None, 
                                                   steps=1, 
                                                   callbacks=[ImageLogger(), WandbCallback()])
                #print (self.generator.metrics_names)
                #print (str(val_loss))
                wandb.log({
                           'epoch': epoch,
                           'generator_loss': g_loss[0],
                           'discriminator_loss': d_loss[0],
                           'val_loss': val_loss[0],
                           'val_perceptual_distance' : val_loss[1],
                           })

    def save_imgs(self, epoch, img_lr, img_hr):
        titles = ['IN', 'GEN', 'ORG']
        os.makedirs('gan', exist_ok=True)
        r, c = 2, 3

        gen_imgs = self.generator.predict(img_lr)
        #print("Max %f Min %f" % (gen_imgs.max(), gen_imgs.min()))

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j, image in enumerate([img_lr, gen_imgs, img_hr]):
                axs[i,j].imshow(image[0])
                axs[i,j].set_title(titles[j])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("gan/sample_%d.png" % epoch)
        plt.close()

#model = Sequential()
#model.add(layers.Conv2D(3, (3, 3), activation='relu', padding='same',
#                       input_shape=(config.input_width, config.input_height, 3)))
#model.add(layers.UpSampling2D())
#model.add(layers.Conv2D(3, (3, 3), activation='relu', padding='same'))
#model.add(layers.UpSampling2D())
#model.add(layers.Conv2D(3, (3, 3), activation='relu', padding='same'))
#model.add(layers.UpSampling2D())
#model.add(layers.Conv2D(3, (3, 3), activation='relu', padding='same'))

# DONT ALTER metrics=[perceptual_distance]
#model.compile(optimizer='adam', loss='mse',
#              metrics=[perceptual_distance])

#model.fit_generator(image_generator(config.batch_size, train_dir),
#                    steps_per_epoch=config.steps_per_epoch,
#                    epochs=config.num_epochs, callbacks=[ImageLogger(), WandbCallback()],
#                    validation_steps=config.val_steps_per_epoch,
#                    validation_data=val_generator)
#if __name__ == '__main__':

gan = GAN()
gan.train(epochs=10000, batch_size=config.batch_size, save_interval=25)
