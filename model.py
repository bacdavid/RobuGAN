from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Progbar
from keras.layers import BatchNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Activation, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.models import Model
from keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt


# Sampling Layer -------------------------------------------------------------------------------------------------------


def sampling(args):
    latent_mean, latent_logvar = args
    batch = K.shape(latent_mean)[0]
    dim = K.int_shape(latent_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return latent_mean + K.exp(0.5 * latent_logvar) * epsilon


# Auto Encoder ---------------------------------------------------------------------------------------------------------


class AutoEncoder:
    """ Auto Encoder class.
    """

    def __init__(self, input_shape, latent_dim, learning_rate=0.0005):
        self.input_shape = input_shape  # (w,h,c)
        self.latent_dim = latent_dim  # n

        # Auto Encoder
        self.encoder = self._build_encoder()
        self.encoder_input = Input(shape=self.input_shape)
        self.encoder_mean_output, self.encoder_logvar_output = self.encoder(self.encoder_input)
        self.decoder = self._build_decoder()
        self.decoder_input = Input(shape=(self.latent_dim,))
        self.decoder_output = self.decoder(self.encoder_mean_output)

        # Generator
        self.latent_output = Lambda(sampling)([self.encoder_mean_output, self.encoder_logvar_output])
        self.gen_output = self.decoder(self.latent_output)

        # Critic
        self.critic = self._build_critic()

        # Disable the generator
        self.critic.trainable = True
        self.encoder.trainable = False
        self.decoder.trainable = False

        # Critic trainer
        self.h1_real, self.h2_real, self.h3_real, self.critic_output_real = self.critic(self.encoder_input)
        self.h1_fake, self.h2_fake, self.h3_fake, self.critic_output_fake = self.critic(self.gen_output)
        self.critic_trainer = Model(self.encoder_input, [self.critic_output_real, self.critic_output_fake])
        critic_loss = self._critic_loss()
        self.critic_trainer.add_loss(K.mean(critic_loss))
        self.critic_trainer.compile(optimizer=RMSprop(lr=learning_rate))
        self.critic_trainer.summary()

        # Disable the critic and re-enable the generator
        self.critic.trainable = False
        self.encoder.trainable = True
        self.decoder.trainable = True

        # Generator trainer
        self.gen_trainer = Model(self.encoder_input, [self.critic_output_real, self.critic_output_fake])
        gen_loss = self._gen_loss()
        self.gen_trainer.add_loss(K.mean(gen_loss))
        self.gen_trainer.compile(optimizer=RMSprop(lr=learning_rate))
        self.gen_trainer.summary()

        # Reconstruction prediction
        self.rec_sample = K.function([self.encoder_input], [self.decoder_output])

        # Generate prediction
        self.gen_sample = K.function([self.decoder_input], [self.decoder(self.decoder_input)])

        # Compute discriminator score (by means of the distance)
        self.compute_score = K.function([self.encoder_input], [gen_loss])

    def _build_encoder(self):
        # Input
        encoder_input = Input(shape=self.input_shape)

        # Encoder
        h = Conv2D(64, 5, strides=2, padding='same')(encoder_input)
        h = Activation('relu')(h)
        h = Conv2D(128, 5, strides=2, padding='same')(h)
        h = BatchNormalization(momentum=0.8)(h)
        h = Activation('relu')(h)
        h = Conv2D(256, 5, strides=2, padding='same')(h)
        h = BatchNormalization(momentum=0.8)(h)
        h = Activation('relu')(h)
        h = Flatten()(h)
        encoder_mean_output = Dense(self.latent_dim)(h)
        encoder_logvar_output = Dense(self.latent_dim)(h)

        # Model
        return Model(encoder_input, [encoder_mean_output, encoder_logvar_output])

    def _build_decoder(self):
        # Input
        decoder_input = Input(shape=(self.latent_dim,))

        # Decoder
        h = Dense(self.input_shape[0] * self.input_shape[1] // 2 ** 6 * 256, activation='relu')(decoder_input)
        h = Reshape((self.input_shape[0] // 2 ** 3, self.input_shape[1] // 2 ** 3, 256))(h)
        h = Conv2DTranspose(256, 5, strides=2, padding='same')(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = Conv2DTranspose(128, 5, strides=2, padding='same')(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = Conv2DTranspose(64, 5, strides=2, padding='same')(h)
        h = Activation('relu')(h)
        decoder_output = Conv2D(self.input_shape[2], 5, padding='same')(h)  # linear activation

        # Model
        return Model(decoder_input, decoder_output)

    def _build_critic(self):
        # Input
        critic_input = Input(shape=self.input_shape)

        # Critic
        h = Conv2D(64, 5, strides=2, padding='same')(critic_input)
        h1 = LeakyReLU(alpha=0.2)(h)
        h = Conv2D(128, 5, strides=2, padding='same')(h1)
        h = BatchNormalization()(h)
        h2 = LeakyReLU(alpha=0.2)(h)
        h = Conv2D(256, 5, strides=2, padding='same')(h2)
        h = BatchNormalization()(h)
        h3 = LeakyReLU(alpha=0.2)(h)
        h = Flatten()(h3)
        critic_output = Dense(1)(h)

        # Model
        return Model(critic_input, [h1, h2, h3, critic_output])

    def _critic_loss(self):
        true_loss = K.mean(K.square(self.critic_output_real - 1.), axis=-1)
        false_loss = K.mean(K.square(self.critic_output_fake), axis=-1)
        return true_loss + false_loss

    def _gen_loss(self):
        kl_loss = -0.5 * K.sum(
            1 + self.encoder_logvar_output - K.square(self.encoder_mean_output) - K.exp(self.encoder_logvar_output),
            axis=-1)
        gen_loss = K.mean(K.square(self.critic_output_real - self.critic_output_fake), axis=-1)
        rec_loss = K.mean(K.abs(self.h1_real - self.h1_fake), axis=[1, 2, 3]) \
                   + K.mean(K.abs(self.h2_real - self.h2_fake), axis=[1, 2, 3]) \
                   + K.mean(K.abs(self.h3_real - self.h3_fake), axis=[1, 2, 3])
        return 0.01 * kl_loss + gen_loss + rec_loss

    def _reconstruct_samples(self, data_gen, vis_id=0):
        x, _ = data_gen.next()
        x_gen = (self.rec_sample([x])[0] * 255.).astype('int') if x.shape[-1] > 1 else self.rec_sample([x])[0]

        f = plt.figure()
        plt.clf()
        for i in range(min(x.shape[0], 25)):
            plt.subplot(5, 5, i + 1)
            plt.imshow(x[i]) if x.shape[-1] > 1 else plt.imshow(np.squeeze(x[i]), cmap='gray')
            plt.axis('off')
        f.canvas.draw()
        plt.savefig('real_samples_e%i.eps' % vis_id)
        plt.close()

        f = plt.figure()
        plt.clf()
        for i in range(min(x.shape[0], 25)):
            plt.subplot(5, 5, i + 1)
            plt.imshow(x_gen[i]) if x.shape[-1] > 1 else plt.imshow(np.squeeze(x_gen[i]), cmap='gray')
            plt.axis('off')
        f.canvas.draw()
        plt.savefig('fake_samples_e%i.eps' % vis_id)
        plt.close()

    def _generate_samples(self, vis_id=0):
        n = np.random.randn(25, self.latent_dim)
        #n = np.ones(shape = (25, self.latent_dim)) * 0.5
        #n[..., 8] = np.linspace(-10, 10, 25) # change
        x_gen = self.gen_sample([n])[0]

        f = plt.figure()
        plt.clf()
        for i in range(min(n.shape[0], 25)):
            plt.subplot(5, 5, i + 1)
            plt.imshow(x_gen[i]) if x_gen.shape[-1] > 1 else plt.imshow(np.squeeze(x_gen[i]), cmap='gray')
            plt.axis('off')
        f.canvas.draw()
        plt.savefig('generated_samples_e%i.eps' % vis_id)
        plt.close()

    def train(self, train_dir, val_dir, epochs=10, batch_size=64):
        # Generators
        color_mode = 'rgb' if self.input_shape[-1] > 1 else 'grayscale'
        datagen = ImageDataGenerator(rescale=1. / 255, fill_mode='constant')
        train_gen = datagen.flow_from_directory(train_dir, target_size=self.input_shape[:2], interpolation='bilinear',
                                                color_mode=color_mode, class_mode='categorical', batch_size=batch_size)
        val_gen = datagen.flow_from_directory(val_dir, target_size=self.input_shape[:2], interpolation='bilinear',
                                              color_mode=color_mode, class_mode='categorical', batch_size=batch_size)

        steps_per_epoch = (np.ceil(train_gen.n / batch_size)).astype('int')
        for i in range(epochs):
            print('Epoch %i/%i' % (i + 1, epochs))
            pbar = Progbar(steps_per_epoch)
            self._reconstruct_samples(val_gen, i)
            for j in range(steps_per_epoch):
                x, _ = train_gen.next()
                critic_loss = self.critic_trainer.train_on_batch(x=x, y=None)
                gen_loss = self.gen_trainer.train_on_batch(x=x, y=None)
                pbar.update(j + 1, [('critic loss', critic_loss), ('generator loss', gen_loss)])

        # Save weights
        self.encoder.save_weights('./encoder.h5')
        self.decoder.save_weights('./decoder.h5')
        self.critic.save_weights('./critic.h5')

    def restore_weights(self):
        self.encoder.load_weights('./encoder.h5')
        self.decoder.load_weights('./decoder.h5')
        self.critic.load_weights('./critic.h5')

    def reconstruct_samples(self, dir, vis_id=0):
        color_mode = 'rgb' if self.input_shape[-1] > 1 else 'grayscale'
        datagen = ImageDataGenerator(rescale=1. / 255, fill_mode='constant')
        gen = datagen.flow_from_directory(dir, target_size=self.input_shape[:2], interpolation='bilinear',
                                          color_mode=color_mode, class_mode='categorical', batch_size=25)
        self._reconstruct_samples(gen, vis_id)

    def generate_samples(self, vis_id=0):
        self._generate_samples(vis_id)

    def compute_distance(self, dir, vis_id=0):
        color_mode = 'rgb' if self.input_shape[-1] > 1 else 'grayscale'
        datagen = ImageDataGenerator(rescale=1. / 255, fill_mode='constant')
        gen = datagen.flow_from_directory(dir, target_size=self.input_shape[:2], interpolation='bilinear',
                                          color_mode=color_mode, class_mode='categorical', batch_size=25)

        x, _ = gen.next()
        dist = self.compute_score([x])[0]

        f = plt.figure()
        plt.clf()
        for i in range(min(x.shape[0], 25)):
            plt.subplot(5, 5, i + 1)
            plt.imshow(x[i]) if x.shape[-1] > 1 else plt.imshow(np.squeeze(x[i]), cmap='gray')
            plt.title('d_%.3f' % dist[i])
            plt.axis('off')
        f.canvas.draw()
        plt.savefig('distance_samples_e%i.eps' % vis_id)
        plt.close()
