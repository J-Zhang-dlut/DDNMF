#-*-coding:UTF-8 -*-
import numpy as np
from sklearn.mixture import GaussianMixture
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.datasets import mnist, fashion_mnist
import tensorflow as tf

import warnings

warnings.filterwarnings("ignore")

import metrics
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
)


e1 = Sequential(Dense(500, activation='relu', kernel_initializer='glorot_uniform', name='e1'))
e2 = Sequential(Dense(500, activation='relu', kernel_initializer='glorot_uniform', name='e2'))
e3 = Sequential(Dense(2000, activation='relu', kernel_initializer='glorot_uniform', name='e3'))
nmf1 = Sequential(Dense(10, activation='sigmoid', kernel_initializer='glorot_uniform', use_bias=True, name='nmf1'))
nmf2 = Sequential(Dense(2000, activation='relu', kernel_initializer='glorot_uniform', use_bias=False, name='nmf2'))
d1 = Sequential(Dense(500, activation='relu', kernel_initializer='glorot_uniform', name='d1'))
d2 = Sequential(Dense(500, activation='relu', kernel_initializer='glorot_uniform', name='d2'))
d3 = Sequential(Dense(784, activation='relu', kernel_initializer='glorot_uniform', name='d3'))

# student net
inputs = Input(shape=(784,))
encoder_1 = e1(inputs)
encoder_2 = e2(encoder_1)
encoder_out = e3(encoder_2)
nmf_1 = nmf1(encoder_out)
decoder_in = nmf2(nmf_1)
decoder_1 = d1(decoder_in)
decoder_2 = d2(decoder_1)
decoder_out = d3(decoder_2)

# supervisor net
ae_inputs = Input(shape=(784,))
ae_encoder_1 = Dense(500, activation='relu', kernel_initializer='glorot_uniform', name='ae_e1')(ae_inputs)
ae_encoder_2 = Dense(500, activation='relu', kernel_initializer='glorot_uniform', name='ae_e2')(ae_encoder_1)
ae_encoder_out = Dense(2000, activation='relu', kernel_initializer='glorot_uniform', name='ae_e3')(ae_encoder_2)
ae_decoder_1 = Dense(500, activation='relu', kernel_initializer='glorot_uniform', name='ae_d1')(ae_encoder_out)
ae_decoder_2 = Dense(500, activation='relu', kernel_initializer='glorot_uniform', name='ae_d2')(ae_decoder_1)
ae_decoder_out = Dense(784, activation='relu', kernel_initializer='glorot_uniform', name='ae_d3')(ae_decoder_2)

model = Model(inputs=[inputs, ae_inputs], outputs=[decoder_out, ae_decoder_out])

# nc loss
model.add_loss(1 * tf.reduce_mean(tf.square(decoder_in - tf.identity(encoder_out))))

# ap loss
model.add_loss(0.01 * tf.reduce_mean(tf.square(encoder_1 - tf.identity(ae_encoder_1))))
model.add_loss(0.01 * tf.reduce_mean(tf.square(encoder_2 - tf.identity(ae_encoder_2))))
model.add_loss(0.01 * tf.reduce_mean(tf.square(encoder_out - tf.identity(ae_encoder_out))))
model.add_loss(0.01 * tf.reduce_mean(tf.square(decoder_1 - tf.identity(ae_decoder_1))))
model.add_loss(0.01 * tf.reduce_mean(tf.square(decoder_2 - tf.identity(ae_decoder_2))))
model.add_loss(0.01 * tf.reduce_mean(tf.square(decoder_out - tf.identity(ae_decoder_out))))

model.compile(optimizer='adam', loss='mse', loss_weights=[1., 1.])

def train_test(dataset='mnist', epochs=0, batchsize=128):
    if dataset == 'mnist':
        (x_train, _), (x_test, y_test) = mnist.load_data()
    elif dataset == 'fmnist':
        (x_train, _), (x_test, y_test) = fashion_mnist.load_data()
    else :
        print('wrong dataset')
        pass

    x_train = x_train.reshape((x_train.shape[0], -1))
    x_train = np.divide(x_train, 255.)
    noise_x_train = tf.nn.dropout(x_train, rate=0.1, seed=1)

    x_test = x_test.reshape((x_test.shape[0], -1))
    x_test = np.divide(x_test, 255.)

    model.fit([noise_x_train, x_train], [x_train, x_train], epochs=epochs, batch_size=batchsize)
    model.save_weights('saved_weights/{}_dn_nmf_300_gmm.h5'.format(dataset))

    x_test = e1(x_test)
    x_test = e2(x_test)
    x_test = e3(x_test)
    x_test = nmf1(x_test)
    gmm = GaussianMixture(n_components=10, covariance_type='spherical', random_state=15)
    y_pred = gmm.fit_predict(x_test)

    acc = np.round(metrics.cluster_acc(y_test, y_pred), 5)
    ari = np.round(metrics.ari(y_test, y_pred), 5)
    nmi = np.round(metrics.nmi(y_test, y_pred), 5)

    print('acc = %.5f, nmi = %.5f, ari = %.5f' % (acc, nmi, ari))

def load_test(dataset='mnist'):
    if dataset == 'mnist':
        (_, _), (x_test, y_test) = mnist.load_data()
    elif dataset == 'fmnist':
        (_, _), (x_test, y_test) = fashion_mnist.load_data()
    else:
        print('wrong dataset')
        pass

    x_test = x_test.reshape((x_test.shape[0], -1))
    x_test = np.divide(x_test, 255.)

    model.load_weights('saved_weights/{}_dn_nmf_300_gmm.h5'.format(dataset))

    x_test = e1(x_test)
    x_test = e2(x_test)
    x_test = e3(x_test)
    x_test = nmf1(x_test)
    gmm = GaussianMixture(n_components=10, covariance_type='spherical', random_state=15)
    y_pred = gmm.fit_predict(x_test)

    acc = np.round(metrics.cluster_acc(y_test, y_pred), 5)
    ari = np.round(metrics.ari(y_test, y_pred), 5)
    nmi = np.round(metrics.nmi(y_test, y_pred), 5)

    print('acc = %.5f, nmi = %.5f, ari = %.5f' % (acc, nmi, ari))


if __name__ == "__main__":
    train_test(dataset='mnist', epochs=300, batchsize=128)
    load_test(dataset='mnist')
