import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
import keras.backend as K
import matplotlib.pyplot as plt
from scipy import misc
import os

dropout , z_size= 0.4 , 128
w , h , chanel = 64, 64, 3
batch_sz ,n_critic , epochs = 64 , 5 , 10000
lr , bet1 , bet2 = 0.004 ,0.5 , 0.9
path = './faces/faces/'
save_path = './ani_out/'
ckpt_dir = './tf_wgan-gp_animation_ckpt'


def generator():
    G = Sequential()
    G.add(Dense(4*4*1024, input_dim=z_size))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))
    G.add(Reshape((4, 4, 1024)))
    G.add(Dropout(dropout))

    G.add(Conv2DTranspose(512 ,5,strides=2 , padding='same'))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))

    G.add(Conv2DTranspose(256, 5, strides=2, padding='same'))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))
    G.add(Dropout(dropout))

    G.add(Conv2DTranspose(128, 5, strides=2, padding='same'))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))

    G.add(Conv2DTranspose(3, 5,strides=2 ,padding='same'))
    G.add(Activation('sigmoid'))
    G.summary()
    return G


def discriminator():

    D = Sequential()
    input_shape = ( w, h, chanel)
    D.add(Conv2D(64, 5, strides=2, input_shape=input_shape,padding='same'))
    D.add(BatchNormalization(momentum=0.9))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(dropout))

    D.add(Conv2D(128, 5, strides=2, padding='same'))
    D.add(BatchNormalization(momentum=0.9))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(dropout))

    D.add(Conv2D(256, 5, strides=2, padding='same'))
    D.add(BatchNormalization(momentum=0.9))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(dropout))

    D.add(Flatten())
    D.add(Dense(1))
    D.summary()
    return D


def read_batch_data():
    imgs = []
    imgs_name = os.listdir(path)
    ix = np.random.randint(0,len(imgs_name),batch_sz)
    for i in ix:
        imgs.append(misc.imresize(misc.imread(path+imgs_name[i]),[64,64,3])/ 255.0)
    return imgs

def train():
    G = generator()
    D = discriminator()
    Z = tf.placeholder(tf.float32, [None, z_size])
    X = tf.placeholder(tf.float32, [None, w,h,chanel])

    X_fake = G(Z)
    Real_loss = D(X)
    Fake_loss = D(X_fake)

    eps = tf.random_uniform([tf.shape(Z)[0], 1, 1, 1], minval=0., maxval=1.)
    X_i = eps * X + (1 - eps) * X_fake
    Inner_loss = D(X_i)
    X_g = tf.gradients(Inner_loss, [X_i])[0]
    X_g = tf.reduce_mean(tf.square(tf.sqrt(tf.reduce_sum(X_g ** 2, axis=1)) - 1)) * 10

    g_loss = -tf.reduce_mean(Fake_loss)
    d_loss = tf.reduce_mean(Fake_loss) - tf.reduce_mean(Real_loss) + X_g

    g_update = tf.train.AdamOptimizer(lr, bet1, bet2).minimize(g_loss, var_list=G.trainable_weights)
    d_update = tf.train.AdamOptimizer(lr, bet1, bet2).minimize(d_loss, var_list=D.trainable_weights)

    if os.path.exists(save_path) == False:
        os.makedirs(save_path)

    if os.path.exists(ckpt_dir) == False:
        os.makedirs(ckpt_dir)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("continue training.....")
            saver.restore(sess, ckpt.model_checkpoint_path)
            start = global_step.eval()
            print(start)

        for epo in range(start+1,epochs):
            for i in range(n_critic):
                x = read_batch_data()
                z = np.random.uniform(-1, 1, size=[batch_sz, z_size])
                ds, _ = sess.run([d_loss, d_update], feed_dict={X: x, Z: z})
            z = np.random.uniform(-1, 1, size=[batch_sz, z_size])
            gs, _ = sess.run([g_loss, g_update], feed_dict={Z: z})

            print("iter: %d , gs: %f , ds: %f" % (epo, gs, ds))
            global_step.assign(epo).eval()
            saver.save(sess, ckpt_dir + '/wgan_gp.ckpt',global_step = global_step)

            if epo % 10 == 0:
                images = sess.run(X_fake, feed_dict={Z: z})[:9]
                plt.figure(figsize=(8, 8))
                for i in range(images.shape[0]):
                    plt.subplot(3, 3, i + 1)
                    plt.imshow(images[i])
                    plt.axis('off')
                plt.tight_layout()
                plt.savefig("ani_out/%d.png" % epo)
                plt.close('all')

if __name__ == '__main__':
    train()
