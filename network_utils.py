import tensorflow as tf 
from math import log
import numpy as np 
import training_params as params

initializer = tf.initializers.random_normal(0, 0.001)    

def encoder(net, batch_size, latent_size=1024, input_shape=(250,250), reuse=False, training=True):
    conv_blocks = int(log(input_shape[0], 2))
    base_chans = 32 #int(latent_size / (2**((conv_blocks - 1)//2)))
    with tf.variable_scope('encoder_', reuse = reuse):
        for j in range(conv_blocks - 2):
            #curr_channels = min(latent_size, 32 * 2**(j//2))
            curr_channels = base_chans * 2**(j//2)
            for i in range(2):
                net = tf.layers.conv2d(net, curr_channels, 3, padding='same',
                                    kernel_initializer = initializer,
                                    activation = tf.nn.leaky_relu, name=str(j)+str(i)) 
                net = tf.layers.batch_normalization(net, training=training, name='bn'+str(j)+str(i))  
            net = tf.layers.conv2d(net, curr_channels, 3, strides=(2,2), 
                                kernel_initializer = initializer, padding='same',
                                activation = tf.nn.leaky_relu, name=str(j)+str(i)+'m')
            net = tf.layers.batch_normalization(net, training=training, name='bn'+str(j)+str(i)+'m') 
        

        nets = tf.split(net, batch_size, axis=0)
        reshaped_nets = []
        for n in nets:
            reshaped_nets.append(tf.layers.flatten(n))
        net = tf.squeeze(tf.stack(reshaped_nets, axis=0), axis=1)
        id_embed = tf.layers.dense(net, latent_size*4, activation=tf.nn.leaky_relu, use_bias=True, kernel_initializer=initializer, name='id_enc_dense_1')
        id_embed = tf.layers.dense(id_embed, latent_size*4, activation=tf.nn.leaky_relu, use_bias=True, kernel_initializer=initializer, name='id_enc_dense_2')
        id_embed = tf.layers.dense(id_embed, latent_size*2, activation=tf.nn.leaky_relu, use_bias=True, kernel_initializer=initializer, name='id_encoder_final')              
        mu_id, logstd_id = tf.split(id_embed, 2, axis=1)
        std_id = tf.exp(logstd_id)

        pose_embed = tf.layers.dense(net, latent_size*4, activation=tf.nn.leaky_relu, use_bias=True, kernel_initializer=initializer, name='pose_enc_dense_1')
        pose_embed = tf.layers.dense(pose_embed, latent_size*4, activation=tf.nn.leaky_relu, use_bias=True, kernel_initializer=initializer, name='pose_enc_dense_2')
        pose_embed = tf.layers.dense(pose_embed, latent_size*2, activation=tf.nn.leaky_relu, use_bias=True, kernel_initializer=initializer, name='pose_encoder_final')              
        mu_pose, logstd_pose = tf.split(pose_embed, 2, axis=1)
        std_pose = tf.exp(logstd_pose)

    return mu_id, std_id, mu_pose, std_pose

def sampling_trick(mu, std):
    return mu + std * tf.random_normal([params.batch_size, params.latent_size])

def get_kl_loss(mu, std):
    return tf.reduce_mean(tf.reduce_sum(tf.square(mu) + tf.square(std) - 0.5 * tf.log(std + 1e-6) - 0.6, axis = [1]) / 2.0)

def fix_dims(t):
    return tf.expand_dims(tf.expand_dims(t, 1), 1)

# ORIGINAL ADAIN
# def adain(x, y, c, u, i):
#     with tf.variable_scope('adain_'+str(i), reuse = False):
#         d = fix_dims(tf.layers.dense(y, c*2, kernel_initializer = initializer, use_bias = u, name='adain_dense'+str(i)))
#         z = tf.split(d, 2, axis=3)
#         mu_x, var_x = tf.nn.moments(x, axes=[1,2], keep_dims=True)
#         std_x = tf.sqrt(var_x)
#         mu_y = z[0]
#         std_y = tf.pow(z[1], 2)
#         norm_x = (x - mu_x) / (std_x + 1e-6)
#         a = std_y * norm_x 
#     return a + mu_y
    
# ADAIN from my MNIST project
def adain(content, style, size, u, i):
    epsilon = 1e-6
    with tf.variable_scope('adain_'+str(i), reuse=False):
        style = tf.layers.dense(style, size*2, kernel_initializer=initializer, use_bias=u, name='adain_dense'+str(i))
        meanS, logvarS = tf.split(style, 2, axis=-1)
        meanS = tf.reshape(meanS, [-1,1,1,size])
        varS = tf.reshape(tf.exp(logvarS), [-1,1,1,size])
        meanC, varC = tf.nn.moments(content, [1,2], keep_dims = True)
        sigmaC = tf.sqrt(tf.add(varC, epsilon))
        sigmaS = tf.sqrt(tf.add(varS, epsilon))
        return (content - meanC) * sigmaS / sigmaC + meanS


def decoder(id, latent_space, batch_size=32, latent_size=1024, output_shape=(250,250), add_noise=True, MLP_inputs=True, use_bias=True, reuse=False):
    with tf.variable_scope('decoder_', reuse = reuse):
        conv_blocks = int(log(output_shape[0], 2)) - 1
        n = 1. if add_noise else 0.
        
        # Latent space is the pose. The id is used to create the base value
        if MLP_inputs:
            for k in range(8):
                latent_space = tf.layers.dense(latent_space, latent_size, tf.nn.leaky_relu, kernel_initializer = initializer, 
                                               use_bias = use_bias, name='MLP_layers_'+str(k))
        
        #init_val = np.random.normal(size=(1, 4, 4, latent_size)).astype(np.float32)
        #base_image = tf.Variable(init_val, dtype=tf.float32)
        # dec = tf.compat.v1.get_variable('init_val', [4, 4, latent_space.shape[1].value], 
        #                                 trainable=True,
        #                                 initializer=initializer)
        # dec = tf.stack([dec] * batch_size, name='base_img')
        dec = tf.layers.dense(id, 4*4*latent_size, activation=tf.nn.leaky_relu, kernel_initializer=initializer, name='decoder_base')
        decs = tf.split(dec, batch_size, axis =0)
        reshaped_decs = []
        for d in decs:
            reshaped_decs.append(tf.reshape(d, [4,4,latent_size]))
        dec = tf.stack(reshaped_decs, axis=0)
        #dec = tf.reshape(dec, [batch_size,4,4,latent_size])
        
        for c in range(conv_blocks):
            curr_channels = max(16, int(latent_size / 2 ** c))
            if c != 0:
                dec = tf.layers.conv2d_transpose(dec, curr_channels, 3, strides = (2,2), 
                                                kernel_initializer = initializer, name = 'dec_conv_trans_'+str(c),
                                                activation = tf.nn.leaky_relu, padding='same', use_bias = use_bias)
                dec = tf.layers.conv2d(dec, curr_channels, 3, kernel_initializer = initializer, name = 'conv_trans_1_'+str(c),
                                    activation = tf.nn.leaky_relu, padding='same', use_bias = use_bias)
            dec = adain(dec, latent_space, curr_channels, use_bias, c*2)
            dec = tf.layers.conv2d(dec, curr_channels, 3, kernel_initializer = initializer, name = 'conv_trans_2_'+str(c),
                                activation = tf.nn.leaky_relu, padding='same', use_bias = use_bias)
            dec = adain(dec, latent_space, curr_channels, use_bias, c*2+1)
        
        dec = tf.layers.conv2d(dec, (curr_channels + 3)//2, 3, kernel_initializer = initializer, name = 'dec_conv_penult',
                            activation = tf.nn.leaky_relu, padding='same', use_bias = use_bias)        
        output = tf.layers.conv2d(dec, 3, 1, kernel_initializer = initializer, name = 'dec_output',
                                activation = tf.nn.tanh, padding='valid', use_bias = use_bias)
    
    return output
    

def get_loss(x, y):
    return tf.reduce_mean(tf.pow((x - y), 2))


def discriminator(net, input_shape, batch_size, latent, reuse = False, training=True):
    conv_blocks = int(log(input_shape[0], 2)) // 2 -1
    base_chans = 16
    with tf.variable_scope('discriminator_', reuse = reuse):
        for j in range(conv_blocks):
            #curr_channels = min(latent_size, 32 * 2**(j//2))
            curr_channels = base_chans * 2**((j+1)//2)
            for i in range(2):
                net = tf.layers.conv2d(net, curr_channels, 3, padding='same',
                                       kernel_initializer = initializer,
                                       activation = tf.nn.leaky_relu, name=str(j)+str(i)) 
                net = tf.layers.batch_normalization(net, training=training, name='bn'+str(j)+str(i))                
                net = tf.layers.conv2d(net, curr_channels, 3, strides=(2,2), 
                                       kernel_initializer = initializer, padding='same',
                                       activation = tf.nn.leaky_relu, name=str(j)+str(i)+'m')
                net = tf.layers.batch_normalization(net, training=training, name='bn'+str(j)+str(i)+'m')  
        nets = tf.split(net, batch_size, axis =0)
        reshaped_nets = []
        for n in nets:
            reshaped_nets.append(tf.reshape(n, [latent]))
        net = tf.stack(reshaped_nets, axis=0)
        net = tf.layers.dense(net, 1, activation = tf.nn.sigmoid, name='disc_output')
    return net #tf.reshape(net, (-1,2))