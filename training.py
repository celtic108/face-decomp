import numpy as np 
import tensorflow as tf 
import image_utils
from data_loader import get_batch
import training_params as params
import network_utils
import time

def update_running_loss(rl, l, i):
    i = min(i, 99)
    rl = (rl * i + l) / (i + 1)
    return rl

# Training phase indicates the number of reduction layers in the network. 
# It is compensated for by resizing the input images.
training_phase = 0 

training = tf.placeholder(tf.bool)
X = tf.placeholder(tf.float32, (None, params.shape[0], params.shape[1], 3))
print("*^*&^*&^*&^*&^")    
print(X.shape)
print("*^*&^*&^*&^*&^")    

mu_id, std_id, mu_pose, std_pose = network_utils.encoder(X, params.batch_size, params.latent_size, params.shape, False, training)
encoded_id = network_utils.sampling_trick(mu_id, std_id)
encoded_pose = network_utils.sampling_trick(mu_pose, std_pose)

output = network_utils.decoder(encoded_id, encoded_pose, params.batch_size, params.latent_size, params.shape)    
injected_id = tf.placeholder(tf.float32, [None, params.latent_size], name='injected_id')
injected_pose = tf.placeholder(tf.float32, [None, params.latent_size], name='injected_pose')
fabricated_decoder = network_utils.decoder(injected_id, injected_pose, params.batch_size, params.latent_size, params.shape, reuse=True)

real_disc = network_utils.discriminator(X, params.shape, params.batch_size, params.latent_size, training=training)
fake_disc = network_utils.discriminator(output, params.shape, params.batch_size, params.latent_size, reuse=True, training=training)
reencoded_ids, _, _, _ = network_utils.encoder(output, params.batch_size, params.latent_size, params.shape, True, training)
    

# encoded_ids = tf.split(mid, 2, axis=-1)[0]
loggamma_x = tf.get_variable('loggamma_x', [], tf.float32, tf.zeros_initializer())
gamma_x = tf.exp(loggamma_x)

encoder_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder_')
if params.phase > 1:
    decoder_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder_')
if params.phase > 2:
    discriminator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator_')

split_ids = tf.split(mu_id, params.batch_size, axis = 0)


id_loss = 0
number_of_pairs = ((params.batch_size + 1) // 2) // 2
print("NUMBER OF PAIRS: ", number_of_pairs)
for i in range(params.batch_size):
    for j in range(i + 1, params.batch_size):
        if i < number_of_pairs * 2 - 1 and i % 2 == 0 and j == i + 1:
            id_loss += 100*tf.reduce_mean(tf.pow(split_ids[i] - split_ids[j], 2))
            print(i, ",", j, "-PAIR")
        else:
            # id_loss += tf.maximum(-1*tf.reduce_mean(tf.pow(split_ids[i] - split_ids[j], 2)) + 2., 0)
            print(i, ",", j, '-NOT A PAIR')
id_loss /= ((params.batch_size * (params.batch_size - 1)) / 2)
pixel_loss = tf.reduce_mean(tf.pow((10*(X - output)) / gamma_x, 2) / 2.0 + loggamma_x)
#train_disc_true = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
#    labels=tf.ones_like(real_disc, dtype=tf.int32), logits=real_disc))
train_disc_true = tf.reduce_mean(tf.pow(tf.ones_like(real_disc) - real_disc, 2))
#train_disc_false = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
#    labels=tf.zeros_like(fake_disc, dtype=tf.int32), logits=fake_disc))
train_disc_false = tf.reduce_mean(tf.pow(tf.zeros_like(fake_disc) - fake_disc, 2))
#trick_disc_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
#    labels=tf.ones_like(fake_disc, dtype=tf.int32), logits=fake_disc))
trick_disc_loss = tf.reduce_mean(tf.pow(tf.ones_like(fake_disc) - fake_disc, 2))
id_regen_loss = 100*network_utils.get_loss(mu_id, reencoded_ids)

# update_group = []
# if params.phase > 1:
#     update_pix = tf.train.AdamOptimizer(0.000005).minimize(pixel_loss, var_list=decoder_vars+encoder_vars)
#     update_group.append(update_pix)
# if params.phase > 2:
#     train_disc = tf.train.AdamOptimizer(0.00001).minimize(train_disc_true + train_disc_false, var_list=discriminator_vars)
#     update_group.append(train_disc)
#     trick_disc = tf.train.AdamOptimizer(0.00001).minimize(trick_disc_loss, var_list=decoder_vars+encoder_vars)
#     update_group.append(trick_disc)
#     id_regen_op = tf.train.AdamOptimizer(0.0001).minimize(id_regen_loss, var_list=decoder_vars+encoder_vars)
#     update_group.append(id_regen_op)
# id_update = tf.train.AdamOptimizer(0.00002).minimize(id_loss, var_list=encoder_vars)
# update_group.append(id_update)
# update_op = tf.group(*update_group)

id_kl_loss = 0.001*network_utils.get_kl_loss(mu_id, std_id)
pose_kl_loss = 0.001*network_utils.get_kl_loss(mu_pose, std_pose)

encoder_loss = pixel_loss + id_loss + id_kl_loss + pose_kl_loss + 0*id_regen_loss
decoder_loss = pixel_loss + id_regen_loss + trick_disc_loss
discriminator_loss = train_disc_true + train_disc_false

loss = encoder_loss + decoder_loss + discriminator_loss

lr = tf.placeholder(tf.float32, [], name='learning_rate')
encoder_update = tf.train.AdamOptimizer(lr).minimize(encoder_loss, var_list = encoder_vars + [loggamma_x])
decoder_update = tf.train.AdamOptimizer(lr).minimize(decoder_loss, var_list = decoder_vars + [loggamma_x])
discriminator_update = tf.train.AdamOptimizer(0.01*lr).minimize(discriminator_loss, var_list = discriminator_vars)


init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    try:
        saver.restore(sess, params.save_path)
    except:
        print("COULD NOT RESTORE MODEL. TRAINING FROM SCATCH")
    running_loss = 0
    iterations = 0
    while True:
        try:
            start_time = time.time()
            next_batch = get_batch()
            # tdt, tdf, tdl, kdl, pl, irl, il = sess.run([train_disc_true, train_disc_false, train_disc_loss, trick_disc_loss, pixel_loss, id_regen_loss, id_loss], 
            #                                   feed_dict={X: next_batch, training:False})
            # print(tdt, tdf, tdl, kdl, pl, irl, il)
            # tdt, tdf, tdl, kdl, pl, irl, il = sess.run([train_disc_true, train_disc_false, train_disc_loss, trick_disc_loss, pixel_loss, id_regen_loss, id_loss], 
            #                                   feed_dict={X: next_batch, training:True})
            # print(tdt, tdf, tdl, kdl, pl, irl, il)
            # print(tdl + kdl + pl + irl + il)
            #l, _, pl = sess.run([loss, update_op, pixel_loss], feed_dict={X: next_batch, training:True})
            runners = [encoder_loss, decoder_loss, discriminator_loss,
                       encoder_update, decoder_update, discriminator_update,
                       id_kl_loss, pose_kl_loss, pixel_loss, id_regen_loss, id_loss,
                       trick_disc_loss, train_disc_true, train_disc_false]
            feed_dict = {X: next_batch, training:True, lr:params.learning_rate}
            el, dl, dcl, _, _, _, idkl, pkl, pl, idrgl, idl, txdl, tdt, tdf = sess.run(runners, feed_dict=feed_dict)
            l = el + dl + dcl
            running_loss = update_running_loss(running_loss, l, iterations)
            print("{0}\t{1:.3f}\t{2:.4f}\t{3:.3f}\t{4:.3f}\t|{5:.3f}|\t{6:.3f}\t{7:.3f}\t{8:.3f}\t{9:.3f}\t{10:.3f}".format(iterations, 
                                                                     l, running_loss, idkl, pkl, pl, idrgl, idl, txdl, tdt, tdf))
            iterations += 1
        except KeyboardInterrupt:
            # tdt, tdf, tdl, kdl, pl, irl, il = sess.run([train_disc_true, train_disc_false, train_disc_loss, trick_disc_loss, pixel_loss, id_regen_loss, id_loss], 
            #                                   feed_dict={X: next_batch, training:False})
            # print(tdt, tdf, tdl, kdl, pl, irl, il)
            mi = sess.run(mu_id, feed_dict={X: next_batch, training:True})
            print("%^%^%^%^%^")
            print(mi.shape)
            running_mi_loss = 0
            for i in range(mi.shape[0]):
                for j in range(i+1, mi.shape[0]):
                    if i < number_of_pairs * 2 - 1 and i % 2 == 0 and j == i + 1:
                        temp_mi_loss = 100*np.average(np.square(mi[i] - mi[j]))
                        print(i, ",", j, "-PAIR", temp_mi_loss, np.sum(np.square(mi[i], np.zeros_like(mi[i])))**0.5)
                    else:
                        temp_mi_loss = 20*max(-1*np.average(np.square(mi[i] - mi[j])) + 2., 0)
                        print(i, ",", j, '-NOT A PAIR', temp_mi_loss, np.sum(np.square(mi[i], np.zeros_like(mi[i])))**0.5)
                    print(np.sum(np.square(mi[i] - mi[j]))**0.5)
                    running_mi_loss += temp_mi_loss
            print("MI LOSS: ", running_mi_loss / ((params.batch_size * (params.batch_size - 1)) / 2))
            o, lgx = sess.run([output, loggamma_x], feed_dict={X: next_batch, training:True})
            print(o.shape)
            for j in range(params.batch_size):
                image_utils.display_image(image_utils.unpreprocess_image(o[j]))
                image_utils.display_image(image_utils.unpreprocess_image(next_batch[j]))
                print("IMAGE ",j,": ", np.average(np.power(10*(o[j]-next_batch[j]) / np.exp(lgx), 2) / 2.0 + lgx))
                diff_img = (np.power(o[j] - next_batch[j], 2) - 2.0) / 2.0
                image_utils.display_image(image_utils.unpreprocess_image(diff_img))
            print("TOTAL LOSS: ", np.average(np.power(10*(o - next_batch) / np.exp(lgx), 2) / 2.0 + lgx))
            print("LOGGAMMA_X: ", lgx)
            # var_img = np.std(o, axis=0)
            # max_var = np.max(var_img)
            # min_var = np.min(var_img)
            # image_utils.display_image(image_utils.unpreprocess_image(2*(var_img - min_var) / (max_var - min_var) - 1))
            image_utils.display_image(image_utils.unpreprocess_image(np.average(next_batch, axis=0)))
            sample_ids, sample_poses = sess.run([encoded_id, encoded_pose], feed_dict={X: next_batch, training:True})
            interpolated_id = np.stack([sample_ids[0]]*params.batch_size, axis=0)
            interpolated_poses = [sample_poses[0] + j*(sample_poses[1] - sample_poses[0]) / (params.batch_size*2) for j in range(params.batch_size*2)]
            for j in range(2):
                poses_to_inject = np.stack(interpolated_poses[j*params.batch_size:(j+1)*params.batch_size])
                print(interpolated_id.shape)
                print(poses_to_inject.shape)
                print("%$%$%$%$%$%$%$%")
                interpolated_images = sess.run(fabricated_decoder, feed_dict={injected_id:interpolated_id, injected_pose:poses_to_inject})
                for k in range(params.batch_size):
                    image_utils.display_image(image_utils.unpreprocess_image(interpolated_images[k]))
            q = input("WOULD YOU LIKE TO QUIT?: ")
            if q in ['YES', "Y", 'y', 'yes', 'Yes']:
                ans = input("WOULD YOU LIKE TO SAVE: ")
                if ans in ['YES', "Y", 'y', 'yes', 'Yes']:
                    saver.save(sess, params.save_path)
                break
    