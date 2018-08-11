from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow.contrib.slim as slim
from functools import partial
from glob import glob

import cv2
import data
import image_utils as im
import models
import numpy as np
import tensorflow as tf
import utils
from utils import gradient_penalty
import importer_tar
    
def train():
    epoch = 200
    batch_size = 1
    lr = 0.0002
    crop_size = 128
    load_size = 128
    tar_db_a = "cameron_images.tgz"
    tar_db_b = "teresa_images.tgz"
    db_a_i = importer_tar.Importer(tar_db_a)
    db_b_i = importer_tar.Importer(tar_db_b)
    image_a_names = db_a_i.get_sorted_image_name()
    image_b_names = db_b_i.get_sorted_image_name()
    train_a_size = int(len(image_a_names) * 0.8)
    train_b_size = int(len(image_b_names) * 0.8)
    
    image_a_train_names = image_a_names[0:train_a_size]
    image_b_train_names = image_b_names[0:train_b_size]
    
    image_a_test_names = image_a_names[train_a_size:]
    image_b_test_names = image_b_names[train_b_size:]
    
    print("A train size:{},test size:{}".format(len(image_a_train_names),len(image_a_test_names)))
    print("B train size:{},test size:{}".format(len(image_b_train_names),len(image_b_test_names))) 
    
    
    """ graph """
    # models
    generator_a2b = partial(models.generator, scope='a2b')
    generator_b2a = partial(models.generator, scope='b2a')
    discriminator_a = partial(models.discriminator, scope='a')
    discriminator_b = partial(models.discriminator, scope='b')   
    
    # operations
    a_real_in = tf.placeholder(tf.float32, shape=[None, load_size, load_size, 3], name="a_real")
    b_real_in = tf.placeholder(tf.float32, shape=[None, load_size, load_size, 3], name="b_real")
    a_real = utils.preprocess_image(a_real_in, crop_size=crop_size)
    b_real = utils.preprocess_image(b_real_in, crop_size=crop_size)    
    
    a2b = generator_a2b(a_real)
    b2a = generator_b2a(b_real)
    b2a2b = generator_a2b(b2a)
    a2b2a = generator_b2a(a2b)
    
    a_logit = discriminator_a(a_real)
    b2a_logit = discriminator_a(b2a)
    b_logit = discriminator_b(b_real)
    a2b_logit = discriminator_b(a2b)
    
    # losses
    g_loss_a2b = -tf.reduce_mean(a2b_logit)
    g_loss_b2a = -tf.reduce_mean(b2a_logit)
   
    cyc_loss_a = tf.losses.absolute_difference(a_real, a2b2a)
    cyc_loss_b = tf.losses.absolute_difference(b_real, b2a2b)
    g_loss = g_loss_a2b + g_loss_b2a + cyc_loss_a * 10.0 + cyc_loss_b * 10.0
 
    wd_a = tf.reduce_mean(a_logit) - tf.reduce_mean(b2a_logit)
    wd_b = tf.reduce_mean(b_logit) - tf.reduce_mean(a2b_logit)
    
    gp_a = gradient_penalty(a_real, b2a, discriminator_a)
    gp_b = gradient_penalty(b_real, a2b, discriminator_b)
    
    d_loss_a = -wd_a + 10.0 * gp_a
    d_loss_b = -wd_b + 10.0 * gp_b
    
    # summaries
    utils.summary({g_loss_a2b: 'g_loss_a2b',
                               g_loss_b2a: 'g_loss_b2a',
                               cyc_loss_a: 'cyc_loss_a',
                               cyc_loss_b: 'cyc_loss_b'})    
    utils.summary({d_loss_a: 'd_loss_a'})    
    utils.summary({d_loss_b: 'd_loss_b'})
    
    for variable in slim.get_model_variables():
        tf.summary.histogram(variable.op.name, variable)
        
    merged = tf.summary.merge_all()
    
    # optim
    t_var = tf.trainable_variables()
    d_a_var = [var for var in t_var if 'a_discriminator' in var.name]
    d_b_var = [var for var in t_var if 'b_discriminator' in var.name]
    g_var = [var for var in t_var if 'a2b_generator' in var.name or 'b2a_generator' in var.name]
    
    d_a_train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(d_loss_a, var_list=d_a_var)
    d_b_train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(d_loss_b, var_list=d_b_var)
    g_train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(g_loss, var_list=g_var)    
    
    """ train """
    ''' init '''
    # session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    # counter
    it_cnt, update_cnt = utils.counter()    
    
    ''' summary '''
    summary_writer = tf.summary.FileWriter('./outputs/summaries/', sess.graph)
    
    ''' saver '''
    saver = tf.train.Saver(max_to_keep=5)
    
    ''' restore '''
    ckpt_dir = './outputs/checkpoints/'
    utils.mkdir(ckpt_dir)
    try:
        utils.load_checkpoint(ckpt_dir, sess)
    except:
        sess.run(tf.global_variables_initializer())
    
    '''train'''
    try:
        batch_epoch = min(train_a_size,train_b_size) // batch_size
        max_it = epoch * batch_epoch
        for it in range(sess.run(it_cnt), max_it):            
            sess.run(update_cnt)
            epoch = it // batch_epoch
            it_epoch = it % batch_epoch + 1
       
    
            # read data
            a_real_np = cv2.resize(db_a_i.get_image(image_a_train_names[it_epoch]),(load_size,load_size))     
            b_real_np = cv2.resize(db_b_i.get_image(image_b_train_names[it_epoch]),(load_size,load_size))                                              
            
            # train G
            sess.run(g_train_op, 
                     feed_dict={a_real_in: [a_real_np], b_real_in: [b_real_np]})       
            
            # train discriminator            
            sess.run([d_a_train_op, d_b_train_op], 
                         feed_dict={a_real_in: [a_real_np], 
                                    b_real_in: [b_real_np]})                        
            
            # make summary
            summary  = sess.run(merged, 
                        feed_dict={a_real_in: [a_real_np], 
                        b_real_in: [b_real_np]})            
            summary_writer.add_summary(summary, it)            
            
            # display
            if it % 100 == 0:
                print("Epoch: (%3d) (%5d/%5d)" % (epoch, it_epoch, batch_epoch))
    
            # save
            if (it + 1) % 1000 == 0:
                save_path = saver.save(sess, '{}/epoch_{}_{}.ckpt'.format(ckpt_dir, epoch, it_epoch))
                print('###Model saved in file: {}'.format(save_path))
    
            # sample
            if (it + 1) % 1000 == 0:
                a_test_index = int(np.random.uniform(high=len(image_a_test_names)))
                b_test_index = int(np.random.uniform(high=len(image_b_test_names)))
                a_real_np = cv2.resize(db_a_i.get_image(image_a_test_names[a_test_index]),(load_size,load_size))     
                b_real_np = cv2.resize(db_b_i.get_image(image_b_test_names[b_test_index]),(load_size,load_size))     
                
                [a_opt, a2b_opt, a2b2a_opt, b_opt, b2a_opt, b2a2b_opt] = sess.run([a_real, a2b, a2b2a, b_real, b2a, b2a2b], 
                    feed_dict={a_real_in: [a_real_np], b_real_in: [b_real_np]})                
                
                sample_opt = np.concatenate((a_opt, a2b_opt, a2b2a_opt, b_opt, b2a_opt, b2a2b_opt), axis=0)
    
                save_dir = './outputs/sample_images_while_training/'
                utils.mkdir(save_dir)
                im.imwrite(im.immerge(sample_opt, 2, 3), 
                           '{}/epoch_{}_it_{}.jpg'.
                           format(save_dir, epoch, it_epoch))
    except:
        raise
    
    finally:
        save_path = saver.save(sess, '{}/epoch_{}_{}.ckpt'.format(ckpt_dir, epoch, it_epoch))
        print('###Model saved in file: {}'.format(save_path))
        sess.close()
        


if __name__ == "__main__":
    train()