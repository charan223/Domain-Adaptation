# --------------------------------------------------------res
# Copyright (c) 2017 IIT KGP
# Licensed under The IIT KGP License [dont see LICENSE for details]
# Written by Charan Reddy
# --------------------------------------------------------

"""Train a ADDA video network."""
#Faster RCNN imports
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.train import get_data_layer,filter_roidb,get_training_roidb
import gt_data_layer.roidb as gdl_roidb
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from utils.timer import Timer
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg,cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from networks.factory import get_network

#Other imports
import numpy as np
import time, os, sys
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.contrib import slim
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorflow.contrib.learn.python.learn.dataframe.queues.feeding_queue_runner import FeedingQueueRunner
from collections import OrderedDict,deque
import argparse
import pprint
import logging
import random
import click
from tqdm import tqdm
import pickle
from contextlib2 import ExitStack
import tflearn
import logging.config
import os.path
import yaml
import pdb



#Command to Run
#python adda_model.py --target_weights /home/charan/target_rcnn/VGGnet_fast_rcnn_iter_70000.ckpt  --source_weights /home/charan/source_rcnn/VGGnet_fast_rcnn_iter_70000.ckpt --iters 5


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a ADDA video network')
    parser.add_argument('--device', dest='device', help='device to use -> gpu/cpu',
                        default='gpu', type=str)
    parser.add_argument('--device_id', dest='device_id', help='device id to use',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=70000, type=int)

    parser.add_argument('--target_weights', dest='target_pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--source_weights', dest='source_pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)

    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='/home/charan/Faster-RCNN_TF/experiments/cfgs/faster_rcnn_end2end.yml', type=str)

    parser.add_argument('--target_imdb', dest='target_imdb_name',
                        help='dataset to train on',
                        default='yto_trainval', type=str)
    parser.add_argument('--source_imdb', dest='source_imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)

    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')

    parser.add_argument('--target_network', dest='target_network_name',
                        help='name of the network',
                        default='VGGnet_train', type=str)
    parser.add_argument('--source_network', dest='source_network_name',
                        help='name of the network',
                        default='source_VGGnet_train', type=str)

    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

#initialize uninitialized variables
def initialize_uninitialized(sess):
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    print [str(i.name) for i in not_initialized_vars] # only for testing
    if len(not_initialized_vars):
   		     sess.run(tf.variables_initializer(not_initialized_vars))

#removes scope of the key to be stored in dict
def remove_first_scope(name):
    return '/'.join(name.split('/')[1:])

#collects adversary variables and returns a dictionary 
def collect_vars(scope, start=None, end=None):
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    var_dict = OrderedDict()
    for var in vars[start:end]:
        var_name = remove_first_scope(var.op.name)
        var_dict[var_name] = var
    return var_dict

#returns dictionary of target variables
def collect_vars_new(vars, start=None, end=None):
    var_dict = OrderedDict()
    for var in vars[start:end]:
        var_name = remove_first_scope(var.op.name)
        var_dict[var_name] = var
    return var_dict

#Discriminator functions as in DC gans
def linear(input_, output_size, scope=None, stddev=0.01, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("weights", [shape[1], output_size], tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable("biases", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

#custom_adversarial discriminator
def custom_adversarial_discriminator(net, layers, scope='adversary', leaky=False, reuse = False):
    with tf.variable_scope(scope):

            if reuse:
               tf.get_variable_scope().reuse_variables()
            if leaky:
               activation_fn = tflearn.activations.leaky_relu
            else:
               activation_fn = tf.nn.relu

	    net = activation_fn(linear(net, 500,'fully_connected'))
            net = activation_fn(linear(net, 500,'fully_connected_1'))
            net = linear(net, 2  ,'fully_connected_2')
    return net


class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, sess, target_vars, source_vars, target_restorer, source_restorer, target_network, source_network, target_imdb, source_imdb, target_roidb, source_roidb):
        """Initialize the SolverWrapper."""
        self.target_imdb = target_imdb
        self.source_imdb = source_imdb
        self.target_roidb = target_roidb
        self.source_roidb = source_roidb
	self.target_network = target_network
	self.source_network = source_network
	self.target_vars= target_vars
	self.source_vars= source_vars
        self.target_restorer = target_restorer
        self.source_restorer = source_restorer
        print 'Computing bounding-box regression targets...'
        if cfg.TRAIN.BBOX_REG:
            self.target_bbox_means, self.target_bbox_stds = rdl_roidb.add_bbox_regression_targets(target_roidb)
        if cfg.TRAIN.BBOX_REG:
            self.source_bbox_means, self.source_bbox_stds = rdl_roidb.add_bbox_regression_targets(source_roidb)
        print 'done'



    def train_model(self, sess, max_iters):

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        target_data_layer = get_data_layer(self.target_roidb, self.target_imdb.num_classes)
        source_data_layer = get_data_layer(self.source_roidb, self.source_imdb.num_classes)

	#inputs
        target_features = self.target_network.get_output('target/fc7')
	source_features = self.source_network.get_output('source/fc7')

        target_ft = target_features
        source_ft = source_features

	print('CheckPoint1: Source and Target features formed!')


        # adversarial network
	target_ft = tf.reshape(target_ft, [-1, int(target_ft.get_shape()[-1])])
	source_ft = tf.reshape(source_ft, [-1, int(source_ft.get_shape()[-1])])
	adversary_ft = tf.concat([source_ft, target_ft], 0)
	#source_adversary_label = tf.zeros([tf.shape(source_ft)[0]], tf.int32)
	#target_adversary_label = tf.ones([tf.shape(target_ft)[0]], tf.int32)
	#adversary_label = tf.concat([source_adversary_label, target_adversary_label], 0)
	#adversary_logits = adversarial_discriminator(adversary_ft, [500,500], leaky=True)
	
        D_logits = custom_adversarial_discriminator(source_ft, [500, 500], leaky=True, reuse=False)  # source logits (real)
        D_logits_ = custom_adversarial_discriminator(target_ft, [500, 500], leaky=True, reuse=True)  # target logits (fake)

        # discriminator losses

        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits, labels=tf.ones_like(D_logits)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.zeros_like(D_logits_)))
        d_loss = d_loss_real + d_loss_fake

        # generator losses: in our case target RCNN losses

        target_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.ones_like(D_logits_)))

        print('CheckPoint2: Adversarial Network Formed!')

	# losses
	#mapping_loss = tf.losses.sparse_softmax_cross_entropy(1 - adversary_label, adversary_logits)
	#adversary_loss = tf.losses.sparse_softmax_cross_entropy(adversary_label, adversary_logits)	
	print('CheckPoint3: Loss Operations Formed!')

	# variable collection
	target_vars = collect_vars_new(self.target_vars)
	adversary_vars = collect_vars('adversary')
	print('CheckPoint4: Variable Collection Done!')

	# optimizer
	lr = 0.0002
	lr_var = tf.Variable(lr, name='learning_rate', trainable=False)
	optimizer = tf.train.AdamOptimizer(lr_var, 0.5)


        lrt = 0.2
        lrt_var = tf.Variable(lrt, name='learning_rate_t', trainable=False)
        optimizer_t = tf.train.AdamOptimizer(lrt_var, 0.5)


	discriminator_step = optimizer.minimize(d_loss, var_list=list(adversary_vars.values()))
       	target_step = optimizer_t.minimize(target_loss, var_list=list(target_vars.values()))
 	print('CheckPoint5: Optimizer Declaration Done!')


	# optimization loop (finally)
	output_dir = os.path.join('snapshot/')
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	target_losses = deque(maxlen=10)
	disc_losses = deque(maxlen=10)

        initialize_uninitialized(sess)

        timer = Timer()
	snapping_iters = 1
	display_iters = 1
	filename= 'adda_'+str(0)+'.ckpt'
	snapshot_path = self.target_restorer.save(sess, os.path.join(output_dir, filename))
	print('Saved snapshot to {}'.format(snapshot_path))
	f=open('losses.txt','w')
	f.close()

        for i in range(max_iters):

            # get one batch
            target_blobs = target_data_layer.forward()
            source_blobs = source_data_layer.forward()
            so1 = source_blobs['data']
            so2 = source_blobs['data']
            print("**essentials ::", so1.shape, so1.min(), so1.max())
            print("**essentials ::", so2.shape, so2.min(), so2.max())

	    feed_dict={self.target_network.data: target_blobs['data'], self.target_network.im_info: target_blobs['im_info'], self.target_network.keep_prob: 0.5, self.target_network.gt_boxes: target_blobs['gt_boxes'], 
		self.source_network.data: source_blobs['data'], self.source_network.im_info: source_blobs['im_info'], self.source_network.keep_prob: 1.0, self.source_network.gt_boxes: source_blobs['gt_boxes']}


            print("will start training::")
            #t, s = sess.run([target_ft, source_ft], feed_dict = feed_dict)
            #res = t - s
            #print("**range: ", res.min(), res.max())
            #sys.exit()
            # Update the discriminator
            disc_loss_val, _ = sess.run([d_loss, discriminator_step], feed_dict = feed_dict)
            print("Disc trained")
            # Update the target generator 
            target_loss_val, _ = sess.run([target_loss, target_step], feed_dict = feed_dict)
            #target_loss_val = 0.
	    target_losses.append(target_loss_val)
	    disc_losses.append(disc_loss_val)
	    f=open('losses.txt','a')
	    if i % display_iters == 0:
			print >>f, ('{:} Target: {:10.4f}     (avg: {:10.4f})'
                        '    Disc: {:10.4f}     (avg: {:10.4f})'
                        .format('Iteration {}:'.format(i),
                                target_loss_val,
                                np.mean(target_losses),
                                disc_loss_val,
                                np.mean(disc_losses)))	    
	    if (i + 1) % snapping_iters == 0:
		    filename= 'adda_'+str(i+1)+'.ckpt'
		    snapshot_path = self.target_restorer.save(sess, os.path.join(output_dir, filename))
		    print('Saved snapshot to {}'.format(snapshot_path))
	    f.close()
	coord.request_stop()
	coord.join(threads)



if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    print('Using config:')
    pprint.pprint(cfg)
    device_name = '/{}:{:d}'.format(args.device,args.device_id)
    print device_name

    if args.device == 'gpu':
        cfg.USE_GPU_NMS = True
        cfg.GPU_ID = args.device_id
    else:
        cfg.USE_GPU_NMS = False
    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)

    target_imdb = get_imdb(args.target_imdb_name)
    target_imdb.competition_mode(args.comp_mode)
    print 'Loaded dataset `{:s}` for training'.format(target_imdb.name)
    target_roidb = get_training_roidb(target_imdb)


    source_imdb = get_imdb(args.source_imdb_name)
    source_imdb.competition_mode(args.comp_mode)
    print 'Loaded dataset `{:s}` for training'.format(source_imdb.name)
    source_roidb = get_training_roidb(source_imdb)



    target_network = get_network(args.target_network_name)
    print 'Use network `{:s}` in training'.format(args.target_network_name)
    target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='target')



    source_network = get_network(args.source_network_name)
    print 'Use network `{:s}` in training'.format(args.source_network_name)
    source_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='source')


    target_roidb = filter_roidb(target_roidb)
    source_roidb = filter_roidb(source_roidb)

    target_restorer = tf.train.Saver(var_list=target_vars,write_version=tf.train.SaverDef.V1)
    source_restorer = tf.train.Saver(var_list=source_vars,write_version=tf.train.SaverDef.V1)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options)) as sess:
	    target_restorer.restore(sess, args.target_pretrained_model)
	    source_restorer.restore(sess, args.source_pretrained_model)
	    sw = SolverWrapper(sess, target_vars, source_vars, target_restorer, source_restorer, target_network, source_network, target_imdb, source_imdb, target_roidb, source_roidb)
	    print 'Solving...'
	    sw.train_model(sess, args.max_iters)
	    print 'Done Solving'
