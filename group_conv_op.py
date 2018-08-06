# Tensorflow mandates these.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


@slim.add_arg_scope
def group_conv2d_by_depthwise_conv(inputs, num_outputs, kernel_size, num_groups=1,
                 stride=1, rate=1, padding='SAME',
                 activation_fn=tf.nn.relu,
                 normalizer_fn=None,
                 normalizer_params=None,
                 biases_initializer=tf.zeros_initializer(),
                 scope=None):
    with tf.variable_scope(scope, 'group_conv2d', [inputs]) as sc:
        biases_initializer = biases_initializer if normalizer_fn is None else None
        if num_groups == 1:
            return slim.conv2d(inputs, num_outputs, kernel_size,
                               stride=stride, rate=rate,
                               padding=padding,
                               activation_fn=activation_fn,
                               normalizer_fn=normalizer_fn,
                               normalizer_params=normalizer_params,
                               biases_initializer=biases_initializer,
                               scope=scope)
        else:
            depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)

            assert num_outputs % num_groups == 0, (
                "num_outputs=%d is not divisible by num_groups=%d" %
                (num_outputs, num_groups))
            assert depth_in % num_groups == 0, (
                "depth_in=%d is not divisible by num_groups=%d" %
                (depth_in, num_groups))

            group_size_in = depth_in // num_groups
            group_size_out = num_outputs // num_groups
            # By passing filters=None
            # separable_conv2d produces only a depthwise convolution layer
            net = slim.separable_conv2d(inputs=inputs,
                                        num_outputs=None,
                                        kernel_size=kernel_size,
                                        depth_multiplier=group_size_out,
                                        stride=stride,
                                        padding=padding,
                                        rate=rate,
                                        activation_fn=None,
                                        normalizer_fn=None,
                                        biases_initializer=biases_initializer,
                                        scope=scope)
            net_shape = net.shape.as_list()
            net = tf.reshape(net, net_shape[:3] + [num_groups, group_size_in,
                                                   group_size_out])
            net = tf.reduce_sum(net, axis=4)
            net = tf.reshape(net, net_shape[:3] + [num_outputs])

            if normalizer_fn is not None:
                normalizer_params = normalizer_params or {}
                net = normalizer_fn(net, **normalizer_params)
            if activation_fn is not None:
                net = activation_fn(net)
            return net


@slim.add_arg_scope
def group_conv2d(inputs, num_outputs, kernel_size, num_groups=1,
                 stride=1, rate=1, padding='SAME',
                 activation_fn=tf.nn.relu,
                 normalizer_fn=None,
                 normalizer_params=None,
                 biases_initializer=tf.zeros_initializer(),
                 scope=None):
    with tf.variable_scope(scope, 'group_conv2d', [inputs]) as sc:
        biases_initializer = biases_initializer if normalizer_fn is None else None
        if num_groups == 1:
            return slim.conv2d(inputs, num_outputs, kernel_size,
                               stride=stride, rate=rate,
                               padding=padding,
                               activation_fn=activation_fn,
                               normalizer_fn=normalizer_fn,
                               normalizer_params=normalizer_params,
                               biases_initializer=biases_initializer,
                               scope=scope)
        else:
            depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)

            assert num_outputs % num_groups == 0, (
                "num_outputs=%d is not divisible by num_groups=%d" %
                (num_outputs, num_groups))
            assert depth_in % num_groups == 0, (
                "depth_in=%d is not divisible by num_groups=%d" %
                (depth_in, num_groups))

            group_size_out = num_outputs // num_groups
            input_slices = tf.split(inputs, num_groups, axis=-1)
            output_slices = [slim.conv2d(inputs=input_slice,
                                         num_outputs=group_size_out,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         rate=rate,
                                         padding=padding,
                                         activation_fn=None,
                                         normalizer_fn=None,
                                         biases_initializer=biases_initializer,
                                         scope=scope + '/group%d' % idx)
                             for idx, input_slice in enumerate(input_slices)]
            net = tf.concat(output_slices, axis=-1)

            if normalizer_fn is not None:
                normalizer_params = normalizer_params or {}
                net = normalizer_fn(net, **normalizer_params)
            if activation_fn is not None:
                net = activation_fn(net)
            return net
