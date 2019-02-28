# calculate kernels in x an y axis for bicubic downsampling
import numpy as np
from math import ceil
import tensorflow as tf


def cubic(x, scale):
    assert scale <= 1
    x = np.array(x * scale).astype(np.float64)
    abs_x = np.absolute(x)
    abs_x2 = np.multiply(abs_x, abs_x)
    abs_x3 = np.multiply(abs_x2, abs_x)
    w = np.multiply(1.5*abs_x3 - 2.5*abs_x2 + 1, abs_x <= 1) + \
        np.multiply(-0.5*abs_x3 + 2.5*abs_x2 - 4*abs_x + 2, (1 < abs_x) & (abs_x <= 2))
    return w * scale


def kernel(in_length, out_length):
    # assume in_length is larger scale
    # decide whether a convolution kernel can be constructed
    assert in_length >= out_length and in_length / out_length == 1.0 * in_length / out_length

    # decide kernel width
    scale = 1.0 * out_length / in_length
    kernel_length = 4.0 / scale

    # calculate kernel weights and padding (symmetric)
    x = np.array([1, out_length]).astype(np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_length / 2)
    p = int(ceil(kernel_length)) + 2
    ind = np.expand_dims(left, axis=1) + np.arange(p) - 1  # -1 because indexing from 0
    indices = ind.astype(np.int32)
    weights = cubic(np.expand_dims(u, axis=1) - indices - 1, scale)  # -1 because indexing from 0
    weights = np.divide(weights, np.expand_dims(np.sum(weights, axis=1), axis=1))
    ind2store = np.nonzero(np.any(weights, axis=0))
    weights = weights[:, ind2store].squeeze()
    indices = indices[:, ind2store].squeeze()
    assert (weights[0] == weights[-1]).all()
    pad_l = int(np.where(indices[0] == 0)[0])
    pad_r = len(indices[-1]) - 1 - int(np.where(indices[-1] == in_length-1)[0])

    return weights[0], [pad_l, pad_r]


def construct_tf_kernels(in_size, out_size):
    kernels, padding = [], []
    for i in range(2):
        k, p = kernel(in_size[0], out_size[0])
        k = np.expand_dims(k, axis=1-i)
        for _ in range(2):
            k = np.expand_dims(k, axis=-1)
        kernels.append(k)
        padding.append(p)
    return kernels, padding


def back_projection_loss(tf_input, tf_output):
    # assume tf_input is LR image, and tf_output is HR image

    stride_h = tf_output.get_shape()[1].value / tf_input.get_shape()[1].value
    stride_w = tf_output.get_shape()[2].value / tf_input.get_shape()[2].value

    in_shape = tf_input.get_shape()[1].value, tf_input.get_shape()[2].value
    out_shape = tf_output.get_shape()[1].value, tf_output.get_shape()[2].value
    num_channels = tf_input.get_shape()[-1].value

    # get bicubic downsampling kernels
    k, p = construct_tf_kernels(out_shape, in_shape)
    tf_kernel_h = tf.constant(k[0], dtype=tf.float32)
    tf_kernel_w = tf.constant(k[1], dtype=tf.float32)

    # downsample the output image
    channels = []
    for i in range(num_channels):
        c = tf.slice(tf_output, [0, 0, 0, i], [-1, -1, -1, 1])
        slice_pad = tf.pad(c, [[0, 0], [p[0][0], p[0][1]], [0, 0], [0, 0]], 'SYMMETRIC')
        tmp = tf.nn.conv2d(slice_pad, tf_kernel_h, strides=[1, stride_h, 1, 1], padding='VALID')
        slice_pad = tf.pad(tmp, [[0, 0], [0, 0], [p[1][0], p[1][1]], [0, 0]], 'SYMMETRIC')
        tmp = tf.nn.conv2d(slice_pad, tf_kernel_w, strides=[1, 1, stride_w, 1], padding='VALID')
        channels.append(tmp)

    tf_output_down = tf.concat(channels, axis=3)

    # loss
    loss = tf.reduce_mean(tf.abs(tf_output_down - tf_input))
    # loss_max = tf.reduce_max(tf.abs(tf_output_down - tf_input))
    # loss_min = tf.reduce_min(tf.abs(tf_output_down - tf_input))

    return loss


if __name__ == '__main__':
    from scipy.misc import imread, imsave, imresize
    # img_out = imread('test/lena_512x512.png', mode='RGB')
    img_out = np.zeros((32, 32, 3), dtype=np.uint8)
    img_out[8:16, 8:16, :] = 255
    out_shape = img_out.shape[:2]
    in_shape = [8, 8]
    img_in = imresize(img_out, in_shape, interp='bicubic').astype(np.float32) / 127.5 - 1
    img_out = img_out.astype(np.float32) / 127.5 - 1

    tf_in = tf.placeholder(dtype=tf.float32, shape=[None, in_shape[0], in_shape[1], 3])
    tf_out = tf.placeholder(dtype=tf.float32, shape=[None, out_shape[0], out_shape[1], 3])

    bp_loss = back_projection_loss(tf_input=tf_in, tf_output=tf_out)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        l_mean = sess.run(bp_loss, {tf_in: [img_in], tf_out: [img_out]})
        print(l_mean)
