import tensorflow as tf
import numpy as np
from scipy.io import loadmat


class VGG19(object):
    def __init__(self, input_image=None, model_path='models/VGG19/imagenet-vgg-verydeep-19.mat', final_layer='relu5_1'):
        if isinstance(input_image, np.ndarray):
            assert len(input_image.shape) == 3
            self.input = tf.constant(np.expand_dims(input_image, axis=0), dtype=tf.float32)
        elif isinstance(input_image, (tf.Tensor, tf.Variable)):
            assert len(input_image.get_shape()) == 4
            self.input = input_image
        else:
            self.input = tf.placeholder(dtype=tf.float32, shape=[None] * 4)
        self.model_path = model_path
        self.mean_pixel = (123.68, 116.779, 103.939)  # RGB
        self.layer_names = (
                'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

                'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

                'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2',
                'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

                'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2',
                'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

                'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2',
                'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
            )
        self.vgg_input = self.input - self.mean_pixel
        self.final_layer = final_layer
        # self.layers = self.vgg19()
        try:
            self.layers = self.vgg19()
        except:
            self.layers = self.vgg19(reuse=True)

    def vgg19(self, reuse=False):
        params = loadmat(self.model_path)
        # mean = params['normalization'][0][0][0]
        # self.mean_pixel = np.mean(mean, axis=(0, 1))
        weights = params['layers'][0]
        net = {}
        current = self.vgg_input
        with tf.variable_scope('vgg19') as scope:
            if reuse:
                scope.reuse_variables()
            for i, name in enumerate(self.layer_names):
                kind = name[:4]
                if kind == 'conv':
                    try:
                        kernels, bias = weights[i][0][0][0][0]
                    except:
                        kernels, bias = weights[i][0][0][2][0]

                    # matconvnet: weights are [width, height, in_channels, out_channels]
                    # tensorflow: weights are [height, width, in_channels, out_channels]
                    kernels = tf.get_variable(
                        name=name+'/w',
                        initializer=tf.constant(np.transpose(kernels, (1, 0, 2, 3))),
                        trainable=False
                    )
                    bias = tf.get_variable(
                        name=name+'/b',
                        initializer=tf.constant(np.squeeze(bias).reshape(-1)),
                        trainable=False
                    )
                    current = tf.nn.conv2d(current, kernels, strides=(1, 1, 1, 1), padding='SAME')
                    current = tf.nn.bias_add(current, bias)
                elif kind == 'relu':
                    current = tf.nn.relu(current)
                elif kind == 'pool':
                    current = tf.nn.max_pool(current, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
                net[name] = current
                # print('\t%s' % name)
                if name == self.final_layer:
                    break

        # assert len(net) == len(self.layer_names)
        return net

    def get_layer_output(self, sess, feed_image=None, layer_name='relu3_1'):
        fetches = []
        if isinstance(layer_name, str):
            assert layer_name in self.layer_names
            fetches = self.layers[layer_name]
        elif isinstance(layer_name, (list, tuple)):
            for item in layer_name:
                assert item in self.layer_names
                fetches.append(self.layers[item])

        try:
            return sess.run(fetches)
        except:
            if feed_image is None:
                print('Require an input image!')
                exit()
            else:
                assert isinstance(feed_image, np.ndarray)
                if len(feed_image.shape) == 3:
                    return sess.run(fetches, {self.input: [feed_image]})
                else:
                    return sess.run(fetches, {self.input: feed_image})

    def image_from_vgg_input(self, image):
        assert isinstance(image, np.ndarray)
        image = np.squeeze(image)
        assert len(image.shape) == 3
        return (image + self.mean_pixel).astype(np.uint8)





if __name__ =='__main__':
    net = VGG19()
    ne2 = VGG19()
    print(net.layers[net.layer_names[-1]].get_shape())
