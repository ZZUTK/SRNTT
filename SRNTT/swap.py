import tensorflow as tf
import numpy as np
import os


class Swap(object):
    def __init__(self, patch_size=3, stride=1, sess=None):
        self.patch_size = patch_size
        self.stride = stride
        self.sess = sess
        self.content = None
        self.style = None
        self.condition = None
        self.conv_input = tf.placeholder(dtype=tf.float32, shape=[1, None, None, None], name='swap_input')
        self.conv_filter = tf.placeholder(dtype=tf.float32, shape=[self.patch_size, self.patch_size, None, None], name='swap_filter')
        self.conv = tf.nn.conv2d(
            input=self.conv_input,
            filter=self.conv_filter,
            strides=(1, self.stride, self.stride, 1),
            padding='VALID',
            name='feature_swap'
        )

    def style2patches(self, feature_map=None):
        """
        sample patches from the style (reference) map
        :param feature_map: array, [H, W, C]
        :return: array (conv kernel), [H, W, C_in, C_out]
        """
        if feature_map is None:
            feature_map = self.style
        h, w, c = feature_map.shape
        patches = []
        for ind_row in range(0, h - self.patch_size + 1, self.stride):
            for ind_col in range(0, w - self.patch_size + 1, self.stride):
                patches.append(feature_map[ind_row:ind_row+self.patch_size, ind_col:ind_col+self.patch_size, :])
        return np.stack(patches, axis=-1)

    def conditional_swap_multi_layer(self, content, style, condition, patch_size=3, stride=1, other_styles=None, is_weight=False):
        """
        feature swapping with multiple references on multiple feature layers
        :param content: array (h, w, c), feature map of content
        :param style: list of array [(h, w, c)], feature map of each reference
        :param condition: list of array [(h, w, c)], augmented feature map of each reference for matching with content map
        :param patch_size: int, size of matching patch
        :param stride: int, stride of sliding the patch
        :param other_styles: list (different layers) of lists (different references) of array (feature map),
                [[(h_, w_, c_)]], feature map of each reference from other layers
        :param is_weight, bool, whether compute weights
        :return: swapped feature maps - [3D array, ...], matching weights - 2D array, matching idx - 2D array
        """
        assert isinstance(content, np.ndarray)
        self.content = np.squeeze(content)
        assert len(self.content.shape) == 3

        assert isinstance(style, list)
        self.style = [np.squeeze(s) for s in style]
        assert all([len(self.style[i].shape) == 3 for i in range(len(self.style))])

        assert isinstance(condition, list)
        self.condition = [np.squeeze(c) for c in condition]
        assert all([len(self.condition[i].shape) == 3 for i in range(len(self.condition))])
        assert len(self.condition) == len(self.style)

        num_channels = self.content.shape[-1]
        assert all([self.style[i].shape[-1] == num_channels for i in range(len(self.style))])
        # assert all([self.condition[i].shape[-1] == num_channels for i in range(len(self.condition))])
        assert all([self.style[i].shape == self.condition[i].shape for i in range(len(self.style))])

        if other_styles is not None:
            assert isinstance(other_styles, list)
            assert all([isinstance(s, list) for s in other_styles])
            other_styles = [[np.squeeze(s) for s in styles] for styles in other_styles]
            assert all([all([len(s.shape) == 3 for s in styles]) for styles in other_styles])

        self.patch_size = patch_size
        self.stride = stride

        # split content, style, and condition into patches
        patches_content = self.style2patches(self.content)
        patches_style = np.concatenate(list(map(self.style2patches, self.style)), axis=-1)
        patches = np.concatenate(list(map(self.style2patches, self.condition)), axis=-1)

        # normalize content and condition patches
        norm = np.sqrt(np.sum(np.square(patches), axis=(0, 1, 2)))
        patches_style_normed = patches / norm
        norm = np.sqrt(np.sum(np.square(patches_content), axis=(0, 1, 2)))
        patches_content_normed = patches_content / norm
        del norm, patches, patches_content

        # match content and condition patches (batch-wise matching because of memory limitation)
        # the size of a batch is 512MB
        batch_size = int(1024. ** 2 * 512 / (self.content.shape[0] * self.content.shape[1]))
        num_out_channels = patches_style_normed.shape[-1]
        print('\tMatching ...')
        max_idx, max_val = None, None
        for idx in range(0, num_out_channels, batch_size):
            print('\t  Batch %02d/%02d' % (idx / batch_size + 1, np.ceil(1. * num_out_channels / batch_size)))
            batch = patches_style_normed[..., idx:idx + batch_size]
            if self.sess:
                corr = self.conv.eval({self.conv_input: [self.content], self.conv_filter: batch}, session=self.sess)
            else:
                corr = self.conv.eval({self.conv_input: [self.content], self.conv_filter: batch})

            corr = np.squeeze(corr)
            max_idx_tmp = np.argmax(corr, axis=-1) + idx
            max_val_tmp = np.max(corr, axis=-1)
            del corr, batch
            if max_idx is None:
                max_idx, max_val = max_idx_tmp, max_val_tmp
            else:
                indices = max_val_tmp > max_val
                max_val[indices] = max_val_tmp[indices]
                max_idx[indices] = max_idx_tmp[indices]

        # compute matching similarity (inner product)
        if is_weight:
            print('\tWeighting ...')
            corr2 = np.matmul(
                np.transpose(np.reshape(patches_content_normed, (-1, patches_content_normed.shape[-1]))),
                np.reshape(patches_style_normed, (-1, patches_style_normed.shape[-1]))
            )
            weights = np.reshape(np.max(corr2, axis=-1), max_idx.shape)
            del patches_content_normed, patches_style_normed, corr2
        else:
            weights = None
            del patches_content_normed, patches_style_normed

        # stitch matches style patches according to content spacial structure
        print('\tSwapping ...')
        maps = []
        target_map = np.zeros_like(self.content)
        count_map = np.zeros(shape=target_map.shape[:2])
        for i in range(max_idx.shape[0]):
            for j in range(max_idx.shape[1]):
                target_map[i:i + self.patch_size, j:j + self.patch_size, :] += patches_style[:, :, :, max_idx[i, j]]
                count_map[i:i + self.patch_size, j:j + self.patch_size] += 1.0
        target_map = np.transpose(target_map, axes=(2, 0, 1)) / count_map
        target_map = np.transpose(target_map, axes=(1, 2, 0))
        maps.append(target_map)

        # stitch other styles
        patch_size, stride = self.patch_size, self.stride
        if other_styles:
            for style in other_styles:
                ratio = float(style[0].shape[0]) / self.style[0].shape[0]
                assert int(ratio) == ratio
                ratio = int(ratio)
                self.patch_size = patch_size * ratio
                self.stride = stride * ratio
                patches_style = np.concatenate(list(map(self.style2patches, style)), axis=-1)
                target_map = np.zeros((self.content.shape[0] * ratio, self.content.shape[1] * ratio, style[0].shape[2]))
                count_map = np.zeros(shape=target_map.shape[:2])
                for i in range(max_idx.shape[0]):
                    for j in range(max_idx.shape[1]):
                        target_map[i*ratio:i*ratio + self.patch_size, j*ratio:j*ratio + self.patch_size, :] += patches_style[:, :, :, max_idx[i, j]]
                        count_map[i*ratio:i*ratio + self.patch_size, j*ratio:j*ratio + self.patch_size] += 1.0
                target_map = np.transpose(target_map, axes=(2, 0, 1)) / count_map
                target_map = np.transpose(target_map, axes=(1, 2, 0))
                maps.append(target_map)

        return maps, weights, max_idx
