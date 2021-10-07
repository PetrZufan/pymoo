import numpy as np
import os
import tensorflow as tf

import sys
sys.path.insert(0, "/home/zufan/git/pymoo/")

from pymoo.datasets.util import check_filename
from pymoo.datasets.cache import Cache


class DIBCO:

    _directory_dataset = 'datasets/dibco2009'
    _directory_in = _directory_dataset + '/DIBC02009_Test_images-handwritten'
    _directory_out = _directory_dataset + '/DIBCO2009-GT-Test-images_handwritten_bmp'

    def __init__(self):
        self._directory_dataset = check_filename(DIBCO._directory_dataset, is_dir=True)
        self._directory_in = check_filename(DIBCO._directory_in, is_dir=True)
        self._directory_out = check_filename(DIBCO._directory_out, is_dir=True)
        self._cache = Cache(self._directory_dataset)

    def load_sampled(self, grid_size=25, use_cache=True, add_padding=False):
        tr_file = "H05.bmp"
        in_tr = self.load_sampled_in_one(tr_file, grid_size, use_cache, add_padding)
        out_tr = self.load_sampled_out_one(tr_file, grid_size, use_cache, add_padding)
        ts_file = "H04.bmp"
        in_ts = self.load_sampled_in_one(ts_file, grid_size, use_cache, add_padding)
        out_ts = self.load_sampled_out_one(ts_file, grid_size, use_cache, add_padding)
        return in_tr, out_tr, in_ts, out_ts

    def load_sampled_in_first(self, count=1, grid_size=25, use_cache=True, add_padding=False):
        images = []
        i = 0
        for file in os.listdir(self._directory_in):
            image = self.load_sampled_in_one(file, grid_size, use_cache, add_padding)
            images.append(image)
            i += 1
            if i >= count:
                break
        return images

    def load_sampled_out_first(self, count=1, grid_size=25, use_cache=True, add_padding=False):
        images = []
        i = 0
        for file in os.listdir(self._directory_in):
            image = self.load_sampled_out_one(file, grid_size, use_cache, add_padding)
            images.append(image)
            i += 1
            if i >= count:
                break
        return images

    def load_sampled_in_all(self, grid_size=25, use_cache=True, add_padding=False):
        images = []
        for file in os.listdir(self._directory_in):
            image = self.load_sampled_in_one(file, grid_size, use_cache, add_padding)
            images.append(image)
        return images

    def load_sampled_out_all(self, grid_size=25, use_cache=True, add_padding=False):
        images = []
        for file in os.listdir(self._directory_out):
            image = self.load_sampled_out_one(file, grid_size, use_cache, add_padding)
            images.append(image)
        return images

    def load_sampled_in_one(self, file, grid_size=25, use_cache=True, add_padding=False):
        return self.load_sampled_one(file, "in", grid_size, use_cache, add_padding)

    def load_sampled_out_one(self, file, grid_size=25, use_cache=True, add_padding=False):
        return self.load_sampled_one(file, "out", grid_size, use_cache, add_padding)

    def load_sampled_one(self, file, inout, grid_size=25, use_cache=True, add_padding=False):
        postfix = "_s" + str(grid_size)
        if add_padding:
            postfix = postfix + "p"
        if inout == "in":
            postfix = postfix + "_in.p"
        else:
            postfix = postfix + "_out.p"

        cache_file = file + postfix
        if use_cache:
            sampled = self._cache.load(cache_file)
            if sampled is not None:
                return sampled

        image = self.load_one(file, inout, use_cache)
        sampled = self.sample(image, grid_size, add_padding)

        self._cache.save(sampled, cache_file)
        return sampled

    def sample(self, img, grid_size=25, add_padding=False):
        padding = int(grid_size / 2)
        if add_padding:
            shape_x = img.shape[0]
            shape_y = img.shape[1]
            pad = tf.constant([[padding, padding], [padding, padding]])
            final_img = tf.pad(img, pad, 'CONSTANT', constant_values=0)
        else:
            shape_x = img.shape[0] - grid_size
            shape_y = img.shape[1] - grid_size
            final_img = img

        i, j = tf.meshgrid(np.arange(shape_x), np.arange(shape_y), indexing="ij")
        i = tf.reshape(i, [i.shape[0] * i.shape[1]])
        j = tf.reshape(j, [j.shape[0] * j.shape[1]])
        samples = tf.map_fn(
            lambda x: final_img[x[0]:x[0] + grid_size, x[1]:x[1] + grid_size],
            tf.stack([i, j], axis=1),
            dtype=tf.uint8
        )
        return samples

    # -----------------------------------

    def load_in_all(self, use_cache=True):
        images = []
        for file in os.listdir(self._directory_in):
            image = self.load_in_one(file, use_cache)
            images.append(image)
        return images

    def load_out_all(self, use_cache=True):
        images = []
        for file in os.listdir(self._directory_out):
            image = self.load_out_one(file, use_cache)
            images.append(image)
        return images

    def load_in_one(self, file, use_cache=True):
        return self.load_one(file, "in", use_cache)

    def load_out_one(self, file, use_cache=True):
        return self.load_one(file, "out", use_cache)

    def load_one(self, file, inout, use_cache=True):
        if inout == "in":
            postfix = "_in.p"
            directory = self._directory_in
        else:
            postfix = "_out.p"
            directory = self._directory_out

        cache_file = file + postfix
        if use_cache:
            image = self._cache.load(cache_file)
            if image is not None:
                return image

        file_path = os.path.join(directory, file)
        image = self.load_bmp(file_path)

        self._cache.save(image, cache_file)
        return image

    # --------------------------------------

    def load_in_first(self):
        return self.load_bmp('datasets/dibco2009/DIBC02009_Test_images-handwritten/H01.bmp')

    def load_out_first(self):
        return self.load_bmp('datasets/dibco2009/DIBCO2009-GT-Test-images_handwritten_bmp/H01.bmp')

    def load_bmp(self, filename):
        filename = check_filename(filename)
        if filename is None:
            return None
        bmp_file = tf.io.read_file(filename)
        image = tf.image.decode_bmp(bmp_file)
        return image[:, :, 0]

    # ------------------------------------


if __name__ == "__main__":
    dataset = DIBCO()
    # in_tr, out_tr, in_ts, out_ts = dataset.load_sampled()
    in_tr = dataset.load_sampled_out_one("H05.bmp")
    print("end")
