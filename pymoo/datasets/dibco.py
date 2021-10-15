import numpy as np
import os
import tensorflow as tf
import tensorflow_io as tf_io

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

        self.loaded = False
        self.in_tr = None
        self.out_tr = None
        self.in_ts = None
        self.out_ts = None
        self.shape_tr = None
        self.shape_ts = None

    def to_image(self, data, is_normalized=False, reshape_to=None):
        if is_normalized:
            data = data * 255.0
        data = tf.cast(data, tf.uint8)
        data = tf.stack([data, data, data], axis=1)
        if reshape_to is not None:
            data = tf.reshape(data, tf.concat([reshape_to, [3]], axis=0))
        return data

    def save_image(self, filename, data, is_normalized=False, reshape_to=None):
        data = tf.constant(data)
        data = self.to_image(data, is_normalized, reshape_to)
        content = tf_io.image.encode_bmp(data)
        tf.io.write_file(filename, content)

    # ------------------------------------------------------------

    def load_sampled(self, grid_size=25, use_cache=True, add_padding=False):
        tr_file = "H06.bmp"
        in_tr = self.load_sampled_in_one(tr_file, grid_size, use_cache, add_padding)
        out_tr = self.load_out_one(tr_file, use_cache)
        out_tr = out_tr[int(grid_size/2):-1*(int(grid_size/2)+1), int(grid_size/2):-1*(int(grid_size/2)+1)] if not add_padding else out_tr # NOTE: The +1 is wrong, but dont want to resample it again
        shape_tr = out_tr.shape
        out_tr = tf.reshape(out_tr, [out_tr.shape[0]*out_tr.shape[1]])
        ts_file = "H07.bmp"
        in_ts = self.load_sampled_in_one(ts_file, grid_size, use_cache, add_padding)
        out_ts = self.load_out_one(ts_file, use_cache)
        out_ts = out_ts[int(grid_size/2):-1*(int(grid_size/2)+1), int(grid_size/2):-1*(int(grid_size/2)+1)] if not add_padding else out_ts # NOTE: The +1 is wrong, but dont want to resample it again
        shape_ts = out_ts.shape
        out_ts = tf.reshape(out_ts, [out_ts.shape[0] * out_ts.shape[1]])

        self.in_tr = in_tr
        self.out_tr = out_tr
        self.shape_tr = shape_tr
        self.in_ts = in_ts
        self.out_ts = out_ts
        self.shape_ts = shape_ts
        return (in_tr, out_tr), (in_ts, out_ts)

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
            shape_x = img.shape[0] - grid_size # + 1 # NOTE: this should be correct but i dont want to sample it again
            shape_y = img.shape[1] - grid_size # + 1 # NOTE: this should be correct but i dont want to sample it again
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
        return self.load_bmp('datasets/dibco2009/DIBC02009_Test_images-handwritten/H05.bmp')

    def load_out_first(self):
        return self.load_bmp('datasets/dibco2009/DIBCO2009-GT-Test-images_handwritten_bmp/H05.bmp')

    def load_bmp(self, filename):
        filename = check_filename(filename)
        if filename is None:
            return None
        bmp_file = tf.io.read_file(filename)
        image = tf.image.decode_bmp(bmp_file)
        return image[:, :, 0]

    def load_bmp_3d(self, filename):
        filename = check_filename(filename)
        if filename is None:
            return None
        bmp_file = tf.io.read_file(filename)
        image = tf.image.decode_bmp(bmp_file)
        return image

    def load_data(self):
        return self.load_sampled()

    # ------------------------------------


def crop_image(filein, fileout, h_offset, w_offset, height, width):
    # filein = '../datasets/dibco2009/DIBCO2009-GT-Test-images_handwritten_bmp/H05.bmp'
    # fileout = '../datasets/dibco2009/DIBCO2009-GT-Test-images_handwritten_bmp/H06.bmp'
    in5 = dataset.load_bmp_3d(filein)
    # in6 = tf.image.crop_to_bounding_box(in5, 70, 80, 600, 600)
    in6 = tf.image.crop_to_bounding_box(in5, h_offset, w_offset, height, width)
    content = tf_io.image.encode_bmp(in6)
    tf.io.write_file(fileout, content)


if __name__ == "__main__":
    dataset = DIBCO()
    # fin = '../../datasets/dibco2009/DIBC02009_Test_images-handwritten/H04.bmp'
    # fout = '../../datasets/dibco2009/DIBC02009_Test_images-handwritten/H07.bmp'
    # crop_image(fin, fout, 150, 0, 380, 1060)

    # in_tr, out_tr, in_ts, out_ts = dataset.load_sampled()

    #dataset.load_sampled(grid_size=3)
    print(5, flush=True)
    dataset.load_sampled(grid_size=5)
    print(7, flush=True)
    dataset.load_sampled(grid_size=7)
    #dataset.load_sampled(grid_size=9)
    print(11, flush=True)
    dataset.load_sampled(grid_size=11)
    #dataset.load_sampled(grid_size=13)
    print(15, flush=True)
    dataset.load_sampled(grid_size=15)
    #dataset.load_sampled(grid_size=17)
    #dataset.load_sampled(grid_size=19)
    print(21, flush=True)
    dataset.load_sampled(grid_size=21)
    #dataset.load_sampled(grid_size=23)

    print("end")
