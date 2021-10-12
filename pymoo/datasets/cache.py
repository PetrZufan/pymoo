import os
import pickle

from pymoo.datasets.util import check_filename


# TODO: Too coomplicated. content.p not needed. Checking the file existence is enough.
class Cache:
    cache_dir = ".cache"
    content_file = "content.p"

    def __init__(self, dataset_dir):
        self.dataset_dir = check_filename(dataset_dir, is_dir=True)
        if self.dataset_dir is None:
            raise Exception(str(dataset_dir) + " directory not found")
        self.cache_dir = os.path.join(self.dataset_dir, Cache.cache_dir)
        self.content_file = os.path.join(self.cache_dir, Cache.content_file)
        self._check_cache_files()

    def save(self, data, file):
        self._add_content(file)
        cache_file = os.path.join(self.cache_dir, file)
        pickle.dump(data, open(cache_file, "wb"))

    def load(self, file):
        cache_file = os.path.join(self.cache_dir, file)
        if self._check_content(file):
            return pickle.load(open(cache_file, "rb"))
        else:
            return None

    def clear(self):
        pickle.dump([], open(self.content_file, "wb"))

    def _check_cache_files(self):
        if not os.path.isdir(self.cache_dir):
            os.mkdir(self.cache_dir)
        if not os.path.isfile(self.content_file):
            pickle.dump([], open(self.content_file, "wb"))

    def _get_content(self):
        if not os.path.isfile(self.content_file):
            return []
        else:
            return pickle.load(open(self.content_file, "rb"))

    def _set_content(self, data):
        pickle.dump(data, open(self.content_file, "wb"))

    def _add_content(self, data):
        cache = self._get_content()
        if not self._check_content(data):
            cache.append(data)
            pickle.dump(cache, open(self.content_file, "wb"))

    def _remove_content(self, data):
        cache = self._get_content()
        keep = []
        for i, item in enumerate(cache):
            if item != data:
                keep.append(i)
        pickle.dump(cache[keep], open(self.content_file, "wb"))

    def _clear_content(self):
        pickle.dump([], open(self.content_file, "wb"))

    def _check_content(self, data):
        cache = self._get_content()
        for item in cache:
            if item == data:
                return True
        return False
