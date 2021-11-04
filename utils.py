import cv2
import os
import numpy as np
import pickle
import multiprocessing


class Extraction:
    def __init__(self, iimgs, label, features, num_cores):
        self.iimgs = iimgs
        self.label = label
        self.features = features
        self.num_cores = num_cores

        m = len(self.iimgs)
        n = len(self.features)

        sjs = self.split_jobs(m)
        pool = multiprocessing.Pool(processes=self.num_cores)
        self.data_output = pool.map(self.start, sjs)
        self.l_np = np.full((m, 1), label)

    def start(self, idx_range):
        dd = {}
        for i in range(idx_range[0], idx_range[1]):
            value = [f.evaluation(self.iimgs[i]) for f in self.features]
            dd[i] = value
        return dd

    def split_jobs(self, jobs):
        each = int(jobs/self.num_cores)
        more = int(jobs - (self.num_cores * each))
        idx_list = np.full((self.num_cores, ), each)
        for i in range(more):
            idx_list[i] += 1
        idx_list = np.cumsum(idx_list)
        s = 0
        split_jobs = []
        for i in idx_list:
            j = (s, i)
            split_jobs.append(j)
            s = i
        return split_jobs


def load_img(dir_name):
    """
    :param dir_name:
    :type dir_name: str

    :return: images in dir
    :rtype: list[numpy.array]
    """
    _, _, fnames = next(os.walk(dir_name))
    imgs = []
    for f in fnames:
        img = cv2.imread(dir_name+f, cv2.IMREAD_GRAYSCALE)
        img = img.astype('int32')
        imgs.append(img)
    return imgs


def load_img_float(dir_name):
    """
    :param dir_name:
    :type dir_name: str

    :return: images in dir
    :rtype: list[numpy.array]
    """
    _, _, fnames = next(os.walk(dir_name))
    imgs = []
    for f in fnames:
        img = cv2.imread(dir_name+f, cv2.IMREAD_GRAYSCALE)
        img = img/255.0
        imgs.append(img)
    return imgs


def integral_images(imgs, padding=False):
    """
    :param imgs:
    :type imgs: list[numpy.array]
    :param padding: padding at top and left
    :type padding: bool
    :rtype : list[numpy.array]
    """
    iimgs = []
    pad_size = ((1,0), (1,0))
    for i in imgs:
        ii = integral(i)
        if padding:
            ii = np.pad(ii, pad_size, 'constant', constant_values=(0))
        iimgs.append(ii)
    return iimgs


def integral(A):
    integral_A = np.cumsum(A, axis=1)
    integral_A = np.cumsum(integral_A, axis=0)
    return integral_A


def data_preprocessing_one(img, padding=False):
    pad_size = ((1,0), (1,0))
    ii = integral(img)
    if padding:
        ii = np.pad(ii, pad_size, 'constant', constant_values=(0))
    return ii


def data_extraction_w_integral(imgs, label, features, padding=False):
    iimgs = integral_images(imgs, padding=padding)
    m = len(iimgs)
    n = len(features)
    data = np.zeros((m, n), dtype=iimgs[0].dtype)
    for i, iimg in enumerate(iimgs) :
        value = np.array([f.evaluation(iimg) for f in features])
        data[i] = value
    l_np = np.full((m, 1), label)
    data = np.c_[data, l_np]
    return data


def data_extraction(iimgs, label, features):
    """
    :param iimgs: face of non-face integral images
    :type iimgs: list[numpy.array]
    :param label : class of iimgs
    :type label: int
    :param features: Haar-like features
    :type features: list[feature.HaarLikeFeature]

    :rtype numpy.array
    """
    m = len(iimgs)
    n = len(features)
    data = np.zeros((m, n), dtype=iimgs[0].dtype)
    for i, iimg in enumerate(iimgs):
        value = np.array([f.evaluation(iimg) for f in features])
        data[i] = value
    l_np = np.full((m, 1), label)
    data = np.c_[data, l_np]
    return data


def data_extraction_mp(iimgs, label, features, num_cores=4):
    ex = Extraction(iimgs, label, features, num_cores)
    m = len(iimgs)
    n = len(features)
    data = np.zeros((m, n), dtype=iimgs[0].dtype)

    for i in ex.data_output:
        for k in i.keys():
            data[k] = np.array(i[k])
    data = np.c_[data, ex.l_np]
    return data


def print_stronglearner(model):
    i = 0
    for clf in model.clfs:
        f = clf.feature
        print(f"{i}th classifier")
        print(f"\tfeature type = {f.featuretype}")
        print(f"\tposition in image = ({f.x}, {f.y})")
        print(f"\tfeature size =({f.h},{f.w})")
        i += 1
    print(f"alpha")
    print(model.alphas)


def print_features(fff):
    print(f"Feature - type : {fff.featuretype}, size : ({fff.h},{fff.w}), position : ({fff.x}, {fff.y})")


def save_cascade(cascaded, filename):
    with open(filename, 'wb') as f:
        pickle.dump(cascaded, f)
    print(f"Cascade model saved as {filename} successfully...")


def load_cascade(filename):
    with open(filename, 'rb') as f:
        cascade = pickle.load(f)
    print(f"Cascade model loaded from {filename}!")
    return cascade


def save_adaboost(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"AdaBoost saved as {filename} successfully...")


def load_adaboost(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f"AdaBoost loaded from {filename}!")
    return model
