import numpy
import numpy as np
import utils
import math


class WeakLeaner:
    def __init__(self, feature, threshold, p):
        self.feature = feature
        self.threshold = threshold
        self.p = p

    def predict(self, iimg):
        value = self.feature.evaluation(iimg)
        p = 1 if value*self.p < self.threshold*self.p else 0
        return p


class AdaBoost:
    """
    clfs : list[boosting.WeakLearner]
    alphas : list[float]
    """
    def __init__(self, num_classifier=3):
        self.num_classifier = num_classifier
        self.clfs = None
        self.alphas = None
        self.th = None

    def train(self, face_iimgs, non_iimgs, features, isExtracted=True, verbose=True):
        """
        :param face_iimgs: if extracted, face_iimgs's type is numpy.array else list[numpy.array]
        :type face_iimgs: list[numpy.array], numpy.array
        :param non_iimgs: if extracted, non_iimgs's type is numpy.array else list[numpy.array]
        :type non_iimgs: list[numpy.array],  numpy.array
        :param features: list of haar-like features
        :type features: list[feature.HaarLikeFeature]
        :param isExtracted: if face_iimgs and non_iimgs is already extracted, True else False
        :type isExtracted: bool
        """
        num_face = len(face_iimgs)
        num_non = len(non_iimgs)
        if isExtracted:
            full_data = np.r_[face_iimgs, non_iimgs]
        else:
            face_data = utils.data_extraction(face_iimgs, 1, features)
            non_data = utils.data_extraction(non_iimgs, 0, features)
            full_data = np.r_[face_data, non_data]

        X = full_data[:, :-1]
        y = full_data[:, -1]

        features_idx = list(range(len(features)))
        weights = [1/(2*num_face) if yy == 1 else 1/(2*num_non) for yy in y]
        self.clfs = []
        self.alphas = []
        for t in range(self.num_classifier):
            # normalize weights
            weights = [w/sum(weights) for w in weights]
            best_error, best_feature_id, best_th, best_p = float('inf'), None, 0, 0

            total_face_w = sum([w for w, yy in zip(weights, y) if yy == 1])
            total_non_w = sum([w for w, yy in zip(weights, y) if yy == 0])
            for j in features_idx:  # j is index of feature
                feature_j = X[:, j]
                sort_j = sorted(zip(feature_j, weights, y), key=lambda x: x[0])
                local_error, local_th, local_p = float('inf'), 0, 0
                face_below, non_below = 0, 0
                below_face_w, below_non_w = 0, 0
                for i, w, yy in sort_j:
                    e_i = min(below_face_w + total_non_w - below_non_w, below_non_w + total_face_w - below_face_w)
                    if e_i < local_error:
                        local_error = e_i
                        local_th = i
                        local_p = 1 if face_below > non_below else -1

                    if yy == 1:
                        face_below += 1
                        below_face_w += w
                    else:
                        non_below += 1
                        below_non_w += w
                # end for-loop sort_j
                if local_error < best_error:
                    best_error = local_error
                    best_feature_id = j
                    best_th = local_th
                    best_p = local_p
            features_idx.remove(best_feature_id)
            # end for-loop feature idx
            if verbose :
                print("Feature ID : ",best_feature_id)
                print("Error :",best_error)

            beta = best_error/(1-best_error)
            alpha = math.log(1/beta)
            weak = WeakLeaner(features[best_feature_id], best_th, best_p)
            self.alphas.append(alpha)
            self.clfs.append(weak)
            # update weights
            acc = accuracy_partial(X, y, best_feature_id, best_th, best_p)
            weights = [w * math.pow(beta, 1-a) for w, a in zip(weights, acc)]
        # end for-loop num_classifier
        self.th = 0.5 * sum(self.alphas)

    def decrese_th(self, value):
        self.th = self.th - value

    def predict(self, iimgs):
        """
        :param iimgs: integral images
        :type iimgs: list[numpy.array]
        :return: prediction of strong learner
        :rtype: list[int]
        """
        pred = []
        alp = np.array(self.alphas)
        for iimg in iimgs :
            weak_pred = np.array([clf.predict(iimg) for clf in self.clfs])
            p = 1 if sum(weak_pred * alp) >= self.th else 0
            pred.append(p)
        return pred

    def predict_single(self, iimg):
        alp = np.array(self.alphas)
        weak_pred = np.array([clf.predict(iimg) for clf in self.clfs])
        p = 1 if sum(weak_pred * alp) >= self.th else 0
        return p


def accuracy_partial(X, y, feature_id, threshold, p):
    values = X[:, feature_id]
    pred = [1 if v*p < threshold*p else 0 for v in values]
    acc = [0 if yy == p else 1 for yy, p in zip(y, pred)]
    return acc
