
class HaarFeature:
    def __init__(self, featuretype, pos_x, pos_y, width, height):
        self.featuretype = featuretype  # (h,w)
        self.x = pos_x
        self.y = pos_y
        self.w = width
        self.h = height

    def evaluation(self, iimg):
        """ calculate feature value. i.e. white region - gray region

        :param iimg:
        :type iimg:
        :return:
        :rtype:
        """
        region_h = int(self.h / self.featuretype[0])
        region_w = int(self.w / self.featuretype[1])
        white, grey = 0, 0
        if self.featuretype == (1, 2):
            white = self.regionsum(iimg, self.x, self.y, region_h, region_w)
            grey = self.regionsum(iimg, self.x, self.y + region_w, region_h, region_w)
        elif self.featuretype == (2, 1):
            white = self.regionsum(iimg, self.x, self.y, region_h, region_w)
            grey = self.regionsum(iimg, self.x + region_h, self.y, region_h, region_w)
            pass
        elif self.featuretype == (3, 1):
            white = self.regionsum(iimg, self.x, self.y, region_h, region_w) \
                    + self.regionsum(iimg, self.x + (2 * region_h), self.y, region_h,  region_w)
            grey = self.regionsum(iimg, self.x + region_h, self.y, region_h, region_w)
            pass
        elif self.featuretype == (1, 3):
            white = self.regionsum(iimg, self.x, self.y, region_h, region_w) \
                    + self.regionsum(iimg, self.x, self.y + (2 * region_w), region_h, region_w)
            grey = self.regionsum(iimg, self.x, self.y + region_w, region_h, region_w)
            pass
        elif self.featuretype == (2, 2):
            white = self.regionsum(iimg, self.x, self.y, region_h, region_w) \
                    + self.regionsum(iimg, self.x + region_h, self.y + region_w, region_h, region_w)
            grey = self.regionsum(iimg, self.x + region_h, self.y, region_h, region_w) \
                   + self.regionsum(iimg, self.x, self.y + region_w, region_h, region_w)

        return white - grey

    def regionsum(self, iimg, x, y, h, w):
        return iimg[x, y] + iimg[x + h, y + w] - iimg[x + h, y] - iimg[x, y + w]


def create_features(img_size, FeatureTypes, min_f_size, max_f_size):
    """ Create Haar-like features

    :param img_size: train image size (height, width)
    :type img_size: iterable
    :param FeatureTypes: feature type
    :type FeatureTypes:
    :param min_f_size: (inclusive) minimum size of feature, (min_height, min_width)
    :type min_f_size: iterable
    :param max_f_size: (inclusive) maximum size of feature, (max_height, max_height)
    :type max_f_size: iterable

    :rtype list[Feature.HaarLikeFeature]
    """
    features = []
    f_types = [(2,1), (1,2), (3,1), (1,3), (2,2)]
    #f_types = FeatureType[FeatureTypes]
    for ft in f_types:
        min_h = max(min_f_size[0], ft[0])
        min_w = max(min_f_size[1], ft[1])
        max_h = max_f_size[0]
        max_w = max_f_size[1]
        # feature size
        for w in range(min_w, max_w+1, ft[1]):
            for h in range(min_h, max_h+1, ft[0]):
                # position of feature
                for y in range(img_size[1]-w+1):
                    for x in range(img_size[0]-h+1):
                        ff = HaarFeature(ft, x, y, w, h)
                        features.append(ff)
    return features

'''
features = create_features((19,19), "basic", (1,1), (19,19))
print(len(features))
'''
