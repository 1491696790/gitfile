import numpy as np


def handle_zeros_in_scale(scale):
    if np.isscalar(scale):
        if scale == .0:
            scale = 1.
        return scale
    elif isinstance(scale, np.ndarray):
        scale[scale == 0.0] = 1.0
        return scale


class MinMaxBase:

    def __init__(self, feature_range=(0, 1), min_=0, max_=1):
        self.feature_range = feature_range
        self.min = min_
        self.max = max_
        self.scale = None
        self.fitted = False

    def _reset(self):
        if self.scale:
            self.min = 0
            self.max = 1
            self.scale = None
            self.fitted = False

    def fit(self):
        self._reset()
        feature_range = self.feature_range
        if feature_range[0] >= feature_range[1]:
            raise ValueError("Minimum of desired feature range must be smaller"
                             " than maximum. Got %s." % str(feature_range))
        max_min = self.max - self.min
        self.scale = ((feature_range[1] - feature_range[0]) /
                      handle_zeros_in_scale(max_min))

        self.scale_min = feature_range[0] - self.min * self.scale
        self.fitted = True
        return self

    def transform(self, X):
        if self.fitted:
            X *= self.scale
            X += self.scale_min
            return X
        else:
            raise Exception("Not fitted")


class HomoMinMax(MinMaxBase):

    def __init__(self, min_li, max_li):

        super(HomoMinMax, self).__init__(min_=np.min(min_li, axis=0),
                                         max_=np.max(max_li, axis=0))

    def fit(self):
        super(HomoMinMax, self).fit()


class HeteroMinMax(MinMaxBase):

    def __init__(self, min_, max_):

        super(HeteroMinMax, self).__init__(min_=min_, max_=max_)

    def fit(self):
        super(HeteroMinMax, self).fit()


class TestMinMax:
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train = np.array([[1., -1., 2.], [2., 0., 0.], [2, 1, 0], [0., 1., -1.]])
    scaler.fit(X_train)
    x_transform = scaler.transform(X_train)

    x1 = np.array([[1., -1., 2.], [2., 0., 0.]])
    x2 = np.array([[2, 1, 0], [0., 1., -1.]])

    x1_min = np.min(x1, axis=0)
    x2_min = np.min(x2, axis=0)

    x1_max = np.max(x1, axis=0)
    x2_max = np.max(x2, axis=0)

    min_li = np.vstack([x1_min, x2_min])
    max_li = np.vstack([x1_max, x2_max])

    homo_scale = HomoMinMax(min_li, max_li)
    homo_scale.fit()
    x1_transform = homo_scale.transform(x1)
    x2_transform = homo_scale.transform(x2)

    np.testing.assert_array_almost_equal(x1_transform,
                                         x_transform[:2,],
                                         decimal=6)
    np.testing.assert_array_almost_equal(x2_transform,
                                         x_transform[2:,],
                                         decimal=6)

    x3 = X_train[:, :2]
    x4 = X_train[:, 2:]

    hetero_scale = HeteroMinMax(np.min(x3, axis=0), np.max(x3, axis=0))
    hetero_scale.fit()
    x3_transfrom = hetero_scale.transform(x3)

    np.testing.assert_array_almost_equal(x3_transfrom,
                                         x_transform[:, :2],
                                         decimal=6)

    hetero_scale = HeteroMinMax(np.min(x4, axis=0), np.max(x4, axis=0))
    hetero_scale.fit()
    x4_transfrom = hetero_scale.transform(x4)

    np.testing.assert_array_almost_equal(x4_transfrom,
                                         x_transform[:, 2:],
                                         decimal=6)
