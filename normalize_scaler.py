import numpy as np
from numpy.core.fromnumeric import mean


def handle_zeros_in_scale(scale):
    if np.isscalar(scale):
        if scale == .0:
            scale = 1.
        return scale
    elif isinstance(scale, np.ndarray):
        scale[scale == 0.0] = 1.0
        return scale


class NormalizeBase:

    def __init__(self, mean_=0, std_=1):
        self.mean_ = mean_
        self.std_ = std_
        self.scale_ = None
        self.fitted = False

    def _reset(self):
        if self.scale_:
            self.mean_ = 0
            self.std_ = 1
            self.scale_ = None
            self.fitted = False

    def fit(self):
        self._reset()
        self.scale_ = handle_zeros_in_scale(self.std_)
        self.fitted = True
        return self

    def transform(self, X):
        if self.fitted:
            return (X - self.mean_) / self.scale_
        else:
            raise Exception("Not fitted")


class HomoNormalize(NormalizeBase):

    def __init__(self, homo_mean, homo_std):
        super(HomoNormalize, self).__init__(mean_=homo_mean, std_=homo_std)

    def fit(self):
        super(HomoNormalize, self).fit()


class HeteroNormalize(NormalizeBase):

    def __init__(self, hetero_mean, hetero_std):

        super(HeteroNormalize, self).__init__(mean_=hetero_mean,
                                              std_=hetero_std)

    def fit(self):
        super(HeteroNormalize, self).fit()


class TestNormalize:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = np.array([[1., -1., 2.], [2., 0., 0.], [2, 1, 0], [0., 1., -1.]])
    scaler.fit(X_train)
    x_transform = scaler.transform(X_train)

    x1 = np.array([[1., -1., 2.], [2., 0., 0.]])
    x2 = np.array([[2, 1, 0], [0., 1., -1.]])
    n_samples = len(x1) + len(x2)
    x1_sum = np.sum(x1, axis=0)
    x2_sum = np.sum(x2, axis=0)

    homo_sum = x1_sum + x2_sum

    x1_ss = np.sum(np.array(x1)**2, axis=0)
    x2_ss = np.sum(np.array(x2)**2, axis=0)

    homo_ss = x1_ss + x2_ss
    homo_mean = homo_sum / n_samples
    homo_std = np.sqrt((homo_ss / n_samples - homo_mean**2))

    homo_scale = HomoNormalize(homo_mean, homo_std)
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

    hetero_scale = HeteroNormalize(np.mean(x3, axis=0), np.std(x3, axis=0))
    hetero_scale.fit()
    x3_transfrom = hetero_scale.transform(x3)

    np.testing.assert_array_almost_equal(x3_transfrom,
                                         x_transform[:, :2],
                                         decimal=6)

    hetero_scale = HeteroNormalize(np.mean(x4, axis=0), np.std(x4, axis=0))
    hetero_scale.fit()
    x4_transfrom = hetero_scale.transform(x4)

    np.testing.assert_array_almost_equal(x4_transfrom,
                                         x_transform[:, 2:],
                                         decimal=6)
