import numpy as np


class OneHotEncoderBase():

    def __init__(self, drop=False):
        self.drop = drop
        self.categories_ = []
        self.n_features = None
        self.fitted = False

    def fit(self, x):
        x = np.array(x)
        self.n_features = x.shape[1]
        for i in range(self.n_features):
            xi = x[:, i]
            cats = np.sort(list(set(xi)))
            self.categories_.append(cats)
        self.fitted = True
        return self

    def transform(self, x):
        if not self.fitted:
            raise Exception("Not fitted")
        else:
            x = np.array(x)
            result = self._encode(x[:, 0], self.categories_[0])
            for i in range(1, self.n_features):
                result = np.hstack(
                    (result, self._encode(x[:, i], self.categories_[i])))
        return result

    def inverse_transform(self, x_encode):
        x_encode = np.array(x_encode)
        x_decode = np.empty(len(x_encode),
                            dtype=self.categories_[0].dtype).reshape([-1, 1])
        for i in range(1, len(self.categories_)):
            x_decode = np.hstack(
                (x_decode,
                 np.empty(len(x_encode),
                          dtype=self.categories_[i].dtype).reshape([-1, 1])))
        col_index = 0
        for i in range(self.n_features):
            if self.drop:
                if len(self.categories_[i]) > 1:
                    x_decode[:, i] = self._decode(
                        x_encode[:, col_index:col_index +
                                 len(self.categories_[i]) - 1],
                        self.categories_[i])
                else:
                    x_decode[:, i] = self._decode(-np.ones([len(x_encode), 1]),
                                                  self.categories_[i])
                col_index += len(self.categories_[i]) - 1
            else:
                x_decode[:, i] = self._decode(
                    x_encode[:, col_index:col_index + len(self.categories_[i])],
                    self.categories_[i])
                col_index += len(self.categories_[i])
        return x_decode

    def get_feature_names(self, col_names=None):
        if self.fitted:
            feature_names = []
            if self.drop:
                for i in range(self.n_features):
                    if len(self.categories_[i]) > 1:
                        feature_names.append(list(self.categories_[i][1:]))
            else:
                for i in range(self.n_features):
                    feature_names.append(list(self.categories_[i]))
        else:
            raise Exception("Not fitted")
        if col_names:
            for i in range(len(feature_names)):
                for j in range(len(feature_names[i])):
                    feature_names[i][
                        j] = col_names[i] + '_' + feature_names[i][j]
        return feature_names

    def _encode(self, xi, uniques):
        if self.drop:
            encode = np.zeros((len(xi), len(uniques) - 1))
            for i in range(len(xi)):
                for j in range(1, len(uniques)):
                    if xi[i] == uniques[j]:
                        encode[i][j - 1] = 1
        else:
            encode = np.zeros((len(xi), len(uniques)))
            for i in range(len(xi)):
                for j in range(len(uniques)):
                    if xi[i] == uniques[j]:
                        encode[i][j] = 1
        return encode

    def _decode(self, xi_encode, uniques):
        xi_decode = np.empty(len(xi_encode), dtype=uniques.dtype)
        if self.drop:
            if np.sum(xi_encode[0]) < 0:
                for i in range(len(xi_decode)):
                    xi_decode[i] = uniques[0]
            else:
                for i in range(len(xi_decode)):
                    if np.sum(xi_encode[i]) == 0:
                        xi_decode[i] = uniques[0]
                    else:
                        for j in range(len(xi_encode[i])):
                            if xi_encode[i][j] == 1:
                                xi_decode[i] = uniques[j + 1]
                                break

        else:
            for i in range(len(xi_decode)):
                for j in range(len(xi_encode[i])):
                    if xi_encode[i][j] == 1:
                        xi_decode[i] = uniques[j]
        return xi_decode


class HomoOneHot(OneHotEncoderBase):

    def __init__(self, drop=False):
        super(HomoOneHot, self).__init__(drop=drop)

    def fit(self, categories1, categories2):
        self.n_features = len(categories1)
        for i in range(self.n_features):
            categories = np.sort(
                list(set(np.concatenate((categories1[i], categories2[i])))))
            self.categories_.append(categories)
        self.fitted = True
        return self


class HeteroOneHot(OneHotEncoderBase):

    def __init__(self, drop=False):
        super(HeteroOneHot, self).__init__(drop=drop)

    def fit(self, x):
        super(HeteroOneHot, self).fit(x)


if __name__ == '__main__':
    X = [['男', '本科', '北京', '否'], ['女', '专科', '江西', '否'],
         ['双性', '硕士', '福建', '否']]
    onehot = OneHotEncoderBase(drop=True)
    onehot.fit(X)
    X_encode = onehot.transform(X)
    print(X_encode)
    print(onehot.get_feature_names(col_names=['性别', '学历', '籍贯', '婚姻']))
    #print(onehot.inverse_transform(X_encode))
    X1 = [['男', '本科', '北京', '否'], ['女', '专科', '江西', '否']]
    X2 = [['双性', '硕士', '福建', '否']]
    homoonehot = HomoOneHot(drop=True)
    homoonehot.fit([['男', '女'], ['本科', '专科'], ['北京', '江西'], ['否']],
                   [['双性'], ['硕士'], ['福建'], ['否']])
    X1_encode = homoonehot.transform(X1)
    X2_encode = homoonehot.transform(X2)
    print(X1_encode)
    print(X2_encode)
    print(onehot.inverse_transform(X_encode))
    print(homoonehot.inverse_transform(X1_encode))
    print(homoonehot.inverse_transform(X2_encode))
    np.testing.assert_array_equal(X_encode, np.vstack((X1_encode, X2_encode)))
    np.testing.assert_array_equal(
        onehot.inverse_transform(X_encode),
        np.vstack((homoonehot.inverse_transform(X1_encode),
                   homoonehot.inverse_transform(X2_encode))))