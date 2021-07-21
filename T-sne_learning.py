from time import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets, manifold, decomposition, discriminant_analysis
from matplotlib import offsetbox

digits = datasets.load_digits(n_class=10)
X = digits.data
y = digits.target
n_samples, n_features = X.shape
n_neighbors = 30


def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0],
                 X[i, 1],
                 str(digits.target[i]),
                 color=plt.cm.Set1(y[i] / 10),
                 fontdict={
                     'weight': 'bold',
                     'size': 9
                 })
    '''
    #不知道在装什么B
    if hasattr(offsetbox, 'AnnotationBbox'):  #(用于判断对象是否包含对应的属性)
        shown_images = np.array([[1., 1.]])
    for i in range(X.shape[0]):
        dist = np.sum((X[i] - shown_images)**2, 1)
        if np.min(dist) < 4e-3:
            continue
        shown_images = np.r_[shown_images, [X[i]]]  #按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r), X[i])
    ax.add_artist(imagebox)
    '''
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=2, init='random', random_state=0)
t0 = time()
X_tsne = tsne.fit_transform(X)
plot_embedding(X_tsne, 't-SNE embedding(time %.2fs)' % (time() - t0))
plt.show()