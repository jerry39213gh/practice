import numpy as np
from sklearn.datasets import make_classification as mc
from scipy.spatial.distance import cosine
from sklearn.metrics import precision_score, accuracy_score, recall_score
from collections import Counter


class Knn(object):
    def __init__(self, k=3, distance='euclidean'):
        choice_dict = {'euclidean': self._euclidean_distance,
                       'cosine': self._cosine_distance}
        self.k = k
        self.distance = choice_dict[distance]

    def _euclidean_distance(self, x, y):
        return (sum([(i - j) ** 2 for i, j in zip(x, y)])) ** 0.5

    def _cosine_distance(self, x, y):
        return cosine(x, y)

    def fit(self, x, y):
        self.x = x
        self.y = y

    def predict(self, p):
        prediction = []
        for dta in p:
            c = Counter(self.y[np.argsort([self.distance(ref, dta) for ref in self.x])[:self.k]])
            prediction.append(c.most_common()[0][0])
        return np.array(prediction)

    def score(self, a, b):
        pred = self.predict(a)
        precision = precision_score(pred, b)
        accuracy = accuracy_score(pred, b)
        recall = recall_score(pred, b)
        return (precision, accuracy, recall)

for i in [5, 10, 20, 50]:
    model = Knn(k = i)
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 15

# import some data to play with
iris = datasets.load_iris()

# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = Knn(k=15)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.show()
