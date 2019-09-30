import warnings
from random import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from scipy.io import loadmat
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

MAX_ITERATIONS = 1000


def plot_3d(data):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(data.T[0], data.T[1], data.T[2], c=data.T[2], cmap='hsv')

    plt.show()


def plot_2d(data, color='blue', size=5):
    fig = go.Figure(data=go.Scattergl(
        x=data.T[0],
        y=data.T[1],
        mode='markers',
        marker=dict(
            size=size,
            color=color,
            colorscale='Viridis',
            line_width=1
        )
    ))

    fig.show()


def open_mat_data(file):
    return loadmat(file)['x']


def random_matrix(data, K):
    return np.array([np.random.dirichlet(np.ones(K), size=1)[0] for x in data])


def fuzzy_clusters_centers(data, U, m):
    return (data.T @ U ** m / np.sum(U ** m, axis=0)).T


def update_random_matrix(X, c, m):
    d = cdist(X, c) ** float(2 / (m - 1))
    return 1 / ((np.expand_dims(d, 2) / (np.expand_dims(d, 1).repeat(d.shape[-1], 1))).sum(2))


def cost_function(U, X, c, m):
    return np.sum((cdist(X, c) ** 2).T * (U.copy() ** m).T, 1).sum()


def fcm(K, X, m, err):
    U = random_matrix(X, K)
    aux_J = None
    for i in range(MAX_ITERATIONS):
        aux_U = U.copy()
        # print(str(i) + '\nCalculating centers')
        c = fuzzy_clusters_centers(X, U, m)
        # print(str(c) + '\nUpdanting Matrix')
        J = cost_function(U, X, c, m)
        U = update_random_matrix(X, c, m)
        # print(str(abs(U - aux_U).max()) + '\n')

        if aux_J is not None and abs(J - aux_J) < err:
            return U, c, i, J
        aux_J = J.copy()
    return U, c, i, J


def _init_cluster_center(X, K):
    return [X[np.random.randint(X.shape[0])] for i in range(K)]


def fkm(X, K, err):
    c = _init_cluster_center(X, K)
    aux_J = None
    for i in range(MAX_ITERATIONS):
        aux_c = c.copy()
        d = cdist(X, c) ** 2
        U = (d == d.min(axis=1)[:, None]).astype(int)
        J = np.diagonal(d.T @ U).sum()
        G = np.expand_dims(np.sum(U, axis=0), 2).repeat((X.T @ U).T.shape[-1], 1)
        c = (X.T @ U).T / G

        if aux_J is not None and abs(aux_J - J) < err:
            return U, c, i, J
        aux_J = J.copy()

    return U, c, i, J


def compare_fkm_fcm(data, K, err, m):
    file = open('output/output_1.txt', 'a+')
    for k in range(50):
        file = open('output/output_1.txt', 'a+')
        start_time = datetime.now()
        U, c, i, J = fcm(K, data, m, err)
        file.write(str(datetime.now()) + ',' + str('fcm') + ',' + str(K) + ',' + str(
            datetime.now() - start_time) + ',' + str(i) + ',' + str(J) + '\n')
        start_time = datetime.now()
        U, c, i, J = fkm(data, K, err)
        file.write(str(datetime.now()) + ',' + str('fkm') + ',' + str(K) + ',' + str(
            datetime.now() - start_time) + ',' + str(i) + ',' + str(J) + '\n')
        file.close()


def generate_img_matrix(U, c, img):
    data = [c[np.argmax(U[i])] for i in range(len(U))]
    data = np.array(data).astype(np.uint8)
    return data.reshape(img.height, img.width, 3)


def generate_img(images, clusters, m, err):
    for image, color in zip(images, clusters):
        file = open('output/output_2.txt', 'a+')
        print('Doing image: ' + image + '\n')
        img = Image.open(image)
        data = np.array((img.getdata()))
        K = color
        start_time = datetime.now()
        U, c, i = fcm(K, data, m, err)
        file.write(str(datetime.now()) + '\n' + str(image) + ',' + str(clusters) + ',' + str(
            datetime.now() - start_time) + ',' + str(i) + '\n')
        im = Image.fromarray(generate_img_matrix(U, c, img), img.mode)
        im.save('output/' + image)
        file.close()


def main():
    warnings.filterwarnings("ignore")
    images = ['photo001.jpg',
              'photo002.jpg',
              'photo003.jpg',
              'photo004.jpg',
              'photo005.jpg',
              'photo006.jpg',
              'photo007.jpg',
              'photo008.jpg',
              'photo009.jpg',
              'photo010.jpg',
              'photo011.png',
              ]
    clusters = [
        12,  # 1
        18,  # 2
        12,  # 3
        10,  # 4
        16,  # 5
        10,  # 6
        10,  # 7
        10,  # 8
        20,  # 9
        7,  # 10
        10  # 11
    ]

    data = open_mat_data('SyntheticDataset.mat')
    K = 4
    m = 2
    err = 1e-9

    compare_fkm_fcm(data, K, err, m)

    m = 2
    err = 1e-3
    generate_img(images, clusters, m, err)


if __name__ == '__main__':
    main()
