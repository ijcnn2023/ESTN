import torch
import ssl
import os
import cv2
import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.datasets import fetch_openml

from torchvision import datasets, transforms

ssl._create_default_https_context = ssl._create_unverified_context



def dsphere(n=100, d=2, r=1, noise=None, ambient=None):
    """
    Sample `n` data points on a d-sphere.

    Parameters
    -----------
    n : int
        Number of data points in shape.
    r : float
        Radius of sphere.
    ambient : int, default=None
        Embed the sphere into a space with ambient dimension equal to `ambient`. The sphere is randomly rotated in this high dimensional space.
    """
    data = np.random.randn(n, d+1)

    # Normalize points to the sphere
    data = r * data / np.sqrt(np.sum(data**2, 1)[:, None])

    if noise:
        data += noise * np.random.randn(*data.shape)

    if ambient:
        assert ambient > d, "Must embed in higher dimensions"
        data = embed(data, ambient)

    return data


def create_sphere_dataset(n_samples=500, d=100, n_spheres=11,
                          r=5, plot=False, seed=42, bigR=None):
    """ from MLDL dataset.py, 11 spheres with 5500 samples """
    np.random.seed(seed)

    variance = 10 / np.sqrt(d)

    shift_matrix = np.random.normal(0, variance, [n_spheres, d+1])

    spheres = []
    n_datapoints = 0
    for i in np.arange(n_spheres-1):
        sphere = dsphere(n=n_samples, d=d, r=r)
        spheres.append(sphere + shift_matrix[i, :])
        n_datapoints += n_samples

    # Additional big surrounding sphere:
    n_samples_big = 1 * n_samples
    big = dsphere(n=n_samples_big, d=d, r=bigR)
    spheres.append(big)
    n_datapoints += n_samples_big

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, n_spheres))
        for data, color in zip(spheres, colors):
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=[color])
        plt.savefig('sphere.png')

    # Create Dataset:
    dataset = np.concatenate(spheres, axis=0)

    labels = np.zeros(n_datapoints)
    label_index = 0
    for index, data in enumerate(spheres):
        n_sphere_samples = data.shape[0]
        labels[label_index:label_index + n_sphere_samples] = index
        label_index += n_sphere_samples

    # index_seed = np.linspace(0, n_samples*20, num=20, dtype='int16', endpoint=False)
    index_seed = np.linspace(0, n_samples*11, num=11, dtype='int16', endpoint=False)
    arr = np.array([], dtype='int16')
    for i in range(n_samples):
        arr = np.concatenate((arr, index_seed+int(i)))

    dataset = dataset[arr]
    labels = labels[arr]
    print("sphere dataset shape={}".format(dataset.shape))

    return dataset / 22 + 0.5, labels


def LoadRealData(args, device, batch_size=60000):
    """ loading datasets for estn-ML """

    n1 = args['N_dataset']
    class_num = args['numberClass']

    if args['DATASET'] == 'mnist':  # 60000 + 10000

        datahome = './scikit_learn_data/mnist'
        dataname = 'class_{}'.format(args['numberClass'])
        try:
            X = np.load(os.path.join(datahome, dataname, 'x.npy'))
            y = np.load(os.path.join(datahome, dataname, 'y.npy'))
            print('total mnist load from {}, X={}, y={}.'.format(datahome + dataname, X.shape, y.shape))
        except:
            X, y = fetch_openml('mnist_784', data_home=datahome, version=1, return_X_y=True)
            X /= 255
            print("download total mnist, X={}, y={}.".format(X.shape, y.shape))
            if not os.path.exists(os.path.join(datahome, dataname)):
                os.mkdir(os.path.join(datahome, dataname))
            np.save(os.path.join(datahome, dataname, 'x.npy'), X)
            np.save(os.path.join(datahome, dataname, 'y.npy'), y)
            print('save to {}'.format(os.path.join(datahome, dataname, 'x.npy')))

        y = y.astype(np.int32)
        index = (y < args['numberClass'] )
        X = X[index]
        y = y[index]

        n2 = 10 if class_num > 10 else class_num
        n3 = int(10000 * n2/10) if n1 > 10000 * n2/10 else n1
        # normal train & test
        data_train, data_test = X[:n1, :], X[int(60000* n2/10): int(60000* n2/10)+n3, :]
        label_train, label_test = y[:n1], y[int(60000* n2/10): int(60000* n2/10)+n3]
    
    elif args['DATASET'] == 'mnist_256':  # resize to 256

        datahome = '../scikit_learn_data/mnist_256/'
        dataname = 'class_{}'.format(args['numberClass'])
        d_train = datasets.MNIST(datahome, train=True, download=True).data.float() / 255
        y_train = datasets.MNIST(datahome, train=True, download=True).targets
        d_test  = datasets.MNIST(datahome, train=False).data.float() / 255
        y_test  = datasets.MNIST(datahome, train=False).targets
        X_train = []
        X_test = []
        for i in range(d_train.shape[0]):
            X_train.append(cv2.resize(np.array(d_train[i, :, :], dtype=np.float), (16, 16), interpolation=cv2.INTER_CUBIC))
        for i in range(d_test.shape[0]):
            X_test.append(cv2.resize(np.array(d_test[i, :, :], dtype=np.float), (16, 16), interpolation=cv2.INTER_CUBIC))
        X_train = np.array(X_train, dtype=np.float)
        X_test = np.array(X_test, dtype=np.float)
        
        print("Ori mnist_256, train shape X={}, Y={}, test X={}, test Y={}".format(
            X_train.shape, y_train.shape, X_test.shape, y_test.shape))

        n2 = 10 if class_num > 10 else class_num
        # train
        index = (y_train < torch.tensor(args['numberClass'], dtype=torch.int32))
        y_train = y_train[index]
        X_train = X_train[index]
        n1 = int(60000 * n2/10) if n1 > 60000 * n2/10 else n1
        data_train = X_train[:n1, :]
        label_train = y_train[:n1]
        # test
        index = (y_test < torch.tensor(args['numberClass'], dtype=torch.int32))
        y_test = y_test[index]
        X_test = X_test[index]
        n3 = int(10000 * n2/10) if n1 > 10000 * n2/10 else n1
        data_test = X_test[:n3, :]
        label_test = y_test[:n3]

        print("loaded mnist, train shape X={}, Y={}, test X={}, Y={}".format(
            data_train.shape, label_train.shape, data_test.shape, label_test.shape))
    
    elif args['DATASET'] == 'Fmnist':  # 60000 + 10000

        datahome = './scikit_learn_data/fmnist'
        dataname = 'class_{}'.format(args['numberClass'])

        X_train = datasets.FashionMNIST(datahome, train=True, download=True).data.float() / 255
        y_train = datasets.FashionMNIST(datahome, train=True, download=True).targets
        X_test  = datasets.FashionMNIST(datahome, train=False).data.float() / 255
        y_test  = datasets.FashionMNIST(datahome, train=False).targets
        X_train = X_train.reshape((X_train.shape[0], -1))
        X_test = X_test.reshape((X_test.shape[0], -1))
        print("Ori fmnist, train shape X={}, Y={}, test X={}, test Y={}".format(
            X_train.shape, y_train.shape, X_test.shape, y_test.shape))

        n2 = 10 if class_num > 10 else class_num
        # train
        index = (y_train < torch.tensor(args['numberClass'], dtype=torch.int32))
        y_train = y_train[index]
        X_train = X_train[index]
        n1 = int(60000 * n2/10) if n1 > 60000 * n2/10 else n1
        data_train = X_train[:n1, :]
        label_train = y_train[:n1]
        # test
        index = (y_test < torch.tensor(args['numberClass'], dtype=torch.int32))
        y_test = y_test[index]
        X_test = X_test[index]
        n3 = int(10000 * n2/10) if n1 > 10000 * n2/10 else n1
        data_test = X_test[:n3, :]
        label_test = y_test[:n3]

        print("loaded fmnist, train shape X={}, Y={}, test X={}, test Y={}".format(
            data_train.shape, label_train.shape, data_test.shape, label_test.shape))
    
   
    elif args['DATASET'] == 'cifar-10':
        trainset = datasets.CIFAR10(
            root='./scikit_learn_data/cifar', train=True, download=True,
        )
        testset = datasets.CIFAR10(
            root='./scikit_learn_data/cifar', train=False, download=True,
        )
        data_train, label_train = trainset.data.astype(np.float32) / 255, trainset.targets
        data_test, label_test = testset.data.astype(np.float32) / 255, testset.targets
        data_train = data_train.reshape((data_train.shape[0], -1))
        data_test = data_test.reshape((data_test.shape[0], -1))

        n2 = 10 if class_num > 10 else class_num
        # train
        n1 = int(50000 * n2/10) if n1 > 50000 * n2/10 else n1
        data_train = data_train[:n1, :]
        label_train = label_train[:n1]
        # test
        n3 = int(10000 * n2/10) if n1 > 10000 * n2/10 else n1
        data_test = data_test[:n3, :]
        label_test = label_test[:n3]

        print("cifar-10 train={}, test={}".format(data_train.shape, data_test.shape))
    
    # return data tensor
    data_train = torch.tensor(data_train, device=device).clone().detach()
    data_test = torch.tensor(data_test, device=device).clone().detach()
    label_train = torch.tensor(label_train, device=device).clone().detach()
    label_test = torch.tensor(label_test, device=device).clone().detach()
    return data_train, data_test, label_train, label_test



if __name__ == "__main__":
    import math

    args = {}
    # args["DATASET"] = "usps"
    # args["DATASET"] = "mnist"
    args["DATASET"] = "Fmnist"
    # args["DATASET"] = "Kmnist"
    # args["DATASET"] = "cifar-10"
    args['numberClass'] = 10

    # args["DATASET"] = "coil-20"
    # args["DATASET"] = "coil-100_class=30"
    # args['numberClass'] = 20

    args['N_dataset'] = 10000
    args['BATCHSIZE'] = 10000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_train, x_test, y_train, y_test = LoadRealData(args, device)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # 1. show data samples
    sample = x_train[1].cpu().numpy() * 255
    print("sample shape={}, label={}".format(sample.shape, y_train[1]))
    if args["DATASET"] == "coil-100_class=30":
        sample = np.reshape(sample, (64, 64, 3))
    else:
        if sample.shape[0] == 3072:
            sample = np.reshape(sample, (32, 32, 3))
        elif len(sample.shape) < 2:
            sample = np.reshape(sample, (int(math.sqrt(sample.shape[0])), int(math.sqrt(sample.shape[0]))))
    cv2.imwrite("sample.png", sample)

    # 2. show classification test
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    def TestClassification(embedding, label, class_num=10):
        """ test embedding with classification task """
        method = LogisticRegression(solver='newton-cg', multi_class='auto', C=class_num)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        n_scores = cross_val_score(method, embedding, label, scoring='accuracy', cv=cv, n_jobs=1)

        return n_scores.mean()
    
    cls_score = TestClassification(x_test.cpu().numpy(), y_test.cpu().numpy(), class_num=args['numberClass'])
    print("dataset classification score={}".format(cls_score))
