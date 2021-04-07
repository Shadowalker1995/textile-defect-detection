import itertools
import numpy as np
import matplotlib.pyplot as plt
from tflearn.data_utils import *


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          isSave=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `mormaliza=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion metrix, without normalization")

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if isSave:
        plt.savefig('{}.png'.format(title))
    plt.show()


def y2indicator(Y):
    K = len(set(Y))
    N = len(Y)
    I = np.empty((N, K))
    I[np.arange(N), Y] = 1
    return I


def load_data(dirname="./data/8Classes-9041/all/", dataset_name='8Classes-9041-all.pkl',
              resize_pics=(200, 200),
              shuffle=True, one_hot=False, convert_gray=False):
    dataset_path = os.path.join(dirname, dataset_name)

    X, Y = build_image_dataset_from_dir(dirname,
                                        dataset_file=dataset_path,
                                        resize=resize_pics,
                                        filetypes=['.bmp'],
                                        convert_gray=convert_gray,
                                        shuffle_data=shuffle,
                                        categorical_Y=one_hot)
    X = np.expand_dims(X, axis=3)

    return X, Y


def gen_label(dirname="./data/8Classes-9041/all/"):
    classes = sorted(os.walk(dirname).__next__()[1])
    label_dict = {}
    for i, c in enumerate(classes):
        label_dict[c] = i
    return label_dict
