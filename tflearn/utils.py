from tflearn.data_utils import *


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
