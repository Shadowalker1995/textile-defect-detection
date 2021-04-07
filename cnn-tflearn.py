import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn import ImagePreprocessing
from tflearn import ImageAugmentation

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os

from util import load_data, gen_label


def Network(class_num):
    tf.compat.v1.reset_default_graph()
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_samplewise_stdnorm()
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_flip_updown()
    convnet = input_data(shape=[None, 200, 200, 1],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug,
                         name='input')
    convnet = conv_2d(convnet, 32, 2, activation='relu', name="con1")
    convnet = max_pool_2d(convnet, 2, name="pool1")

    convnet = conv_2d(convnet, 64, 2, activation='relu', name="con2")
    convnet = max_pool_2d(convnet, 2, name="pool2")

    convnet = conv_2d(convnet, 128, 2, activation='relu', name="con3")
    convnet = max_pool_2d(convnet, 2, name="pool3")

    convnet = conv_2d(convnet, 256, 2, activation='relu', name="con4")
    convnet = max_pool_2d(convnet, 2, name="pool4")

    convnet = conv_2d(convnet, 256, 2, activation='relu', name="con5")
    convnet = max_pool_2d(convnet, 2, name="pool5")

    convnet = conv_2d(convnet, 128, 2, activation='relu', name="con6")
    convnet = max_pool_2d(convnet, 2, name="pool6")

    convnet = conv_2d(convnet, 64, 2, activation='relu', name="con7")
    convnet = max_pool_2d(convnet, 2, name="pool7")

    convnet = fully_connected(convnet, 1000, activation='relu', name="full1")
    convnet = dropout(convnet, 0.75)

    convnet = fully_connected(convnet, class_num, activation='softmax', name="full2")

    convnet = regression(convnet,
                         optimizer='adam',
                         learning_rate=0.0008,
                         loss='categorical_crossentropy',
                         name='regression')

    return convnet


def train(x_train, x_eval, y_train, y_eval, model, save_file=None):
    model.fit(x_train, y_train,
              n_epoch=300,
              batch_size=32,
              validation_set=(x_eval, y_eval),
              show_metric=True,
              snapshot_step=100,
              run_id='20210407')
    if save_file:
        model.save(save_file)

    return model


def predict(input_array, targets, model, model_file=None, csv_file=None):
    if model_file:
        model.load(model_file)

    predictions = model.predict(input_array)
    predictions = np.argmax(predictions, axis=1)
    targets = np.argmax(targets, axis=1)

    acc = np.mean(targets == predictions)

    if csv_file:
        rst = {"targets": targets,
               "predictions": predictions}
        rst = pd.DataFrame(rst)
        rst.to_csv(csv_file)

    return acc


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def main():
    # label_dict = gen_label()
    # print(label_dict)

    X, Y = load_data(dirname="./data/8Classes-9041/all/", dataset_name='8Classes-9041-onehot.pkl',
                     one_hot=True, convert_gray=True)
    x, x_test, y, y_test = train_test_split(X, Y, test_size=0.2)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

    network = Network(len(Y[0]))

    model = tflearn.models.dnn.DNN(network,
                                   tensorboard_dir='./tflearn_logs/',
                                   checkpoint_path='./ckpt/ckpt/',
                                   best_checkpoint_path='./ckpt/ckpt//best/',
                                   tensorboard_verbose=3,
                                   best_val_accuracy=0.9
                                   )

    trained_model = train(x_train, x_val, y_train, y_val, model, save_file="./models/dd.tfl")
    # acc_train = predict(x_train, y_train, model,
    #                     model_file="./models/MilikenTrain.tfl",
    #                     csv_file="./results/predicted_train.csv")
    # print("Train accuracy: ", acc_train)
    # acc_val = predict(x_val, y_val, model,
    #                   model_file="./models/MilikenTrain.tfl",
    #                   csv_file="./results/predicted_val.csv")
    # print("Validation accuracy: ", acc_val)
    # acc_test = predict(x_test, y_test, model,
    #                    model_file="./models/MilikenTrain.tfl",
    #                    csv_file="./results/predicted_test.csv")
    # print("Test accuracy: ", acc_test)


if __name__ == "__main__":
    main()
