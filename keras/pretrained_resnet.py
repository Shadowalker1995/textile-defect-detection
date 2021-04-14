from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
import sys


class Logger():
    def __init__(self, filename='results.log'):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main():
    sys.stdout = Logger()

    # resize all the image to this, feel free to change depending on dataset
    IMAGE_SIZE = [224, 224]

    # training config
    epochs = 1
    batch_size = 32

    # https://www.kaggle.com/paultimothymooney/blood-cells
    train_path = '../data_split/train'
    valid_path = '../data_split/val/'

    # useful for getting number of files
    image_file = glob(train_path + '/*/*.bmp')
    valid_image_file = glob(valid_path + '/*/*.bmp')

    # useful for getting number of classes
    folders = glob(train_path + '/*')

    # look at an image for fun
    # plt.imshow(image.img_to_array(image.load_img(np.random.choice(image_file))).astype('uint8'))
    plt.imshow(image.load_img(np.random.choice(image_file)))
    plt.show()

    # add preprocessing layer to the front of res
    res = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

    # don't train existing weights
    for layer in res.layers:
        layer.trainable = False

    # our layer - you can add more if you want
    x = Flatten()(res.output)
    # x = Dense(1000, activation='relu')(x)
    prediction = Dense(len(folders), activation='softmax')(x)

    # create a model object
    model = Model(inputs=res.input, outputs=prediction)

    # view the structure of the model
    model.summary()

    # tell the model what cost and optimization method to use
    model.compile(
            loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])

    # create an instance of ImageDataGenerator
    gen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            preprocessing_function=preprocess_input)

    # test generator to see how it works and some other useful things
    # get label mapping for confusion matrix plot later
    test_gen = gen.flow_from_directory(valid_path, target_size=IMAGE_SIZE)
    print(test_gen.class_indices)
    labels = [None] * len(test_gen.class_indices)
    for k, v in test_gen.class_indices.items():
        labels[v] = k

    # should be a strangely colored image (due to res weights being BGR)
    for x, y in test_gen:
        print("min:", x[0].min(), "max:", x[0].max())
        plt.title(labels[np.argmax(y[0])])
        plt.imshow(x[0])
        plt.show()
        break

    # create generators
    train_generator = gen.flow_from_directory(
            train_path,
            target_size=IMAGE_SIZE,
            shuffle=True,
            batch_size=batch_size,)
    valid_generator = gen.flow_from_directory(
            valid_path,
            target_size=IMAGE_SIZE,
            shuffle=True,
            batch_size=batch_size,)

    # fit the model
    r = model.fit(
            train_generator,
            validation_data=valid_generator,
            epochs=epochs,
            steps_per_epoch=len(image_file) // batch_size,
            validation_steps=len(valid_image_file) // batch_size,)

    model.save(
        './models/model.tf',
        overwrite=True,
        include_optimizer=True,
        save_format='tf',
    )


    def get_confusion_matrix(data_path, N):
        # we need to see the data in the same order
        # for both predictions and targets
        print("Generating confusion matrix", N)
        predictions = []
        targets = []
        i = 0
        for x, y in gen.flow_from_directory(data_path, target_size=IMAGE_SIZE, shuffle=False, batch_size=batch_size * 2):
            i += 1
            if i % 50 == 0:
                print(i)
            p = model.predict(x)
            p = np.argmax(p, axis=1)
            y = np.argmax(y, axis=1)
            predictions = np.concatenate((predictions, p))
            targets = np.concatenate((targets, y))
            if len(targets) >= N:
                break

        cm = confusion_matrix(targets, predictions)
        return cm

    cm = get_confusion_matrix(train_path, len(image_file))
    print(cm)
    valid_cm = get_confusion_matrix(valid_path, len(valid_image_file))
    print(valid_cm)

    # plot some data
    # loss
    print('loss: ', r.history['loss'])
    print('val_loss: ', r.history['val_loss'])
    plt.plot(r.history['loss'], label='train loss')
    plt.plot(r.history['val_loss'], label='val loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()

    # accurate
    print('accuracy: ', r.history['accuracy'])
    print('val_accuracy: ', r.history['val_accuracy'])
    plt.plot(r.history['accuracy'], label='train acc')
    plt.plot(r.history['val_accuracy'], label='val acc')
    plt.legend()
    plt.savefig('accuracy.png')
    plt.show()

    from utils import plot_confusion_matrix
    plot_confusion_matrix(cm, labels, title='Train confusion matrix', isSave=True)
    plot_confusion_matrix(valid_cm, labels, title='Validation confusion matrix', isSave=True)


if __name__ == '__main__':
    main()
