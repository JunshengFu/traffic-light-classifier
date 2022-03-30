import pickle
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import load_model


def network():
    """
    Define the network
    :return:
    """
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(4))
    model.add(Activation('softmax'))

    return model


def train(file_path, model):

    x_,y_ = pickle.load( open(file_path, "rb" ) )
    random_state = 130
    X_train, x_validation, y_train, y_validation = train_test_split(x_, y_, train_size = 0.80,
                                                                    test_size = 0.2,
                                                                    random_state = random_state)
    # preprocess data
    X_normalized = np.array(X_train / 255.0 - 0.5 )
    label_binarizer = LabelBinarizer()
    y_one_hot = label_binarizer.fit_transform(y_train)

    model.summary()
    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    history = model.fit(X_normalized, y_one_hot, epochs=20, validation_split=0.2)

    model.save('model.h5')
    return history



def test(file_path, model):

    X_test,y_test = pickle.load( open(file_path, "rb" ) )

    # preprocess data
    X_normalized_test = np.array(X_test / 255.0 - 0.5 )
    label_binarizer = LabelBinarizer()
    y_one_hot_test = label_binarizer.fit_transform(y_test)

    print("Testing")

    metrics = model.evaluate(X_normalized_test, y_one_hot_test)
    for metric_i in range(len(model.metrics_names)):
        metric_name = model.metrics_names[metric_i]
        metric_value = metrics[metric_i]
        print('{}: {}'.format(metric_name, metric_value))


def test_an_image(file_path, model):
    """
    resize the input image to [32, 32, 3], then feed it into the NN for prediction
    :param file_path:
    :return:
    """

    desired_dim=(32,32)
    img = cv2.imread(file_path)
    img_resized = cv2.resize(img, desired_dim, interpolation=cv2.INTER_LINEAR)
    img_ = np.expand_dims(np.array(img_resized), axis=0)

    tmp_state = model.predict(img_)
    predicted_state = tmp_state.argmax(axis=-1)

    return predicted_state


if __name__ == "__main__":
    model = network()
    train_file = "./data/bosch_udacity_train.p"
    test_file = "./data/bosch_udacity_test.p"

    # Train the network
    train(train_file, model)

    # Test the network
    test(test_file, model=load_model('model.h5'))

    #---Test with a single image---#
    demo_flag = True
    file_path = './data/green.jpg'
    states = ['red', 'yellow', 'green', 'off']
    if demo_flag:
        predicted_state = test_an_image(file_path, model=load_model('model.h5'))
        for idx in predicted_state:
            print(states[idx])



