import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Cố định seed ngẫu nhiên
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# Cấu hình để đảm bảo tái lập trên GPU
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Helper: Early stopping
early_stopper = EarlyStopping(patience=5)

def get_mnist():
    """Retrieve the MNIST dataset and process the data."""
    nb_classes = 10
    batch_size = 128
    input_shape = (784,)

    # Lấy dữ liệu
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Chuyển đổi nhãn sang dạng one-hot
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

def compile_model(network, nb_classes, input_shape):
    """Compile a sequential model."""
    nb_neurons = network['nb_neurons']
    nb_layers = network['nb_layers']
    activation = network['activation']
    optimizer = network['optimizer']
    model = Sequential()

    for layer in range(nb_layers):
        if layer == 0:
            model.add(Dense(nb_neurons, activation=activation, input_shape=input_shape))
        else:
            model.add(Dense(nb_neurons, activation=activation))
        model.add(Dropout(0.2))
    
    # Lớp đầu ra
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def train_and_score(network):
    """Train the model, return test accuracy."""
    nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test = get_mnist()
    model = compile_model(network, nb_classes, input_shape)

    model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=10000,
        verbose=0,
        validation_data=(x_test, y_test),
        callbacks=[early_stopper]
    )

    score = model.evaluate(x_test, y_test, verbose=0)
    return score[1]