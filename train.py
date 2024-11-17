from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)

def get_mnist():
    """Retrieve the MNIST dataset and process the data."""
    # Set defaults.
    nb_classes = 10
    batch_size = 128
    input_shape = (784,)

    # Get the data.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

"""
    nn_param_choices = {
        'nb_neurons': [64, 128, 256, 512, 768, 1024],
        'nb_layers': [1, 2, 3, 4],
        'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
        'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad',
                      'adadelta', 'adamax', 'nadam'],
    }
"""
def compile_model(network, nb_classes, input_shape):
    """Compile a sequential model.

    Args:
        network (dict): the parameters of the network

    Returns:
        a compiled network.
    """
    nb_neurons  = network['nb_neurons']
    nb_layers   = network['nb_layers']
    activation  = network['activation']
    optimizer   = network['optimizer']
    model   = Sequential()
    for layer in range(nb_layers):
        if layer == 0:
            model.add(Dense(nb_neurons, activation = activation, input_shape = input_shape))
        else:
            model.add(Dense(nb_neurons, activation = activation))

        model.add(Dropout(0.2))
    #output layer
    model.add(Dense(nb_classes, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return model




def train_and_score(network):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
    """
    nb_classes, batch_size, \
        input_shape,x_train, x_test, y_train, y_test = get_mnist()
    model = compile_model(network= network, nb_classes= nb_classes, input_shape= input_shape)

    model.fit(
        x_train,y_train,
        batch_size      = batch_size,
        epochs          = 10000,
        verbose         = 0,
        validation_data = (x_test,y_test),
        callbacks        = [early_stopper]
    )

    score = model.evaluate(x_test, y_test, verbose = 0)
    return score[1] 

