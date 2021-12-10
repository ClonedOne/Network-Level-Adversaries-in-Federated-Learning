import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD

EPOCHS_PER_ROUND = 2
BATCH_SIZE = 32


# SHARED MODEL DEFINITION

def build_model(momentum=0.0, dropouts=False):
    """ Build the local model

    Args:
        momentum (float, optional): momentum value for SGD. Defaults to 0.0.
        dropouts (bool, optional): if True use dropouts. Defaults to False.

    Returns:
        object: model object
    """

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if dropouts:
        model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    if dropouts:
        model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    if momentum:
        sgd = SGD(lr=0.1, momentum=momentum)
    else:
        sgd = SGD(lr=0.1)

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=sgd, metrics=['accuracy'])

    return model


# CLIENT SIDE

class LocalModel:
    """ Class representing a client's model in the federated learning protocol
    """

    def __init__(self, x, y, momentum=0.0, dropouts=False):
        """ Initialize a client's model

        Args:
            X (numpy.ndarray): training data
            y (numpy.ndarray): training labels
            momentum (float): momentum parameter for SGD
            dropouts (bool): whether to use dropout layers
        """

        self.model = build_model(momentum, dropouts)
        self.x_train = x
        self.y_train = y

    def get_weights(self):
        """ Retrieve the model's weights

        Returns:
            list: list of numpy arrays containing the model's weights
        """
        return self.model.get_weights()

    def set_weights(self, new_weights):
        """ Assign weights to the model

        Args:
            new_weights (list): list of numpy arrays containing the model's new weights
        """
        self.model.set_weights(new_weights)

    def train_one_round(self):
        """ Train the local model for one roung

        This can correspond to training for multiple epochs, or a single epoch.
        Returs return final weights, train loss, train accuracy

        Returns:
            tuple: final weights, train loss, train accuracy
        """

        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=EPOCHS_PER_ROUND,
            batch_size=BATCH_SIZE,
            #verbose=1
            verbose=0
        )

        score = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        #score = self.model.evaluate(self.x_train, self.y_train, verbose=1)

        return self.model.get_weights(), score[0], score[1]


class LocalBackdoorModel(LocalModel):
    """ Wrapper for the poisoning attack
    """

    def __init__(self, X, y):
        LocalModel.__init__(self, X, y)


class FederatedClient(object):
    """ Class representing a client in the federated learning protocol
    """

    def __init__(self, X, y, momentum=0.0, dropouts=False):
        """ Initialize a client

        Args:
            X (numpy.ndarray): training data
            y (numpy.ndarray): training labels
            momentum (float): momentum parameter for SGD
            dropouts (bool): whether to use dropout layers
        """

        self.local_model = LocalModel(X, y, momentum, dropouts)

    def weights_update(self, current_weights):
        """ Apply the current round's weights and perform one round of training

        Args:
            current_weights (list): list of numpy arrays representing the global model weights

        Returns:
            tuple: final weights, train loss, train accuracy
        """

        self.local_model.set_weights(current_weights)

        my_weights, train_loss, train_accuracy = self.local_model.train_one_round()

        return my_weights, train_loss, train_accuracy

    def weights_backdoor(self, current_weights, boost_factor):
        """ Aplly the current round's weights and perform model substitution attack

        This is a boosting model replacement attack as described in 
        https://arxiv.org/pdf/1807.00459.pdf
        The model update required to substitute the general model is
        evil_update = old_weights + boost_factor * (adv_weights - old_weights)
        boost_factor = n_clients / learning_rate

        Args:
            current_weights (list): list of numpy arrays representing the global model weights
            boost_factor (float): factor to boost the poisoned model's weights 

        Returns:
            tuple: final weights, train loss, train accuracy
        """

        self.local_model.set_weights(current_weights)
        my_weights, train_loss, train_accuracy = self.local_model.train_one_round()

        diff = [new - old for (old, new) in zip(current_weights, my_weights)]
        new_weights = [d * boost_factor +
                       old for (old, d) in zip(current_weights, diff)]

        return new_weights, train_loss, train_accuracy


# SERVER SIDE

class GlobalModel(object):

    def __init__(self, momentum=0.0, dropouts=False):
        """ Initialize a global model

        Args:
            momentum (float): momentum parameter for SGD
            dropouts (bool): whether to use dropout layers
        """

        self.model = build_model(momentum, dropouts)

        # all rounds; losses[i] = [round#, timestamp, loss]
        self.train_losses = []
        self.valid_losses = []

        self.train_accuracies = []
        self.valid_accuracies = []

    def get_weights(self):
        """ Retrieve the model's weights

        Returns:
            list: list of numpy arrays containing the model's weights
        """
        return self.model.get_weights()

    def set_weights(self, new_weights):
        """ Assign weights to the model

        Args:
            new_weights (list): list of numpy arrays containing the model's new weights
        """
        self.model.set_weights(new_weights)

    def aggregate_weights(self, weights_li, agg_fn='mean', **kwargs):
        """ Wrapper for the aggregation function

        Args:
            weights_li (list): list of lists containing all clients' weights
            agg_fn (str, optional): type of aggregation method. Defaults to 'mean'.

        Returns:
            [type]: [description]
        """
        if agg_fn == 'mean':
            return self.aggregate_weights_mean(weights_li, **kwargs)
        elif agg_fn == 'clip':
            return self.aggregate_weights_clip(weights_li, **kwargs)

    def aggregate_weights_mean(self, weights_li, lr=0.1):
        """ Aggregate the weights from all clients with basic averaging

        The aggregation scheme for n clients is:
        new_weights = old_weights + lr * SUM_i(weights_i - old_weights)
                                    ---
                                     n

        Args:
            weights_li (list): list of lists containing all clients' weights
            lr (float, optional): aggregation learning rate. Defaults to 0.1.

        Returns:
            numpy.ndarray: numpy array of the new weights
        """

        old_weights = self.get_weights()
        ave_weights = np.mean(weights_li, axis=0)

        # Here new_weights gets automatically converted to a numpy array
        new_weights = old_weights + lr*(ave_weights - old_weights)

        return new_weights

    def aggregate_weights_clip(self, weights_li, lr=.1, clip_nm=1):
        """ Aggregate the weights from all clients with clipping

        Here the maximum contribution of each model is clipped so that its 
        norm is bounded by clip_nm.

        Args:
            weights_li (list): list of lists containing all clients' weights
            lr (float, optional): aggregation learning rate. Defaults to 0.1.
            clip_nm (int, optional): clipping value. Defaults to 1.

        Returns:
            numpy.ndarray: numpy array of the new weights
        """

        old_weights = self.get_weights()

        # Compute the 2-norm of the difference between the old and new weights for
        # each client.
        norm_li = [np.linalg.norm([np.linalg.norm(
            old - w) for (old, w) in zip(old_weights, user)]) for user in weights_li]

        # Clip the norms
        norm_li = [max(1, norm / clip_nm) for norm in norm_li]

        # Divide the updated by the clipped norms
        clipped_li = [[(w - old) / norm + old for (old, w) in zip(old_weights, user)]
                      for (user, norm) in zip(weights_li, norm_li)]

        ave_weights = np.mean(clipped_li, axis=0)
        new_weights = old_weights + lr * (ave_weights - old_weights)

        return new_weights


class FLServer:
    """ Class representing a server in the federated learning protocol
    """

    def __init__(self):
        """ Initialize a server object
        """

        self.global_model = GlobalModel()

    def score(self, X, y):
        """ Evaluate the global FL model on the given data

        Args:
            X (numpy.array): evaluation data
            y (numpy.array): evaluation labels

        Returns:
            list: loss, accuracy
        """

        score = self.global_model.model.evaluate(X, y, verbose=0)

        return score


# UTILITY

def test():
    """ Small testing script
    """
    model = GlobalModel()
    a = model.get_weights()
    print('Type of model.get_weights():', type(a))
    print(len(a))

    b = [a] * 5
    nw = model.aggregate_weights_mean(b)
    print('Type of the returned object:', type(nw))
    print(nw.shape)
    print(nw)


if __name__ == '__main__':
    test()
