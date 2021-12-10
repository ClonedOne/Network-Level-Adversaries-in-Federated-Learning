import os
import random

import numpy as np

from nlafl import common



def load_emnist():
    """ Load the EMNIST dataet

    Returns:
        tuple: tuple of numpy arrays trn_x, trn_y, tst_x, tst_y
    """

    trn_x = np.load(os.path.expanduser(common.emnist_trn_x_pth))
    trn_y = np.load(os.path.expanduser(common.emnist_trn_y_pth))
    tst_x = np.load(os.path.expanduser(common.emnist_tst_x_pth))
    tst_y = np.load(os.path.expanduser(common.emnist_tst_y_pth))
    return trn_x, trn_y, tst_x, tst_y


def partition(x, y):
    """ Given a dataset matrix and labels, return the data matrix partitioned by class.

    The list of classes is assumed to be the number of classes for the dataset.

    Example output:
        [ [class 1's x ..], [class 2's x ..] ,  ... [class 10^s x ..]  ]

    Args:
        x (numpy.ndarray): data matrix
        y (numpy.ndarray): data labels

    Returns:
        list: Partitioned data matrix, as list of ndarray objects
    """

    all_x = []
    y_list = range(common.num_classes['emnist'])

    for y_val in y_list:
        all_x.append(x[np.where(y == y_val)[0]])
    return all_x


# Note: Unused, keeping for reference
def dirichlet_sample(all_x, num_clients, client_size, alpha=100, seed=None):
    """  Sample the dataset using a Dirichlet distribution

    Exact client size is int( districtled alpha * client size) for each class.
    `all_x` contains x values of each class in the format:
        [ [class 1's x ..], [class 2's x ..] ,  ... [class 10^s x ..]  ]

    Args:
        all_x (list): Partitioned data matrix, as list of ndarray objects
        num_clients (int): number of clients
        client_size (int): desired number of samples per client
        alpha (int, optional): Dirichlet parameter alpha. Defaults to 100.
        seed (int, optional): seed for PRNGs. Defaults to None.

    Returns:
        list: list of tuples, (data, labels)) for each client
    """

    # Seed the PRNGs
    np.random.seed(seed)
    random.seed(seed)

    # Initialize per-client data structures
    num_classes = common.num_classes['emnist']
    clients = []
    all_dirichlets = np.random.dirichlet(
        [alpha for _ in range(num_classes)],
        num_clients
    )

    # shape (num_clients, num_classes)
    for i in range(num_clients):
        this_x, this_y = [], []
        # dirichlet[i] -> distribution of each class for  client i
        # multiplied with client_size gives the number of expected samples for each class for client i
        cur_counts = client_size*all_dirichlets[i]

        for y in range(num_classes):
            # since cur_counts is float and number of samples converted to int
            y_ct = cur_counts[y].astype(np.int)

            # take y_ct many samples for each class using for loop
            this_x.append(all_x[y][:y_ct])

            # from all_x[y] disgard the ones already pick, continue with remaining samples
            all_x[y] = all_x[y][y_ct:]

            this_y.append(np.zeros(y_ct, dtype=np.int)+y)  # -> [y ] * y_ct

        this_x = np.concatenate(this_x)
        this_y = np.concatenate(this_y)
        assert this_x.shape[0] == this_y.shape[0]
        clients.append((this_x, this_y))

    return clients


def fixed_sample(
    all_x,
    num_clients,
    client_size,
    targ_class=0,
    client_targ=5,
    targ_frac=.2,
    alpha=100,
    seed=None
):
    """ Use a Dirichlet distribution to assign target class samples to clients

    `all_x` -> [ [class 1's x ..], [class 2's x ..] ,  ... [class 10^s x ..]  ]
    `client Size` is used to calculate number samples for each class with
    dirichlet distirbution alpha

    Args:
        all_x (list): partitioned data matrix, as list of ndarray objects
        num_clients (int): number of clients
        client_size (int): desired number of samples per client
        targ_class (int, optional): identifier of target class. Defaults to 0
        client_targ (int, optional): number of clients having target class points. Defaults to 5
        targ_frac (float, optional): fraction of target class points for clients having them. Defaults to .2
        alpha (int, optional): Dirichlet parameter alpha. Defaults to 100
        seed (int, optional): seed for PRNGs. Defaults to None

    Returns:
        list: list of tuples, (data, labels)) for each client
    """

    # Seed the PRNGs
    np.random.seed(seed)
    random.seed(seed)

    num_classes = common.num_classes['emnist']
    num_nontarget = num_classes - 1

    # Initialize per-client data structures
    clients = []
    orig_dirichlets = np.random.dirichlet([alpha] * num_nontarget, num_clients)
    all_dirichlets = np.zeros((num_clients, num_classes))

    # Fill up the columns of `all_dirichlets` up to the target class,
    # and from the one following the target class to the end using the
    # values generated in `orig_dirichlets`
    all_dirichlets[:, :targ_class] = orig_dirichlets[:, :targ_class]
    all_dirichlets[:, targ_class+1:] = orig_dirichlets[:, targ_class:]

    # targ_x is the numpy array of all target class samples
    targ_x = all_x[targ_class]

    for i in range(num_clients):
        this_x, this_y = [], []
        total_ct = client_size

        # The first client_targ clients will have the target class samples
        if i < client_targ:
            # number of target class samples for client i
            num_targ = int(total_ct * targ_frac)
            total_ct -= num_targ

            # Assign the target class samples to client i and create a label vector
            this_x.append(targ_x[:num_targ])
            this_y.append(np.zeros(num_targ, dtype=np.int) + targ_class)

            # Remove the samples used for this client from targ_x
            targ_x = targ_x[num_targ:]

        counts = (total_ct * all_dirichlets[i]).astype(np.int)
        assert counts[targ_class] == 0

        for y in range(num_classes):
            # Ignore the target class
            if y == targ_class:
                continue

            y_ct = counts[y].astype(np.int)
            this_x.append(all_x[y][:y_ct])
            all_x[y] = all_x[y][y_ct:]
            this_y.append(np.zeros(y_ct, dtype=np.int) + y)

        this_x = np.concatenate(this_x)
        this_y = np.concatenate(this_y)
        assert this_x.shape[0] == this_y.shape[0]
        clients.append((this_x, this_y))

    return clients


def fixed_poison(
    all_x,
    num_clients,
    client_size,
    poison_ct,
    targ_class=0,
    client_targ=5,
    targ_frac=.2,
    alpha=100,
    seed=None
):
    """

    Args:
        all_x (list): partitioned data matrix, as list of ndarray objects
        num_clients (int): number of clients
        client_size (int): desired number of samples per client
        poison_ct (int): number of clients participating in the poisoning attack
        targ_class (int, optional): identifier of target class. Defaults to 0
        client_targ (int, optional): number of clients having target class points. Defaults to 5
        targ_frac (float, optional): fraction of target class points for clients having them. Defaults to .2
        alpha (int, optional): Dirichlet parameter alpha. Defaults to 100
        seed (int, optional): seed for PRNGs. Defaults to None

    Returns:
        list: list of tuples, (data, labels)) for each client
    """

    # Seed the PRNGs
    np.random.seed(seed)
    random.seed(seed)

    num_classes = common.num_classes['emnist']
    num_nontarget = num_classes - 1

    # Initialize per-client data structures
    clients = []
    orig_dirichlets = np.random.dirichlet([alpha] * num_nontarget, num_clients)
    all_dirichlets = np.zeros((num_clients, num_classes))

    # Fill up the columns of `all_dirichlets` up to the target class,
    # and from the one following the target class to the end using the
    # values generated in `orig_dirichlets`
    all_dirichlets[:, :targ_class] = orig_dirichlets[:, :targ_class]
    all_dirichlets[:, targ_class+1:] = orig_dirichlets[:, targ_class:]

    # targ_x is the numpy array of all target class samples
    targ_x = all_x[targ_class]

    for i in range(num_clients):
        this_x, this_y = [], []
        total_ct = client_size

        # The first client_targ clients will have the target class samples
        if i < client_targ:
            # number of target class samples for client i
            num_targ = int(total_ct * targ_frac)
            total_ct -= num_targ

            # Assign the target class samples to client i and create a label vector
            this_x.append(targ_x[:num_targ])
            this_y.append(np.zeros(num_targ, dtype=np.int)+targ_class)

            # Remove the samples used for this client from targ_x
            targ_x = targ_x[num_targ:]

        # The successive `poison_ct` clients will have the poisoned points
        elif i < client_targ + poison_ct:
            num_targ = int(total_ct * targ_frac)
            total_ct -= num_targ
            counts = (total_ct * all_dirichlets[i]).astype(np.int)

            # Flip the labels for the target class samples
            for y in range(num_classes):
                if y == targ_class:
                    y_ct = num_targ
                    y_local = (y + 1) % num_classes

                else:
                    y_ct = counts[y].astype(np.int)
                    y_local = y

                # Assign the samples to this client
                this_x.append(all_x[y][:y_ct])
                this_y.append(np.zeros(y_ct, dtype=np.int) + y_local)

                # Remove the samples used for this client
                all_x[y] = all_x[y][y_ct:]

            this_x = np.concatenate(this_x)
            this_y = np.concatenate(this_y)
            assert this_x.shape[0] == this_y.shape[0]
            clients.append((this_x, this_y))
            continue

        counts = (total_ct*all_dirichlets[i]).astype(np.int)
        assert counts[targ_class] == 0

        for y in range(num_classes):
            # Ignore the target class
            if y == targ_class:
                continue

            y_ct = counts[y].astype(np.int)
            this_x.append(all_x[y][:y_ct])
            all_x[y] = all_x[y][y_ct:]
            this_y.append(np.zeros(y_ct, dtype=np.int) + y)

        this_x = np.concatenate(this_x)
        this_y = np.concatenate(this_y)
        assert this_x.shape[0] == this_y.shape[0]
        clients.append((this_x, this_y))

    return clients


if __name__ == '__main__':
    """ Test utility """
    target_class = 3
    targ_frac = .5
    cli_targ = 5
    poison_ct = 5
    cli_size = 1000

    trn_x, trn_y, tst_x, tst_y = load_emnist()
    partitioned = partition(trn_x, trn_y)
    print('\nPer-class dataset shapes')
    print([v.shape for v in partitioned])

    if poison_ct > 0:
        clients = fixed_poison(
            partitioned, 100, cli_size, poison_ct=poison_ct, targ_class=target_class,
            client_targ=cli_targ, targ_frac=targ_frac, alpha=1, seed=None
        )

    else:
        clients = fixed_sample(
            partitioned, 100, cli_size, targ_class=target_class,
            client_targ=cli_targ, targ_frac=targ_frac, alpha=1, seed=0
        )

    print('\nPer-client local dataset shapes')
    print([client[0].shape for client in clients])

    print('\nNumber of clients and total number of points selected')
    print(len(clients), sum([client[0].shape[0] for client in clients]))

    print(f"\nNumber of clients having class {target_class}:", sum(
        [(target_class in np.unique(client[1])) for client in clients]))

    # Sanity check
    for client in clients[:cli_targ]:
        assert sum(client[1] == target_class) == int(cli_size * targ_frac)

    print('Per-client breakdown of classes and number of points per class')
    for counter, client in enumerate(clients):
        client_x, client_y = client
        print(f"Client {counter} has unique",
              np.unique(client_y, return_counts=True))
