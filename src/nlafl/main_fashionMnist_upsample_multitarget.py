import os
import random
from collections import defaultdict

import numpy as np
from copy import deepcopy
import tensorflow as tf

from nlafl import fashionMnist_sample
from nlafl.fashionMnist_models import FLServer
from nlafl.fashionMnist_models import FederatedClient
from  nlafl import common
# Disable gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ALPHA = 1
NUM_ROUNDS = 310
NUM_CLIENTS = 10
NUM_USERS = 60
CLIENT_SIZE = 400
NUM_EPOCHS_CLIENT = 20
UPSAMPLE_FACTOR = 2
TARGET_FRACTION = 0.6

# backdoor indexes mnist
SERVER_AGG_DICT = {"lr": .25}

# Stores the improvements for the server
per_round_agg_improvements_attacker = []
per_round_agg_improvements_defender = []

# Stores the improvements for the adversary
per_round_each_improvements_attacker = []
per_round_each_improvements_defender = []

per_round_selected_users   = []
per_round_target_users = []
per_round_selection_probs = []
per_round_selection_probs_argmax = []
per_round_adversary_overlap = []
per_round_defender_overlap = []


agg_improvements_attacker = defaultdict(list)
agg_improvements_defender = defaultdict(list)

each_improvements_attacker = defaultdict(list)
each_improvements_defender = defaultdict(list)

# List of visible clients for adversary
visibility = None


# AUXILIARY FUNCTIONS

def update_information(
    sampled_clients,
    new_model,
    client_wt_li,
    last_performance_attacker,
    last_performance_defender,
    tst_x_attacker,
    tst_y_attacker,
    tst_x_defender,
    tst_y_defender,
    normalize_performance=False
):
    """ Update the lists of losses on population data for the current round

    Args:
        sampled_clients (list): list of clients sampled at the current round
        new_model (object): global model at the current round
        client_wt_li (list): list of clients weights (lists) for the sampled clients
        last_performance (float): loss on the target population at previous round
        tst_x (numpy.ndarray): target population data
        tst_y (numpy.ndarray): target population labels
        normalize_performance (bool, optional): If true normalize the current
            round loss using previous round loss. Defaults to True.

    Returns:
        float: loss on the target population at current round for global model
    """

    # Snapshot the state of the global model accuracy and loss
    # Here tst_x and tst_y are the data and lables for the target population
    cur_performance_defender = new_model.evaluate(tst_x_defender, tst_y_defender, verbose=0)
    cur_performance_attacker = new_model.evaluate(tst_x_attacker, tst_y_attacker, verbose=0)
    print("Defender Loss,Acc on Target Test Set:", cur_performance_defender)
    print("Attacker Loss,Acc on Target Test Set:", cur_performance_attacker)


    # If it's the first round return loss for the server, attacker
    if last_performance_defender is None:
        return cur_performance_defender[0],cur_performance_attacker[0]

    # Find the loss delta for the server, and normalize by previous round loss
    agg_perf_diff_defender = cur_performance_defender[0] - last_performance_defender
    if normalize_performance:
        agg_perf_diff_defender = agg_perf_diff_defender / last_performance_defender

    # Find the loss delta for the attacker, and normalize by previous round loss
    agg_perf_diff_attacker = cur_performance_attacker[0] - last_performance_attacker
    if normalize_performance:
        agg_perf_diff_attacker = agg_perf_diff_attacker / last_performance_attacker

    # tst_x_defender and tst_y_defender only passed for object init, has no effect.
    dummy_client = FederatedClient(tst_x_defender, tst_y_defender)

    # Iterate over all the clients and their weights. Find the loss delta
    # of the local models on the target population, and normalize it.
    for cl, client_wt in zip(sampled_clients, client_wt_li):
        dummy_client.local_model.set_weights(client_wt)
        # For the attacker
        cl_perf_attacker = dummy_client.local_model.model.evaluate(
            tst_x_attacker, tst_y_attacker, verbose=0)
        cl_perf_diff_attacker = cl_perf_attacker[0] - last_performance_attacker

        if normalize_performance:
            cl_perf_diff_attacker = cl_perf_diff_attacker / last_performance_attacker

        # For the defender 
        cl_perf_defender = dummy_client.local_model.model.evaluate(
            tst_x_defender, tst_y_defender, verbose=0)
        cl_perf_diff_defender= cl_perf_defender[0] - last_performance_defender

        if normalize_performance:
            cl_perf_diff_defender = cl_perf_diff_defender / last_performance_defender


        # Log the loss deltas for the clients and the server
        # The server loss is being logged for each client because it represents
        # the scenario where the server itself does not have knowledge of the
        # single client contributions.
        agg_improvements_attacker[cl].append(agg_perf_diff_attacker)
        agg_improvements_defender[cl].append(agg_perf_diff_defender)
        each_improvements_attacker[cl].append(cl_perf_diff_attacker)
        each_improvements_defender[cl].append(cl_perf_diff_defender)

    return cur_performance_defender[0],cur_performance_attacker[0]


def create_dropli(ct, knowledge_dict, pop_clients, visibility=None):
    """ Create list of clients to drop

    The selected clients will be the ones with the lowest average loss on the target

    Args:
        ct (int): size of the drop list
        knowledge_dict (dict): dictionary of floats representing losses for each client
        pop_clients (list): list of clients with target population

    Returns:
        list: list of clients to drop 
    """

    all_changes = [knowledge_dict.get(ind, np.inf) for ind in range(NUM_USERS)]
    all_means = [np.mean(change) for change in all_changes]
    worst_means = np.argsort(all_means)[:ct]

    # Compute overlap between dropped clients and actual target clients
    overlap = len([i for i in worst_means if i in pop_clients])
    print(f"Removing {worst_means}, {overlap} 0 clients")
    
    # If adversary has limited visibility, assign infinite loss to invisible clients .ie. np.inf
    partial_worst_means = []
    if visibility is not None:

        partial_all_changes = []
        for ind in range(NUM_USERS):
            if ind in visibility:
                partial_all_changes.append(knowledge_dict.get(ind, np.inf))
            else:
                partial_all_changes.append(np.inf)

        partial_all_means = [np.mean(change) for change in partial_all_changes]
        partial_worst_means = np.argsort(partial_all_means)[:ct]

    return worst_means,partial_worst_means


def upsample(knowledge_dict, ct, drop_li):
    """ Upsample clients which likely contribute positively to the target class

    Args:
        knowledge_dict (dict): dictionary of floats representing losses for each client
        ct (int): size of the upsample list
        drop_li (list): list of clients to drop

    Returns:
        list: list of sampling probabilites for each client
    """

    all_changes = [knowledge_dict.get(ind, np.inf) for ind in range(NUM_USERS)]
    all_means = [np.mean(change) for change in all_changes]

    # Fint the `ct` clients with the lowest mean loss change.
    # Those are the clients contributing most to the target class
    best_means = []
    for i in np.argsort(all_means):
        if i not in drop_li:
            best_means.append(i)

        if len(best_means) >= ct:
            break
    print('Selected {} clients, with ct = {}'.format(len(best_means), ct))

    # Default uniform probabilities
    prob = (NUM_USERS - UPSAMPLE_FACTOR * ct) / \
        (NUM_USERS * NUM_USERS - NUM_USERS * ct)
    probs = [prob] * NUM_USERS

    # Assign higher probabilities to the clients contributing most to the target class
    for upsample_id in best_means:
        probs[upsample_id] = UPSAMPLE_FACTOR / NUM_USERS

    print('Sum of the generated probabilities: ', sum(probs))
    assert sum(probs) < 1.01 and sum(probs) > 0.99

    return probs




def populate_visibility(num_users,num_pop_clients,visible_frac,visible_alpha):

    weight_list = [visible_alpha] * num_pop_clients + [1] * (num_users-num_pop_clients)
    assert len(weight_list) == num_users
    sample_prob = np.random.dirichlet(weight_list)
    visibility  = np.random.choice(list(range(num_users)),
        size=int(num_users*visible_frac), p=sample_prob, replace=False)

    return visibility



# MAIN SCRIPT

def setup_argparse():
    import argparse
    parser = argparse.ArgumentParser('run federated learning')
    parser.add_argument('target_class', type=int,
                        help='which class is only present in subpopulation and it will be attacked ?')
    parser.add_argument('num_pop_clients', type=int,
                        help='how many clients have target class points?')
    parser.add_argument('remove_pop_clients', type=int,
                        help='how many clients are dropped at round 0, perfect knowledge')
    parser.add_argument("drop_epoch", type=int,
                        help='client identification + dropping epoch T_N, set to -1 to deactivate, overrides perfect knowledge attack')
    parser.add_argument("drop_count", type=int,
                        help='client id + dropping drop count k_N')
    parser.add_argument("poison_count", type=int,
                        help='number of poisoned clients')
    parser.add_argument("trial_ind", type=int, help='trial number')
    parser.add_argument('agg_fn', choices=[
                        'clip', 'mean'], help='aggregate with fedavg or clipped fedavg?')
    parser.add_argument('boost_factor', type=float,
                        help='model poisoning boost factor')
    parser.add_argument('upsample_epoch', type=int,
                        help='defensive upsampling epoch T_S, set to -1 to deactivate')
    parser.add_argument('upsample_ct', type=int,
                        help='defensive upsampling client count k_S')
    parser.add_argument('network_knowledge', choices=[
                        'random', 'agg', 'each'], help='how much knowledge does the network have: random dropping, mean of updates, or update')
    parser.add_argument('server_knowledge', choices=[
                        'agg', 'each'], help='how much knowledge does server have: mean of updates or update')
    parser.add_argument("visible_frac", nargs="?" , default=None, type=float)
    parser.add_argument("visible_alpha", nargs="?",default=None, type=float)


    return parser


if __name__ == '__main__':
    """ Run the federated learning experiment as specified in the arguments
    """

    data_id = 'fashionMnist'
    num_classes = common.num_classes[data_id]

    # Read the command line arguments
    parser = setup_argparse()
    args = parser.parse_args()
    print('Arguments received:\n', args)

    # Set the seed for PRNGs to be equal to the trial index
    seed = args.trial_ind
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    # Populate adversary visibility list
    if (args.visible_frac is not None) and (args.visible_alpha is not None):
        visibility = populate_visibility(NUM_USERS, args.num_pop_clients,
            args.visible_frac, args.visible_alpha)
        print(f'Visibility Set of Attacker: {visibility}')

    # Initialize structures to keep track of the loss improvement
    # for each client on the target population
    knowledge_dicts = {
        'agg_attacker': agg_improvements_attacker,
        'agg_defender': agg_improvements_defender,
        'each_attacker': each_improvements_attacker,
        'each_defender': each_improvements_defender,
        'random_attacker': {}
    }
    network_dict = knowledge_dicts[args.network_knowledge + "_attacker"]
    server_dict = knowledge_dicts[args.server_knowledge + "_defender"]

    # generate clients
    trn_x, trn_y, tst_x, tst_y = fashionMnist_sample.load_fashionMNIST()
    print("trn_x.shape",trn_x.shape)
    print("tst_x.shape",tst_x.shape)
    partitioned = fashionMnist_sample.partition(trn_x, trn_y)
    tst_partitioned = fashionMnist_sample.partition(tst_x, tst_y)

    # One-hot encode the labels
    trn_y, tst_y = np.eye(num_classes)[trn_y], np.eye(num_classes)[tst_y]

    # Sample data from the original dataset according to a Dirichlet distribution.
    # If poison_count is > 0, introduce posioned data for some clients.
    if args.poison_count > 0:
        client_data = fashionMnist_sample.fixed_poison(
            partitioned,
            NUM_USERS,
            CLIENT_SIZE,
            args.poison_count,
            targ_class=args.target_class,
            client_targ=args.num_pop_clients,
            targ_frac=TARGET_FRACTION,
            alpha=ALPHA,
            seed=seed
        )

    else:
        client_data = fashionMnist_sample.fixed_sample(
            partitioned,
            NUM_USERS,
            CLIENT_SIZE,
            targ_class=args.target_class,
            client_targ=args.num_pop_clients,
            targ_frac=TARGET_FRACTION,
            alpha=ALPHA,
            seed=seed
        )

    # One-hot encode the labels for each client
    client_data = [(x, np.eye(num_classes)[y]) for (x, y) in client_data]

    pop_clients = []
    for i, client in enumerate(client_data):
        this_pop_frac = (client[1][:, args.target_class] == 1).mean()

        # This works because non-population clients have 0 fraction of target
        # class data
        if this_pop_frac > 0.1:
            pop_clients.append(i)

    print("Clients have the target population:", pop_clients)

    # Select the method for dropping clients
    # Dropping based on client identification overrides perfect knowledge attack
    if args.drop_epoch >= 1:
        target_clients = []

    # Random dropping baseline
    elif args.network_knowledge == 'random':
        target_clients = np.random.choice(
            NUM_USERS, args.remove_pop_clients, replace=False)
        print("Randomly choisen target class prediction: ", target_clients)

    # Perfect knowledge dropping
    else:
        target_clients = np.random.choice(
            pop_clients, args.remove_pop_clients, replace=False)
        print("Perfect knowledge chosen target class prediction: ", target_clients)

    # Initialize server and accumulation variables
    server = FLServer()
    server_weights = server.global_model.get_weights()
    mean_pop_frac_li = []
    mean_improved_li = []
    accuracy_li = []
    # Last performance for the attacker
    last_performance_attacker = None
    # Last Performance for the defender
    last_performance_defender = None

    # Target class test set - change this to change target dataset
    pop_ds = (
        # Test data
        tst_partitioned[args.target_class],
        # Test labels, one-hot encoded
        np.eye(num_classes)[
            np.zeros(tst_partitioned[args.target_class].shape[0], dtype=np.int)
            + args.target_class
        ]
    )
    assert (pop_ds[0].shape[0] == pop_ds[1].shape[0])

    # Size of the pop ds
    pop_ds_size = pop_ds[0].shape[0]

    # Attacker has access to half of the test set, no overlap
    pop_ds_attacker = (pop_ds[0][:pop_ds_size//2], pop_ds[1][:pop_ds_size//2])

    # Defender has access to half of the test set, no overlap 
    pop_ds_defender = (pop_ds[0][pop_ds_size//2:], pop_ds[1][pop_ds_size//2:])


    # Sampling probs start as uniform, but are modified with upsampling
    sample_probs = [1 / NUM_USERS] * NUM_USERS

    # Run federated learning
    for r in range(NUM_ROUNDS):
        print("round", r)

        # Accumulation list for client model weights for this round
        client_wt_li = []

        # Store sample probs for current round
        per_round_selection_probs.append(deepcopy(sample_probs))

        # Calculate selection probs argmax of current round
        current_round_argmax_sample_probs = np.argsort(sample_probs)[:NUM_CLIENTS]

        # Store selection probs argmax of current round
        per_round_selection_probs_argmax.append(current_round_argmax_sample_probs)

        # Store current round target_clients
        per_round_target_users.append(deepcopy(target_clients))

        # Calculate adversary overlap
        adversary_overlap =  len([i for i in target_clients if i in pop_clients])

        # Calculate defender overlap
        defender_overlap =  len([i for i in current_round_argmax_sample_probs if i in pop_clients])

        # Store overlap of adversary's knowledge and actual pop clients
        per_round_adversary_overlap.append(adversary_overlap)

        # Store overlap of defenders's knowledge and actual pop clients
        per_round_defender_overlap.append(defender_overlap)

        # Select NUM_USERS clients for the current round based on sampling probs
        idxs_users = np.random.choice(
            range(NUM_USERS), NUM_CLIENTS, replace=False, p=sample_probs
        )
        print('Clients selected for the current round:', idxs_users)
        
        #Store selected users for current round
        per_round_selected_users.append(deepcopy(idxs_users))

        server_weights = server.global_model.get_weights()

        mean_pop_frac = 0
        clients_used = 0
        clients_attend_protocol = []

        # Simulate the protocol for each client
        for client_ind in idxs_users:
            
            # Skip clients in `target_clients` to simulate dropping
            if client_ind in target_clients:
                print(f"Skipping client: {client_ind}")
                continue

            clients_attend_protocol.append(client_ind)
            x_client, y_client = client_data[client_ind]

            pop_frac = y_client.mean(axis=0)[args.target_class]
            mean_pop_frac += pop_frac

            # Initialize current client
            client = FederatedClient(x_client, y_client)

            # Update the client's model weights.
            # The poisoning clients correspond to the args.poison_count
            # indices after the last population client
            if (args.num_pop_clients <= client_ind) and (client_ind < args.poison_count + args.num_pop_clients):
                print("Poison client:", client_ind)
                client_wt, loss, acc = client.weights_backdoor(
                    server_weights,
                    args.boost_factor
                )

                # Debugging information 
                print('Performance of the current poisoned client on the target data {}'.format(
                    client.local_model.model.evaluate(pop_ds_attacker[0], pop_ds_attacker[1], verbose=0)
                ))
                client.local_model.set_weights(client_wt)
                print('Performance of the current boosed poisoned clients weights on target data: {}'.format(
                    client.local_model.model.evaluate(pop_ds_attacker[0], pop_ds_attacker[1], verbose=0)
                ))

            # Normal client weight update
            else:
                client_wt, loss, acc = client.weights_update(server_weights)

            # Track weights for each client
            client_wt_li.append(client_wt)
            clients_used += 1

        # Aggregate the weights from all clients and update the server weights
        mean_pop_frac /= clients_used
        new_weights = server.global_model.aggregate_weights(
            client_wt_li,
            agg_fn=args.agg_fn,
            **SERVER_AGG_DICT
        )
        server.global_model.set_weights(new_weights)

        # Update the target loss information for each client and the server
        last_performance_defender,last_performance_attacker  = update_information(
            clients_attend_protocol,
            server.global_model.model,
            client_wt_li,
            last_performance_attacker,
            last_performance_defender,
            pop_ds_attacker[0],
            pop_ds_attacker[1],
            pop_ds_defender[0],
            pop_ds_defender[1],
            normalize_performance=False
        )

        # Evaluate the server at the current round
        acc = server.score(tst_x, tst_y)
        perf_pop = server.score(pop_ds_defender[0], pop_ds_defender[1])
        mean_improved_li.append(perf_pop)
        mean_pop_frac_li.append(mean_pop_frac)
        accuracy_li.append(acc)
        print(
            'Server loss/accuracy on test data, target class data, '
            'and fraction of target class clients: {} {} {}'.format(
                acc, perf_pop, mean_pop_frac)
        )

        # Create the droplist of clients if round >= args.drop_epoch
        if args.drop_epoch != -1 and r >= args.drop_epoch:
            target_clients, visible_target_clients = create_dropli(
                args.drop_count,
                network_dict,
                pop_clients,
                visibility=visibility
            )

        # Update the sampling probabilties if round >= args.upsample_epoch
        if args.upsample_epoch != -1 and r >= args.upsample_epoch:
            sample_probs = upsample(
                server_dict,
                args.upsample_ct,
                target_clients
            )

            # Show the clients that will be upsampled, as they have higher than
            # uniform sampling probability
            upsample_inds = [
                i for (i, p) in enumerate(sample_probs) if p > (1 / NUM_USERS)
            ]
            print('Clients to upsample:', upsample_inds)

        if visibility is not None and args.drop_epoch != -1 and r >= args.drop_epoch:
            target_clients = visible_target_clients

        # Save the loss information dictionaries for the current round
        per_round_agg_improvements_attacker.append(deepcopy(agg_improvements_attacker))
        per_round_agg_improvements_defender.append(deepcopy(agg_improvements_defender))
        per_round_each_improvements_attacker.append(deepcopy(each_improvements_attacker))
        per_round_each_improvements_defender.append(deepcopy(each_improvements_defender))

        # Periodic data dump
        if r % 50 == 0:
            print("improvements", mean_improved_li)
            print("pop frac", mean_pop_frac_li)  
            print("acc", accuracy_li)

    result_dict = {
        'pop_frac': mean_pop_frac_li,
        'accs': accuracy_li,
        'pop_accs': mean_improved_li,
        'pop_clients': pop_clients,  
        'target_clients': target_clients,
        'per_round_agg_improvements_attacker': per_round_agg_improvements_attacker,
        'per_round_agg_improvements_defender': per_round_agg_improvements_defender,
        'per_round_each_improvements_attacker': per_round_each_improvements_attacker,
        'per_round_each_improvements_defender': per_round_each_improvements_defender,
        'per_round_selected_users':per_round_selected_users,
        "per_round_target_users":per_round_target_users,
        'per_round_selection_probs':per_round_selection_probs,
        'per_round_selection_probs_argmax':per_round_selection_probs_argmax,
        'per_round_adversary_overlap':per_round_adversary_overlap,
        'per_round_defender_overlap':per_round_defender_overlap,

    }
    if (args.visible_frac == None) and (args.visible_alpha == None):

        npyFilePath = "results_upsample_multitarget_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.npy".format(
                args.target_class,
                args.num_pop_clients,
                args.remove_pop_clients,
                args.drop_count,
                args.drop_epoch,
                args.poison_count,
                args.trial_ind,
                args.agg_fn,
                args.boost_factor,
                args.upsample_epoch,
                args.upsample_ct,
                args.network_knowledge,
                args.server_knowledge
            )
    else:
            npyFilePath = "results_upsample_multitarget_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.npy".format(
                args.target_class,
                args.num_pop_clients,
                args.remove_pop_clients,
                args.drop_count,
                args.drop_epoch,
                args.poison_count,
                args.trial_ind,
                args.agg_fn,
                args.boost_factor,
                args.upsample_epoch,
                args.upsample_ct,
                args.network_knowledge,
                args.server_knowledge,
                args.visible_frac,
                args.visible_alpha

            )
        

    # Save all the results
    np.save(os.path.expanduser(os.path.join(
        common.npy_SaveDir["fashionMnist"],
        "v1",
        npyFilePath
        )),
        result_dict
    )
