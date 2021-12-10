"""
Common constants and utilities used by all modules
"""

# PATHS

# TODO:
# # Change theese to point to the federated EMNIST data files
# fed_emnist_train = 'path to h5 file train
# fed_emnist_test = 'path to h5 file test'

# TODO:
# # Change this to point to the cache directory for Huggingface
# hf_cache = 'path/to/project/dir/text_data'

REPO_DIR = '' #must end with /
ENV_DIR  = '' #must end with /
DATA_DIR = f'{REPO_DIR}data/'
RESULT_DIR = f'{REPO_DIR}experimentResults/'
LOG_DIR = f'{REPO_DIR}experimentLogs/'
# Glove embeddings
glove_path = f'{REPO_DIR}glove.6B.100d.txt'

# Change theese to point to desired path for generated numpy files
emnist_trn_x_pth = f'{DATA_DIR}trn_x_emnist.npy'
emnist_trn_y_pth = f'{DATA_DIR}trn_y_emnist.npy'
emnist_tst_x_pth = f'{DATA_DIR}tst_x_emnist.npy'
emnist_tst_y_pth = f'{DATA_DIR}tst_y_emnist.npy'

fashionMnist_trn_x_pth = f'{DATA_DIR}trn_x_fashionMnist.npy'
fashionMnist_trn_y_pth = f'{DATA_DIR}trn_y_fashionMnist.npy'
fashionMnist_tst_x_pth = f'{DATA_DIR}tst_x_fashionMnist.npy'
fashionMnist_tst_y_pth = f'{DATA_DIR}tst_y_fashionMnist.npy'


dbpedia_trn_x_pth = f'{DATA_DIR}trn_x_dbpedia.npy'
dbpedia_trn_y_pth = f'{DATA_DIR}trn_y_dbpedia.npy'
dbpedia_tst_x_pth = f'{DATA_DIR}tst_x_dbpedia.npy'
dbpedia_tst_y_pth = f'{DATA_DIR}tst_y_dbpedia.npy'
dbpedia_embedding_matrix_path = f'{DATA_DIR}dbpedia_embedding_matrix.npy'

# Change theese to point to desired results paths
npy_SaveDir = {
    'base':         f'{RESULT_DIR}',
    'emnist':       f'{RESULT_DIR}emnist',
    'fashionMnist': f'{RESULT_DIR}fashionMnist',
    'dbpedia':      f'{RESULT_DIR}dbpedia',
}

npy_LogDir = {
    'base':         f'{LOG_DIR}',
    'emnist':       f'{LOG_DIR}emnist',
    'fashionMnist': f'{LOG_DIR}fashionMnist',
    'dbpedia':      f'{LOG_DIR}dbpedia',
}


version = {
    'base':'v1',
    'emnist': 'v1',
    'fashionMnist': 'v1',
    'dbpedia': 'v1',
}

network = {
    'workDir' :f"{REPO_DIR}src",
    'repoDir' :f"{REPO_DIR}",
    'environmentPath' : f"source {ENV_DIR}bin/activate",
    'bin' : f". {ENV_DIR}bin/activate",
    'group':  ['localhost']
}

execfile = {
    'emnist':       f'cd {network["repoDir"]} && {network["bin"]} && python {REPO_DIR}src/nlafl/main_emnist_upsample_multitarget.py ',
    'fashionMnist': f'cd {network["repoDir"]} && {network["bin"]} && python {REPO_DIR}src/nlafl/main_fashionMnist_upsample_multitarget.py ',
    'dbpedia':      f'cd {network["repoDir"]} && {network["bin"]} && python {REPO_DIR}src/nlafl/main_dbpedia_upsample_multitarget.py ',
}
# NUMERICAL CONSTANTS

# Number of classes for each dataset
num_classes = {
    'emnist': 10,
    'fashionMnist':10,
    'dbpedia': 14
}
