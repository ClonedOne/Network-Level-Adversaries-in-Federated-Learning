"""
Create the DBPedia dataset numpy files using Huggingface's datasets
and transformers libraries.

WARNING: THIS SCRIPT HAS TO BE RUN WITH TENSORFLOW >= 2.6. TESTED
ONLY ON TENSORFLOW 2.7.0.

# This code will tokenize the samples using bert-tiny tokenizer.

https://huggingface.co/datasets/dbpedia_14
https://rdrr.io/cran/textdata/man/dataset_dbpedia.html
Classes are: 
    Company
    EducationalInstitution
    Artist
    Athlete
    OfficeHolder
    MeanOfTransportation
    Building
    NaturalPlace
    Village
    Animal
    Plant
    Album
    Film
    WrittenWork
"""

from collections import Counter

import datasets
import numpy as np
import tensorflow as tf

# from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


import common

# Change this to change the tokenizer max length
# MAX_LEN = 256
MAX_LEN = 200

# Load data from Huggingface's datasets
dbpedia_ds = datasets.load_dataset('dbpedia_14', cache_dir=common.hf_cache)
print('Loaded dataset:\n', dbpedia_ds)

print('Classes in the dataset:', np.unique(dbpedia_ds['train']['label']))
n_classes = common.num_classes['dbpedia']

print('Distribution of classes in train/test:\n')
print(Counter(dbpedia_ds['train']['label']).most_common())
print(Counter(dbpedia_ds['test']['label']).most_common())

# USING BERT-TINY TOKENIZER
# Load tokenizer and create tokenized vectors
# tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
# def tokenize_function(examples):
#     return tokenizer(examples['content'], padding="max_length", truncation=True)

# # Impose length of MAX_LEN tokens -- bert-tiny doesn't specify one
# tokenizer.model_max_length = MAX_LEN

# dbpedia_tk = dbpedia_ds.map(tokenize_function, batched=True)

# trn_x_all = np.array(dbpedia_tk['train']['input_ids'])
# tst_x = np.array(dbpedia_tk['test']['input_ids'])
# trn_y_all = np.array(dbpedia_ds['train']['label'])
# tst_y = np.array(dbpedia_ds['test']['label'])

# USING TextVectorization TOKENIZER
vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=MAX_LEN)

train_data = dbpedia_ds['train']['content']
test_data = dbpedia_ds['test']['content']
train_labels = dbpedia_ds['train']['label']
test_labels = dbpedia_ds['test']['label']

trn_y_all = np.array(train_labels)
tst_y = np.array(test_labels)

text_ds = tf.data.Dataset.from_tensor_slices(train_data).batch(128)
vectorizer.adapt(text_ds)

# Create the vocabulary
voc = vectorizer.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))

# Create the embedding matrix
embeddings_index = {}
with open(common.glove_path, 'r') as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))

num_tokens = len(voc) + 2
embedding_dim = 100
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))

np.save(common.dbpedia_embedding_matrix_path, embedding_matrix)

trn_x_all = vectorizer(np.array([[s] for s in train_data])).numpy()
tst_x = vectorizer(np.array([[s] for s in test_data])).numpy()

# We will take 10K samples for each class, for a totalk of 114K samples.
trn_x, _, trn_y, _ = train_test_split(
    trn_x_all, 
    trn_y_all,
    train_size=n_classes * 25000,
    random_state=42,
    stratify=trn_y_all
)

print('Shapes and distributions after sampling')
print(trn_x.shape, trn_y.shape, tst_x.shape, tst_y.shape)
print(Counter(trn_y).most_common())
print(Counter(tst_y).most_common())

# Save the numpy arrays
np.save(common.dbpedia_trn_x_pth, trn_x)
np.save(common.dbpedia_trn_y_pth, trn_y)
np.save(common.dbpedia_tst_x_pth, tst_x)
np.save(common.dbpedia_tst_y_pth, tst_y)
