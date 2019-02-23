import json
import pandas as pd
import tensorflow as tf
from create_embedding import Embedding
import os


def create_embedding_tensor(word_vectors, text, maxlen):
    """ Returns embedding tensor of shape (maxlen, 300)
        If number of words is less than maxlen, pad with special
        character: *
        If number of words is more than maxlen, cut off at maxlen
    """
    tokenized = text.split()
    if ( len(tokenized) > maxlen): # cut words off
        tokenized = tokenized[:maxlen]
        tensor_arr = []
        for word in tokenized:
            tensor_arr.append(tf.constant(word_vectors.get_word_vector(word)))
        return tf.stack(tf.cast(tensor_arr, dtype=tf.float32))
    else: # pad with special character
        tensor_arr = []
        for word in tokenized:
            tensor_arr.append(tf.constant(word_vectors.get_word_vector(word)))
        paddings = [ [0, maxlen - len(tensor_arr)], [0,0] ] # pads to maxlen rows with 0's
        return tf.pad(tf.stack(tf.cast(tensor_arr, dtype=tf.float32)), paddings, 'CONSTANT')



def get_review_data(word_vectors, filename):
    """ Gets data from Yelp Dataset and creates embedding matrix
        of size (maxlen x 300)
        args:
            word_vectors: Embedding class for word2vec vectors
            filename: filename for yelp reviews
        returns:
            Tensor of shape (num_reviews, maxlen, 300)
    """
    texts = []
    labels = []
    maxlen = 50
    with open(filename) as f:
        review_embeddings = []
        for i, line in enumerate(f):
            review = json.loads(line)

            review_embedding = create_embedding_tensor(word_vectors, review['text'], maxlen)
            review_embeddings.append(review_embedding)

            texts.append(review['text'])
            labels.append(review['stars'])

    review_data = tf.stack(review_embeddings)
    return review_data, labels

def train_dev_test_split(review_json):
    """
    Splits file into train, dev, and test review files
        args:
            review_json: review.json file for yelp reviews
        returns:
            returns nothing; write to files
                train_file: review_train.json
                dev_file: review_dev.json
                test_file: review_test.json file
    """
    print("splitting file into train, dev, test ...")
    train_perc = 0.7
    dev_perc = 0.2
    test_perc = 0.1
    if os.path.exists('review_train.json'):
        os.remove('review_train.json')
    if os.path.exists('review_dev.json'):
        os.remove('review_dev.json')
    if os.path.exists('review_test.json'):
        os.remove('review_test.json')
    with open(review_json) as f:
        for i, line in enumerate(f):
            if (i < 10000):
                with open('review_train.json', 'a') as out:
                    out.write(line)
            elif (i >= 10000 and i < 13000):
                with open('review_dev.json', 'a') as out:
                    out.write(line)
            elif (i >= 13000 and i < 15000):
                with open('review_test.json', 'a') as out:
                    out.write(line)


def get_train_data():
    """ Returns review_data and labels
        returns:
            review_data: train_data tensor of shape (num_train_reviews, maxlen, 300)
            labels: labels of shape (num_train_reviews)
    """
    train_dev_test_split('review.json')
    word_vectors = Embedding()
    review_data, labels = get_review_data(word_vectors, 'review_train.json')

    return review_data, labels

def get_dev_data():
    """ Returns dev set review_data and labels
        returns:
            review_data: dev_data tensor of shape (num_dev_reviews, maxlen, 300)
            labels: labels of shape (num_dev_reviews)
    """
    pass
