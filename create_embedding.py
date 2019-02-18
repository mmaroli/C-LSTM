import gensim
import numpy as np

class Embedding:
    def __init__(self, pretrained_word2vec_embedding='GoogleNews-vectors-negative300.bin'):
        print("loading word vectors...")
        self.model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_word2vec_embedding, binary=True)

    def get_word_vector(self, word):
        """ Returns word vector for given word
            word_embedding from pretrained word2vec vectors on 100B Google
            News Dataset with dimensionality of 300
        """
        if word in self.model.wv:
            return self.model.wv[word]
        else: # initialize from random uniform distribution from [-0.25, 0.25)
            return np.zeros(shape=300, dtype=np.float32)
