        
import os

import numpy as np

# utility to count the number of lines in a fiel
def wc(
    filename: str,
) -> int:
    return sum(1 for _ in open(filename, 'r', encoding='utf-8'))

class Glove:

    """
    NLP Model 
    Glove is a wrapper for loading pretrained word 
    embedding vectors hosted at http://nlp.stanford.edu/data/glove.6B.zip.

    References:
        .. [#] `Pennington, Jeffrey, Richard Socher, and Christopher Manning,
            “Glove: Global Vectors for Word Representation”,
            In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)`_
    """

    def __init__(
        self,
        dim : int = 300,
    ):
        """
        Initialize Glove 

        Args:
            dim (int): dimension of the emebddings (50, 100, 200, 300)
        """

        self._file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'data', f'glove.6B.{dim}d.txt'
        )
        self._dim = dim
        self._vocab_size = wc(self._file)

        self._embeddings = np.zeros(
            (self._vocab_size, self._dim), 
            dtype = float,
        )
        self._vocab = {}

        # do this to count the number of lines
        with open(self._file, encoding='utf-8') as f:

            for i in range(self._vocab_size):
                # word, .. vals ...
                line = f.readline().strip().split()
                self._vocab[line[0]] = i
                self._embeddings[
                    i
                ] = [float(x) for x in line[1:]]

    @property
    def vocab(self):
        """
        Returns a dictionary mapping word to embdding vector index 
        """

        return self._vocab

    def embedding(self, word: str):

        """
        Embedding Vector

        Args:
            word (str): a word for which the corresponding embedding is desired

        Returns:
            a numpy float array of above dimension

        raises ValueError: if word is not in the vocab
        """

        if word not in self._vocab:
            raise ValueError(f'word \'{word}\' not in vocab')

        return self._embeddings[
            self._vocab[word]
        ]
