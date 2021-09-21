import unittest
import os
import shutil

from aix360.algorithms.TracInF import TracInFExplainer
import numpy as np


class TestTracInFExplainer(unittest.TestCase):

    def test_TracInF(self):
        """
        future test case should probably not train a model it self;
        instead use a pretrained model and check classification with set example input
        """

        data_dir = '../../aix360/data/'
        train_data_name = data_dir + 'model_5_train_w_qa.json'
        dev_data_name = data_dir + 'model_5_val_w_qa.json'
        model_output = 'model_5_train_w_qa.out-otc'
        max_seq_length = 128
        batch_size = 128
        epochs = 1
        BERT_name = 'roberta-large'

        # returns accuracy, default is 0, so if return is greater 0 training has occurred and a model was saved
        result = TracInFExplainer.explain_instance(max_seq_length, BERT_name, data_dir, train_data_name, dev_data_name, batch_size, epochs, model_output)
        self.assertGreater(result, 0)



if __name__ == '__main__':
    unittest.main