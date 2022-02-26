import numpy as np
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
from aix360.datasets.SIF_dataset import DataSetTS
from aix360.algorithms.sif.SIF_NN import AllRNN, AllLSTM, AllAR
from aix360.algorithms.sif.SIF_utils import get_contaminate_series
import unittest
import argparse
import os
import pickle


class SIF(unittest.TestCase):
    def get_model_train_ar2(self, data_sets, series, timesteps, w=None, gammas=None, num_train_steps=20000,
                            model_dir=None):
        initial_learning_rate = 0.01
        decay_epochs = [10000, 20000]
        batch_size = 20
        n_sample = series.shape[1] if len(series.shape) > 1 else 1

        # model can be changed to AllLSTM or AllRNN model defined in SIF_NN.py
        model = AllAR(
            time_steps=timesteps,
            x_dim=n_sample,
            y_dim=n_sample,
            share_param=True,
            batch_size=batch_size,
            time_series=series,
            data_sets=data_sets,
            initial_learning_rate=initial_learning_rate,
            damping=1e-3,
            decay_epochs=decay_epochs,
            mini_batch=False,
            train_dir='arma_output',
            log_dir='logs',
            model_name='ar_test',
            calc_hessian=False,
            w=w,
            gammas=gammas,
        )
        if model_dir is not None:
            print('Loading pre-trained model......')
            model.restore(model_dir)
        else:
            model.train(num_steps=num_train_steps, iter_to_switch_to_batch=10, iter_to_switch_to_sgd=10)
        return model

    def test_SIF(self):
        parser = argparse.ArgumentParser(description='RGCN')
        parser.add_argument('-fast_mode', dest='fast_mode', action='store_true')
        parser.set_defaults(fast_mode=True)
        args = parser.parse_args()

        # parameters
        timesteps = 2
        np.random.seed(1)
        n_sample = 1000
        n_time_stamp = 100
        gammas = np.arange(0.0, 0.09, 0.01)
        data_dir = '../../aix360/data/sif_data/data.pkl'
        model_dir = '../../aix360/models/sif/ar2'

        # Skip generating the synthetic dataset and training the model.
        # In the fast mode, we directly load the saved dataset and pre-trained model
        if args.fast_mode:
            assert os.path.exists(data_dir), "Could not find the data.pkl in {}".format(data_dir)
            # load time series from data.pkl file
            series = pickle.load(open(data_dir, "rb"))
            data_sets = DataSetTS(np.arange(len(series)), np.array(['Y']), None, None, None,
                                  lag=timesteps).datasets_gen_rnn()
            # initialize and train the model which takes the clean time sequence as input and makes prediction
            model = self.get_model_train_ar2(data_sets, series, timesteps, gammas=gammas, model_dir=model_dir)
        else:
            # ar and ma are two parameters controlling the synthetic time sequence data
            ar = np.r_[1, -np.array([0.7, -0.3])]
            ma = np.r_[1, -np.array([0.1])]

            # generate the core process or clean time sequence data
            series = [smt.arma_generate_sample(ar, ma, n_time_stamp) for i in range(n_sample)]
            series = np.vstack(series)
            pickle.dump(series, open(data_dir, "wb"))
            data_sets = DataSetTS(np.arange(len(series)), np.array(['Y']), None, None, None,
                                  lag=timesteps).datasets_gen_rnn()
            # initialize and train the model which takes the clean time sequence as input and makes prediction
            model = self.get_model_train_ar2(data_sets, series, timesteps, gammas=gammas, model_dir=None)
            model.save(model_dir)

        # generate the contaminating process
        y_contaminate = np.random.randn(n_sample)

        # insert the contaminated data into the clean time sequence data
        contaminated_series = get_contaminate_series(series, y_contaminate, data_sets.train.labels)

        # plot contaminated series
        plt.plot(contaminated_series)
        plt.savefig('arma')
        plt.close()

        # compute SIF value
        model.update_configure(y_contaminate, gammas)
        # sif = model.get_sif(y_contaminate, timesteps, 1, None, 30, verbose=False)
        sif = model.explain_instance(y_contaminate, timesteps, 1, None, 30, verbose=False)
        print('SIF = {}'.format(sif))


if __name__ == '__main__':
    unittest.main()
