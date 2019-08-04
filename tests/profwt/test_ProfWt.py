
import unittest
import sys
import numpy as np
import os

sys.path.append(os.getcwd())
from aix360.algorithms.profwt import prof_weight_compute


class Testprofwexplainer(unittest.TestCase):

    def test_prof_weight_compute(self):


        """ Test Prof Weight Computation """

        ## Create a fixed probability matrix (samples=3,num_classes=9) and rotate it 3 times along the columns to generate 3 layers.
        ## The Labels are appropriately generated such that the prof_weight computation would compute the normalized sum of the first 3 probabilities for the 
        ## first sample, the next 3 probabilities for the second sample, the next 3 probabilities for the third sample etc..

        ## Generate a random probability vector and replicate it 3 times.
        a=np.random.random_sample((1,9))
        a=a/np.sum(a) 
        data_matrix=np.outer(np.ones((3,1)),a)
        np.save("./test_case_data_1",data_matrix)
        np.save("./test_case_data_2",np.roll(data_matrix,-1,axis=1))
        np.save("./test_case_data_3",np.roll(data_matrix,-2,axis=1))


        y=np.zeros((1,9))
        y[0,0]=1
        y1=np.roll(y,3,axis=1)
        y2=np.roll(y1,3,axis=1)
        Y=np.vstack((np.vstack((y,y1)),y2))

        np.save("./test_case_train_labels",Y)



        # Load the probes already stored numbered by layers.
        list_probe_filenames=['./test_case_data_'+str(x)+'.npy' for x in range(1,4)]
        print(list_probe_filenames)
        # Load the label corresponding to this
        train_label_path='./test_case_train_labels.npy'
        y_train=np.load(train_label_path)

        #Compute Prof-Weight by calling prof_weight_compute function
        start_layer=0
        final_layer=2
        w=prof_weight_compute(list_probe_filenames,start_layer,final_layer,y_train)
        w=w.reshape(w.shape[0],)
        self.assertTrue( (abs(3*w[0]-np.sum(a[0,0:3]))<0.0001) & (abs(3*w[1]-np.sum(a[0,3:6]))<0.0001) & (abs(3*w[2]-np.sum(a[0,6:9]))<0.0001), "Weight Computation has an error")

        print("prof_weight_compute Test passed..")
        #Delete the npy files created
        print("Removing Created test files....")
        if os.path.isfile(train_label_path):
            os.remove(train_label_path)
        for f in list_probe_filenames:
            if os.path.isfile(f):
                os.remove(f)



if __name__ == '__main__':
    unittest.main()
