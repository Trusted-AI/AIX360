import numpy as np
import json
import sys,os
import urllib.request
import tarfile
import pickle as cp
from sklearn.preprocessing import OneHotEncoder
import shutil

class CIFARDataset():
    """
    The CIFAR-10 dataset [#]_ consists of 60000 32x32 color images. Target variable is one amongst 10 classes. The dataset has
    6000 images per class. There are 50000 training images and 10000 test images. The classes are: airplane, automobile,
    bird, cat, deer, dog, frog, horse, ship ,truck. We further divide the training set into train1 (30000 samples) and
    train2 (20,000 samples). For ProfWt, the complex model is trained on train1 while the simple model is trained on train2.

    References:
        .. [#] `Krizhevsky, Hinton. Learning multiple layers of features from tiny images. Technical Report, University of
           Toronto 1 (4), 7. 2009 <https://www.cs.toronto.edu/~kriz/cifar.html>`_
    """

    def __init__(self, dirpath=None):
        self._dirpath = dirpath
        if not self._dirpath:
            self._dirpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                '..', 'data','cifar_data')
        self._download_data()

    def _download_data(self):
        name = 'cifar-10-python.tar.gz'
        json_file_name = 'cifar-10-train1-image.json'
        full_name = os.path.join(self._dirpath, name)
        if not os.path.exists(os.path.join(self._dirpath, json_file_name)):
            if not os.path.exists(full_name):
                print("retrieving file", name)
                urllib.request.urlretrieve('https://www.cs.toronto.edu/~kriz/' + name, full_name)
                print("retrieved")
            #now extract the files
            #print("extracting files")
            tar = tarfile.open(full_name, "r:gz")
            tar.extractall(self._dirpath)
            tar.close()        
            #print("extracted files")
            
            self._process_data()

            #now cleanup
            if os.path.exists(full_name):
                os.remove(full_name)
            
    def _process_data(self):
        image_size=32
        num_classes=10
        per_file_size=10000
        
        print("processing files...")
        datafile_path = os.path.join(self._dirpath, 'cifar-10-batches-py')
        for i in range(5):
            with open(os.path.join(datafile_path,'data_batch_'+str(i+1)), 'rb') as fileobj:
                dictionary = cp.load(fileobj, encoding='bytes')
                dum=dictionary[b'data'].reshape((per_file_size,3,32,32))
                dum_1=np.transpose(dum,(0,2,3,1)).astype('uint8')
                lab=np.asarray(dictionary[b'labels']).reshape((per_file_size,1))
                if i==0:
                    x_train=dum_1
                    y_train=lab     
                else:
                    x_train=np.concatenate((x_train,dum_1),0)
                    y_train=np.concatenate((y_train,lab),0)
        
        y_train=OneHotEncoder(sparse=False).fit_transform(y_train).astype('uint8')    
                
        assert x_train.shape==(5*per_file_size,image_size,image_size,3)
        assert y_train.shape==(5*per_file_size,num_classes)
        
        x_train.astype(float)/255
        
        x_train_1=x_train[0:30000,:,:,:]
        x_train_2=x_train[30000:,:,:,:]
        y_train_1=y_train[0:30000,:]
        y_train_2=y_train[30000:,:]
        
        
        with open(os.path.join(datafile_path,'test_batch'), 'rb') as fileobj:
            dictionary = cp.load(fileobj, encoding='bytes')
            x_test=dictionary[b'data'].reshape((per_file_size,3,32,32))
            x_test=np.transpose(x_test,[0,2,3,1]).astype('uint8')
            y_test=np.asarray(dictionary[b'labels']).reshape((per_file_size,1))
            y_test=OneHotEncoder(sparse=False).fit_transform(y_test).astype('uint8')
        
        with open(os.path.join(self._dirpath,'cifar-10-train1-image.json'),'w') as outfile:
            print("writing ",os.path.join(self._dirpath,'cifar-10-train1-image.json'))
            json.dump(x_train_1.tolist(),outfile)
        outfile.close()
        
        with open(os.path.join(self._dirpath,'./cifar-10-train2-image.json'),'w') as outfile:
            print("writing ",os.path.join(self._dirpath,'cifar-10-train2-image.json'))
            json.dump(x_train_2.tolist(),outfile)
        outfile.close()
        
        with open(os.path.join(self._dirpath,'./cifar-10-test-image.json'),'w') as outfile:
            print("writing ",os.path.join(self._dirpath,'cifar-10-test-image.json'))
            json.dump(x_test.tolist(),outfile)
        outfile.close()
        
        with open(os.path.join(self._dirpath,'./cifar-10-train1-label.json'),'w') as outfile:
            print("writing ",os.path.join(self._dirpath,'cifar-10-train1-label.json'))
            json.dump(y_train_1.tolist(),outfile)
        outfile.close()
        
        with open(os.path.join(self._dirpath,'./cifar-10-train2-label.json'),'w') as outfile:
            print("writing ",os.path.join(self._dirpath,'cifar-10-train2-label.json'))
            json.dump(y_train_2.tolist(),outfile)
        outfile.close()
        
        with open(os.path.join(self._dirpath,'./cifar-10-test-label.json'),'w') as outfile:
            print("writing ",os.path.join(self._dirpath,'cifar-10-test-label.json'))
            json.dump(y_test.tolist(),outfile)
        outfile.close()
        print("processing completed")
        #cleanup
        if os.path.exists(datafile_path):
            shutil.rmtree(datafile_path)
        

        
                        
    def load_file(self, filename):
        try:
            with open(os.path.join(self._dirpath, filename)) as file:
                data=json.load(file)
            file.close()
        except IOError as err:
            print("IOError: {}".format(err))
            sys.exit(1)
        return np.array(data)
