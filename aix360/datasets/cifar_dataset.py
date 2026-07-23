import numpy as np
import json
import sys,os
import hashlib
import urllib.request
import tarfile
import pickle as cp
from sklearn.preprocessing import OneHotEncoder
import shutil

# SHA-256 of the authentic CIFAR-10 python archive published at
# https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# Verified against the vendor-published MD5 (c58f30108f718f92721af3b95e74349a)
# before computing this digest. The archive is deserialized with pickle, so
# extraction/processing only proceeds when the download matches this pin;
# this prevents a tampered/MITM'd archive from executing arbitrary code
# (CWE-502) or writing outside the target directory via crafted tar members
# (CWE-22).
CIFAR10_ARCHIVE_SHA256 = '6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce'


def _onehot_dense(labels):
    """One-hot encode ``labels`` returning a dense array, across scikit-learn
    versions (``sparse`` was renamed to ``sparse_output`` in sklearn 1.2)."""
    try:
        encoder = OneHotEncoder(sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(sparse=False)
    return encoder.fit_transform(labels)


def _sha256_of_file(path, chunk_size=1024 * 1024):
    """Return the hex SHA-256 digest of the file at ``path``."""
    hasher = hashlib.sha256()
    with open(path, 'rb') as fileobj:
        for chunk in iter(lambda: fileobj.read(chunk_size), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def _is_within_directory(directory, target):
    """Return True if ``target`` resolves to a path inside ``directory``.

    Compares on a path-separator boundary (not a raw string prefix) so that a
    sibling like ``/a/bad`` is not treated as being inside ``/a/b``.
    """
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)
    if abs_target == abs_directory:
        return True
    return abs_target.startswith(abs_directory + os.sep)


def _safe_extract(tar, dirpath):
    """Extract ``tar`` into ``dirpath``, rejecting any member that would
    escape the target directory (tar-slip / path traversal, CWE-22).

    Uses the tarfile 'data' filter (Python 3.12+) as the primary defense and
    adds an explicit per-member containment check for defense-in-depth.
    """
    for member in tar.getmembers():
        member_path = os.path.join(dirpath, member.name)
        if not _is_within_directory(dirpath, member_path):
            raise Exception(
                "Attempted path traversal in tar archive: {}".format(member.name))
        # Reject non-regular members that can be abused (e.g. symlinks/links
        # pointing outside the target directory).
        if member.issym() or member.islnk():
            link_target = os.path.join(os.path.dirname(member_path), member.linkname)
            if not _is_within_directory(dirpath, link_target):
                raise Exception(
                    "Attempted link traversal in tar archive: {}".format(member.name))
    try:
        # 'data' filter (Python 3.12+) blocks absolute paths, traversal,
        # links escaping the destination, and unsafe permission/device members.
        tar.extractall(dirpath, filter='data')
    except TypeError:
        # Older Python without the filter argument: the manual checks above
        # have already validated every member.
        tar.extractall(dirpath)


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

            # Verify the archive integrity before trusting its contents. The
            # batch files are loaded with pickle, so a tampered/MITM'd archive
            # could execute arbitrary code (CWE-502) or write files outside the
            # target directory (CWE-22). Fail closed: on any mismatch, remove
            # the untrusted archive and abort.
            actual_sha256 = _sha256_of_file(full_name)
            if actual_sha256 != CIFAR10_ARCHIVE_SHA256:
                os.remove(full_name)
                raise Exception(
                    "SHA-256 mismatch for {}: expected {}, got {}. "
                    "Refusing to extract an untrusted archive.".format(
                        name, CIFAR10_ARCHIVE_SHA256, actual_sha256))

            #now extract the files
            #print("extracting files")
            # extract with tar-slip protection (validated members only)
            tar = tarfile.open(full_name, "r:gz")
            try:
                _safe_extract(tar, self._dirpath)
            except Exception:
                tar.close()
                # remove any partially-extracted contents and the archive
                extracted_dir = os.path.join(self._dirpath, 'cifar-10-batches-py')
                if os.path.exists(extracted_dir):
                    shutil.rmtree(extracted_dir)
                if os.path.exists(full_name):
                    os.remove(full_name)
                raise
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
        
        y_train=_onehot_dense(y_train).astype('uint8')
                
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
            y_test=_onehot_dense(y_test).astype('uint8')
        
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
