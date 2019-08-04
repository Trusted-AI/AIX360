import numpy as np
import os


class CelebADataset():
    """
    Images are based on the CelebA Dataset [#1]_ [#2]_. Specifically, we use a GAN developed by Karras et. al [#3]_ in order to generate
    new images similar to CelebA. We use these generated images in order to also store the latent variables used to generate
    them, which are required for generating pertinent negatives in CEM-MAF [#4]_.

    References:
        .. [#1] `Liu, Luo, Wang, Tang. Large-scale CelebFaces Attributes (CelebA) Dataset. <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_
        .. [#2] `Liu, Luo, Wang, Tang. Deep Learning Face Attributes in the Wild. ICCV. 2015. <https://arxiv.org/pdf/1411.7766.pdf>`_
        .. [#3] `Karras, Aila, Laine, Lehtinen Progressive Growing of GANs for Improved Quality, Stability, and Variation. ICLR. 2018.
           <https://github.com/tkarras/progressive_growing_of_gans>`_
        .. [#4] `Luss, Chen, Dhurandhar, Sattigeri, Shanmugam, Tu. Generating Contrastive Explanations with Monotonic Attribute Functions.
           2019. <https://arxiv.org/abs/1905.12698>`_
    """
    def __init__(self, dirpath=None):
        self._dirpath = dirpath
        if not self._dirpath:
            self._dirpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                '..', 'data','celeba_data')

    def get_img(self, img_id):
        return np.load(os.path.join(self._dirpath, "{}_img.npy".format(img_id)))

    def get_latent(self, img_id):
        return np.load(os.path.join(self._dirpath, "{}_latent.npy".format(img_id))).astype("float32")
