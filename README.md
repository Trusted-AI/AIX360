# AI Explainability 360 (v0.3.0) 

[![Build](https://github.com/Trusted-AI/AIX360/actions/workflows/Build.yml/badge.svg)](https://github.com/Trusted-AI/AIX360/actions/workflows/Build.yml)
[![Documentation Status](https://readthedocs.org/projects/aix360/badge/?version=latest)](https://aix360.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/aix360.svg?)](https://badge.fury.io/py/aix360)

The AI Explainability 360 toolkit is an open-source library that supports interpretability and explainability of datasets and machine learning models. The AI Explainability 360 Python package includes a comprehensive set of algorithms that cover different dimensions of explanations along with proxy explainability metrics. The AI Explainability 360 toolkit supports tabular, text, images, and time series data. 

The [AI Explainability 360 interactive experience](https://aix360.res.ibm.com/) provides a gentle introduction to the concepts and capabilities by walking through an example use case for different consumer personas. The [tutorials and example notebooks](./examples) offer a deeper, data scientist-oriented introduction. The complete API is also available. 

There is no single approach to explainability that works best. There are many ways to explain: data vs. model, directly interpretable vs. post hoc explanation, local vs. global, etc. It may therefore be confusing to figure out which algorithms are most appropriate for a given use case. To help, we have created some [guidance material](https://aix360.res.ibm.com/resources#guidance) and a [taxonomy tree](./aix360/algorithms/README.md) that can be consulted. 

We have developed the package with extensibility in mind. This library is still in development. We encourage you to contribute your explainability algorithms, metrics, and use cases. To get started as a contributor, please join the [AI Explainability 360 Community on Slack](https://aix360.slack.com) by requesting an invitation [here](https://join.slack.com/t/aix360/shared_invite/enQtNzEyOTAwOTk1NzY2LTM1ZTMwM2M4OWQzNjhmNGRiZjg3MmJiYTAzNDU1MTRiYTIyMjFhZTI4ZDUwM2M1MGYyODkwNzQ2OWQzMThlN2Q). Please review the instructions to contribute code and python notebooks [here](CONTRIBUTING.md).

## Supported explainability algorithms

### Data explanations

- ProtoDash ([Gurumoorthy et al., 2019](https://arxiv.org/abs/1707.01212))
- Disentangled Inferred Prior VAE ([Kumar et al., 2018](https://openreview.net/forum?id=H1kG7GZAW))

### Local post-hoc explanations 

- ProtoDash ([Gurumoorthy et al., 2019](https://arxiv.org/abs/1707.01212))
- Contrastive Explanations Method ([Dhurandhar et al., 2018](https://papers.nips.cc/paper/7340-explanations-based-on-the-missing-towards-contrastive-explanations-with-pertinent-negatives))
- Contrastive Explanations Method with Monotonic Attribute Functions ([Luss et al., 2019](https://arxiv.org/abs/1905.12698))
- Exemplar based Contrastive Explanations Method
- Grouped Conditional Expectation (Adaptation of Individual Conditional Expectation Plots by [Goldstein et al.](https://arxiv.org/abs/1309.6392) to higher dimension )
- LIME ([Ribeiro et al. 2016](https://arxiv.org/abs/1602.04938),  [Github](https://github.com/marcotcr/lime))
- SHAP ([Lundberg, et al. 2017](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions),  [Github](https://github.com/slundberg/shap))

### Time-Series local post-hoc explanations

- Time Series Saliency Maps using Integrated Gradients (Inspired by [Sundararajan et al.](https://arxiv.org/pdf/1703.01365.pdf) )
- Time Series LIME (Time series adaptation of the classic paper by [Ribeiro et al. 2016](https://arxiv.org/abs/1602.04938) )
- Time Series Individual Conditional Expectation (Time series adaptation of Individual Conditional Expectation Plots [Goldstein et al.](https://arxiv.org/abs/1309.6392) )

### Local direct explanations

- Teaching AI to Explain its Decisions ([Hind et al., 2019](https://doi.org/10.1145/3306618.3314273)) 
- Order Constraints in Optimal Transport ([Lim et al.,2022](https://arxiv.org/abs/2110.07275), [Github](https://github.com/IBM/otoc))

### Global direct explanations

- Interpretable Model Differencing (IMD) ([Haldar et al., 2023](https://arxiv.org/abs/2306.06473))
- CoFrNets (Continued Fraction Nets) ([Puri et al., 2021](https://papers.nips.cc/paper/2021/file/b538f279cb2ca36268b23f557a831508-Paper.pdf))
- Boolean Decision Rules via Column Generation (Light Edition) ([Dash et al., 2018](https://papers.nips.cc/paper/7716-boolean-decision-rules-via-column-generation))
- Generalized Linear Rule Models ([Wei et al., 2019](http://proceedings.mlr.press/v97/wei19a.html))
- Fast Effective Rule Induction (Ripper) ([William W Cohen, 1995](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.107.2612&rep=rep1&type=pdf))

### Global post-hoc explanations

- ProfWeight ([Dhurandhar et al., 2018](https://papers.nips.cc/paper/8231-improving-simple-models-with-confidence-profiles))


## Supported explainability metrics
- Faithfulness ([Alvarez-Melis and Jaakkola, 2018](https://papers.nips.cc/paper/8003-towards-robust-interpretability-with-self-explaining-neural-networks))
- Monotonicity ([Luss et al., 2019](https://arxiv.org/abs/1905.12698))

## Setup

### Supported Configurations:

| Installation keyword | Explainer(s)      | OS                            | Python version |
| ---------------|---------------| ------------------------------| -------------- |
| cofrnet        |cofrnet        | macOS, Ubuntu, Windows        | 3.10 |
| contrastive    |cem, cem_maf    | macOS, Ubuntu, Windows        | 3.6  |
| dipvae         | dipvae| macOS, Ubuntu, Windows        | 3.10 |
| gce            | gce | macOS, Ubuntu, Windows        | 3.10 |
| imd            | imd | macOS, Ubuntu                 | 3.10 |
| lime           | lime| macOS, Ubuntu, Windows        | 3.10 |
| matching       | matching| macOS, Ubuntu, Windows        | 3.10 |
| nncontrastive  | nncontrastive | macOS, Ubuntu, Windows        | 3.10 |
| profwt         | profwt | macOS, Ubuntu, Windows        | 3.6  |
| protodash      | protodash | macOS, Ubuntu, Windows        | 3.10 |
| rbm            | brcg, glrm            | macOS, Ubuntu, Windows        | 3.10 |
| rule_induction | ripper | macOS, Ubuntu, Windows        | 3.10 |
| shap           | shap | macOS, Ubuntu, Windows        | 3.6  |
| ted            | ted | macOS, Ubuntu, Windows        | 3.10 |
| tsice          | tsice | macOS, Ubuntu, Windows        | 3.10 |
| tslime         | tslime |  macOS, Ubuntu, Windows        | 3.10 |
| tssaliency     | tssaliency | macOS, Ubuntu, Windows        | 3.10 |


### (Optional) Create a virtual environment

AI Explainability 360 requires specific versions of many Python packages which may conflict
with other projects on your system. A virtual environment manager is strongly
recommended to ensure dependencies may be installed safely. If you have trouble installing the toolkit, try this first.

#### Conda

Conda is recommended for all configurations though Virtualenv is generally
interchangeable for our purposes. Miniconda is sufficient (see [the difference between Anaconda and
Miniconda](https://conda.io/docs/user-guide/install/download.html#anaconda-or-miniconda)
if you are curious) and can be installed from
[here](https://conda.io/miniconda.html) if you do not already have it.

Then, create a new python environment based on the explainability algorithms you wish to use by referring to the [table](https://github.com/Trusted-AI/AIX360/tree/master#supported-configurations) above. For example, for python 3.10, use the following command:

```bash
conda create --name aix360 python=3.10
conda activate aix360
```

The shell should now look like `(aix360) $`. To deactivate the environment, run:

```bash
(aix360)$ conda deactivate
```

The prompt will return back to `$ ` or `(base)$`.

Note: Older versions of conda may use `source activate aix360` and `source
deactivate` (`activate aix360` and `deactivate` on Windows).


### Installation

Clone the latest version of this repository:

```bash
(aix360)$ git clone https://github.com/Trusted-AI/AIX360
```

If you'd like to run the examples and tutorial notebooks, download the datasets now and place them in
their respective folders as described in
[aix360/data/README.md](aix360/data/README.md).

Then, navigate to the root directory of the project which contains `setup.py` file and run:

```bash
(aix360)$ pip install -e .[<algo1>,<algo2>, ...]
```
The above command installs packages required by specific algorithms. Here `<algo>` refers to the installation keyword in [table](https://github.com/Trusted-AI/AIX360/tree/master#supported-configurations) above. For instance to install packages needed by BRCG, DIPVAE, and TSICE algorithms, one could use
```bash
(aix360)$ pip install -e .[rbm,dipvae,tsice]
```
 The default command `pip install .` installs [default dependencies](https://github.com/Trusted-AI/AIX360/blob/462c4d575bfc71c5cbfd32ceacdb3df96a8dc2d1/setup.py#L9) alone. 

Note that you may not be able to install two algorithms that require different versions of python in the same environment (for instance `contrastive` along with `rbm`). 

If you face any issues, please try upgrading pip and setuptools and uninstall any previous versions of aix360 before attempting the above step again. 

```bash
(aix360)$ pip install --upgrade pip setuptools
(aix360)$ pip uninstall aix360
```

## PIP Installation of AI Explainability 360

If you would like to quickly start using the AI explainability 360 toolkit without explicitly cloning this repository, you can use one of these options: 

* Install v0.3.0 via repository link
```bash
(your environment)$ pip install -e git+https://github.com/Trusted-AI/AIX360.git#egg=aix360[<algo1>,<algo2>,...]
```
For example, use `pip install -e git+https://github.com/Trusted-AI/AIX360.git#egg=aix360[rbm,dipvae,tsice]` to install BRCG, DIPVAE, and TSICE. You may need to install `cmake` if its not already installed in your environment using `conda install cmake`. 

* Install v0.3.0 (or previous versions) via [pypi](https://pypi.org/project/aix360/)
```bash
(your environment)$ pip install aix360
```

If you follow either of these two options, you will need to download the notebooks available in the [examples](./examples) folder separately. 

## Dealing with installation errors

AI Explainability 360 toolkit is [tested](https://github.com/Trusted-AI/AIX360/blob/master/.github/workflows/Build.yml) on Windows, MacOS, and Linux. However, if you still face installation issues due to package dependencies, please try installing the corresponding package via conda (e.g. conda install package-name) and then install the toolkit by following the usual steps. For example, if you face issues related to pygraphviz during installation, use `conda install pygraphviz` and then install the toolkit.

Please use the right python environment based on the [table](https://github.com/Trusted-AI/AIX360/tree/master#supported-configurations) above.

## Running in Docker

* Under `AIX360` directory build the container image from Dockerfile using `docker build -t aix360_docker .`
* Start the container image using command `docker run -it -p 8888:8888 aix360_docker:latest bash` assuming port 8888 is free on your machine.
* Inside the container start jupuyter lab using command `jupyter lab --allow-root --ip 0.0.0.0 --port 8888 --no-browser`
* Access the sample tutorials on your machine using URL `localhost:8888`

## Using AI Explainability 360

The `examples` directory contains a diverse collection of jupyter notebooks
that use AI Explainability 360 in various ways. Both examples and tutorial notebooks illustrate
working code using the toolkit. Tutorials provide additional discussion that walks
the user through the various steps of the notebook. See the details about
tutorials and examples [here](examples/README.md). 

## Citing AI Explainability 360

If you are using AI Explainability 360 for your work, we encourage you to

* Cite the following [paper](https://arxiv.org/abs/1909.03012). The bibtex entry is as follows: 

```
@misc{aix360-sept-2019,
title = "One Explanation Does Not Fit All: A Toolkit and Taxonomy of AI Explainability Techniques",
author = {Vijay Arya and Rachel K. E. Bellamy and Pin-Yu Chen and Amit Dhurandhar and Michael Hind
and Samuel C. Hoffman and Stephanie Houde and Q. Vera Liao and Ronny Luss and Aleksandra Mojsilovi\'c
and Sami Mourad and Pablo Pedemonte and Ramya Raghavendra and John Richards and Prasanna Sattigeri
and Karthikeyan Shanmugam and Moninder Singh and Kush R. Varshney and Dennis Wei and Yunfeng Zhang},
month = sept,
year = {2019},
url = {https://arxiv.org/abs/1909.03012}
}
```

* Put a star on this repository.

* Share your success stories with us and others in the [AI Explainability 360 Community](https://aix360.slack.com). 

## AIX360 Videos

* Introductory [video](https://www.youtube.com/watch?v=Yn4yduyoQh4) to AI
  Explainability 360 by Vijay Arya and Amit Dhurandhar, September 5, 2019 (35 mins)

## Acknowledgements

AIX360 is built with the help of several open source packages. All of these are listed in setup.py and some of these include: 
* Tensorflow https://www.tensorflow.org/about/bib
* Pytorch https://github.com/pytorch/pytorch
* scikit-learn https://scikit-learn.org/stable/about.html

## License Information

Please view both the [LICENSE](https://github.com/vijay-arya/AIX360/blob/master/LICENSE) file and the folder [supplementary license](https://github.com/vijay-arya/AIX360/tree/master/supplementary%20license) present in the root directory for license information. 

