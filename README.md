# AI Explainability 360 (v0.2.0)

[![Build Status](https://travis-ci.com/IBM/AIX360.svg?branch=master)](https://travis-ci.com/IBM/AIX360)
[![Documentation Status](https://readthedocs.org/projects/aix360/badge/?version=latest)](https://aix360.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/aix360.svg)](https://badge.fury.io/py/aix360)

The AI Explainability 360 toolkit is an open-source library that supports interpretability and explainability of datasets and machine learning models. The AI Explainability 360 Python package includes a comprehensive set of algorithms that cover different dimensions of explanations along with proxy explainability metrics. 

The [AI Explainability 360 interactive experience](http://aix360.mybluemix.net/data) provides a gentle introduction to the concepts and capabilities by walking through an example use case for different consumer personas. The [tutorials and example notebooks](./examples) offer a deeper, data scientist-oriented introduction. The complete API is also available. 

There is no single approach to explainability that works best. There are many ways to explain: data vs. model, directly interpretable vs. post hoc explanation, local vs. global, etc. It may therefore be confusing to figure out which algorithms are most appropriate for a given use case. To help, we have created some [guidance material](http://aix360.mybluemix.net/resources#guidance) and a [chart](./aix360/algorithms/README.md) that can be consulted. 

We have developed the package with extensibility in mind. This library is still in development. We encourage the contribution of your explainability algorithms and metrics. To get started as a contributor, please join the [AI Explainability 360 Community on Slack](https://aix360.slack.com) by requesting an invitation [here](https://join.slack.com/t/aix360/shared_invite/enQtNzEyOTAwOTk1NzY2LTM1ZTMwM2M4OWQzNjhmNGRiZjg3MmJiYTAzNDU1MTRiYTIyMjFhZTI4ZDUwM2M1MGYyODkwNzQ2OWQzMThlN2Q). Please review the instructions to contribute code [here](CONTRIBUTING.md).

## Supported explainability algorithms

### Data explanation

- ProtoDash ([Gurumoorthy et al., 2019](https://arxiv.org/abs/1707.01212))
- Disentangled Inferred Prior VAE ([Kumar et al., 2018](https://openreview.net/forum?id=H1kG7GZAW))

### Local post-hoc explanation 

- ProtoDash ([Gurumoorthy et al., 2019](https://arxiv.org/abs/1707.01212))
- Contrastive Explanations Method ([Dhurandhar et al., 2018](https://papers.nips.cc/paper/7340-explanations-based-on-the-missing-towards-contrastive-explanations-with-pertinent-negatives))
- Contrastive Explanations Method with Monotonic Attribute Functions ([Luss et al., 2019](https://arxiv.org/abs/1905.12698))
- LIME ([Ribeiro et al. 2016](https://arxiv.org/abs/1602.04938),  [Github](https://github.com/marcotcr/lime))
- SHAP ([Lundberg, et al. 2017](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions),  [Github](https://github.com/slundberg/shap))

### Local direct explanation

- Teaching AI to Explain its Decisions ([Hind et al., 2019](https://doi.org/10.1145/3306618.3314273)) 
   
### Global direct explanation

- Boolean Decision Rules via Column Generation (Light Edition) ([Dash et al., 2018](https://papers.nips.cc/paper/7716-boolean-decision-rules-via-column-generation))
- Generalized Linear Rule Models ([Wei et al., 2019](http://proceedings.mlr.press/v97/wei19a.html))

### Global post-hoc explanation 

- ProfWeight ([Dhurandhar et al., 2018](https://papers.nips.cc/paper/8231-improving-simple-models-with-confidence-profiles))


## Supported explainability metrics
- Faithfulness ([Alvarez-Melis and Jaakkola, 2018](https://papers.nips.cc/paper/8003-towards-robust-interpretability-with-self-explaining-neural-networks))
- Monotonicity ([Luss et al., 2019](https://arxiv.org/abs/1905.12698))

## Setup

Supported Configurations:

| OS      | Python version |
| ------- | -------------- |
| macOS   | 3.6  |
| Ubuntu  | 3.6  |
| Windows | 3.6  |


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

Then, to create a new Python 3.6 environment, run:

```bash
conda create --name aix360 python=3.6
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
(aix360)$ git clone https://github.com/IBM/AIX360
```

If you'd like to run the examples and tutorial notebooks, download the datasets now and place them in
their respective folders as described in
[aix360/data/README.md](aix360/data/README.md).

Then, navigate to the root directory of the project which contains `setup.py` file and run:

```bash
(aix360)$ pip install -e .
```

## Using AI Explainability 360

The `examples` directory contains a diverse collection of jupyter notebooks
that use AI Explainability 360 in various ways. Both examples and tutorial notebooks illustrate
working code using the toolkit. Tutorials provide additional discussion that walks
the user through the various steps of the notebook. See the details about
tutorials and examples [here](examples/README.md). 

## Citing AI Explainability 360

A technical description of AI Explainability 360 is available in this
[paper](https://arxiv.org/abs/1909.03012). Below is the bibtex entry for this
paper.

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

## AIX360 Videos

* Introductory [video](https://www.youtube.com/watch?v=Yn4yduyoQh4) to AI
  Explainability 360 by Vijay Arya and Amit Dhurandhar, September 5, 2019 (35 mins)

