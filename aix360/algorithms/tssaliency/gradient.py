from abc import ABC, abstractmethod
from typing import Any, Callable
import numpy as np
import warnings


def mc_gradient_compute(
    x, fn: Callable, n_samples: int = 10, mu: float = 0.01, **kwargs
):
    """
    Monte Carlo Gradient computation.

    References:
        .. [#0] `Sijia Liu et al. "Zeroth-Order Stochastic Variance Reduction for
        Nonconvex Optimization". <https://arxiv.org/pdf/1906.00117.pdf>`_
    """

    ### 1- Sample Directions in the uniform ball
    sigma = 10
    u_normal_shape = [n_samples]

    for _ in x.shape:
        u_normal_shape.append(_)
    u_normal = np.random.normal(0, sigma, tuple(u_normal_shape))

    axis_norm = [i + 1 for i in range(len(x.shape))]
    u_normal = u_normal / (
        np.linalg.norm(u_normal, axis=tuple(axis_norm), keepdims=True)
    )

    ### 2 - Generate Perturbations

    perturbed_samples = x + mu * u_normal
    f_predict_samples = None

    try:
        f_predict_samples = fn(perturbed_samples)
    except Exception as ex:
        warnings.warn(
            "Batch scoring failed with error: {}. Scoring sequentially...".format(ex)
        )
        f_predict_samples = [
            fn(perturbed_samples[i]) for i in range(perturbed_samples.shape[0])
        ]
        f_predict_samples = np.array(f_predict_samples)

    if f_predict_samples is None:
        raise Exception("Model prediction could not be computed for gradient samples.")

    f_predict_x = fn(x)

    df = f_predict_samples - f_predict_x

    mc_gradient = None
    try:  # expecting single output
        if len(u_normal.shape) > 2:
            df = df[..., np.newaxis]
        mc_gradient = (np.prod(x.shape) / (mu)) * np.mean(df * u_normal, axis=0)
    except ValueError:  # multi output e.g., forecasting etc.
        warnings.warn(
            "Model function returns multi-output. It is advised to use single output for saliency explanation."
        )
        df = np.mean(df, axis=2).reshape(-1, 1)  # or take first value
        if len(u_normal.shape) > 2:
            df = df[..., np.newaxis]
        mc_gradient = (np.prod(x.shape) / (mu)) * np.mean(df * u_normal, axis=0)

    return mc_gradient
