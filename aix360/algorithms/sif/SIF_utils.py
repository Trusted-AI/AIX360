### Adapted from TF repo

from tensorflow import gradients
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import RNNCell, LSTMCell
import statsmodels.tsa.api as smt


def get_contaminate_series(core_series, y_contaminate, train_label, gamma=0.1):
    '''
    gamma: the percentage of the contaminated sample insert into the orginal time sequence.
    '''
    series = core_series.copy()
    n_timestamp = train_label.shape[0]
    idx = np.random.binomial(size=n_timestamp, n=1, p=gamma).astype(bool)
    idx = np.where(idx)
    num_to_chg = len(idx)
    if num_to_chg > 0:
        series[idx, :] = y_contaminate[idx, np.newaxis]
    return series


def sample_Z(batch_size, seq_length, latent_dim):
    sample = np.float32(np.random.normal(size=[batch_size, seq_length, latent_dim]))
    return sample


def generator_lstm(z, hidden_units_g, seq_length, batch_size, num_signals, reuse=False, parameters=None):
    """
    If parameters are supplied, initialise as such
    """
    with tf.variable_scope("generator") as scope:
        if reuse:
            scope.reuse_variables()
        if parameters is None:
            W_out_G_initializer = tf.truncated_normal_initializer()
            b_out_G_initializer = tf.truncated_normal_initializer()
            lstm_initializer = None

        else:
            W_out_G_initializer = tf.constant_initializer(value=parameters['generator/W_out_G:0'])
            b_out_G_initializer = tf.constant_initializer(value=parameters['generator/b_out_G:0'])

            lstm_initializer = tf.constant_initializer(value=parameters['generator/rnn/lstm_cell/weights:0'])
            bias_start = parameters['generator/rnn/lstm_cell/biases:0']

        W_out_G = tf.get_variable(name='W_out_G', shape=[hidden_units_g, num_signals],
                                  initializer=W_out_G_initializer)
        b_out_G = tf.get_variable(name='b_out_G', shape=num_signals, initializer=b_out_G_initializer)

        # inputs
        inputs = z

        cell = LSTMCell(num_units=hidden_units_g,
                        state_is_tuple=True,
                        initializer=lstm_initializer,
                        reuse=reuse)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            sequence_length=[seq_length] * batch_size,
            inputs=inputs)
        rnn_outputs = rnn_outputs[:, -1, :]
        output = tf.matmul(rnn_outputs, W_out_G) + b_out_G  # output weighted sum

        m = tf.reduce_mean(output, axis=0)
        std = tf.sqrt(tf.reduce_mean(tf.square(output-m)))
    return (output - m)/std


def generator_arma(params, p, q, nsample):
    """
    If parameters are supplied, initialise as such
    """
    assert len(params) == p+q, "The length of the parameters must equals p+q"
    ar = np.r_[1, -params[:p]]
    ma = np.r_[1, params[p:]]
    output = smt.arma_generate_sample(ar, ma, nsample, burnin=200)
    m = np.mean(output)
    std = np.std(output)
    if std < 1e-9:
        std = 1
    return (output - m) / std


def hessian_vector_product(ys, xs, v):
    """Multiply the Hessian of `ys` wrt `xs` by `v`.
    This is an efficient construction that uses a backprop-like approach
    to compute the product between the Hessian and another vector. The
    Hessian is usually too large to be explicitly computed or even
    represented, but this method allows us to at least multiply by it
    for the same big-O cost as backprop.
    Implicit Hessian-vector products are the main practical, scalable way
    of using second derivatives with neural networks. They allow us to
    do things like construct Krylov subspaces and approximate conjugate
    gradient descent.
    Example: if `y` = 1/2 `x`^T A `x`, then `hessian_vector_product(y,
    x, v)` will return an expression that evaluates to the same values
    as (A + A.T) `v`.
    Args:
      ys: A scalar value, or a tensor or list of tensors to be summed to
          yield a scalar.
      xs: A list of tensors that we should construct the Hessian over.
      v: A list of tensors, with the same shapes as xs, that we want to
         multiply by the Hessian.
    Returns:
      A list of tensors (or if the list would be length 1, a single tensor)
      containing the product between the Hessian and `v`.
    Raises:
      ValueError: `xs` and `v` have different length.
    """


    # Validate the input
    length = len(xs)
    if len(v) != length:
        raise ValueError("xs and v must have the same length.")

    # First backprop
    grads = gradients(ys, xs)

    # grads = xs

    assert len(grads) == length

    elemwise_products = [math_ops.multiply(grad_elem, array_ops.stop_gradient(v_elem))
        for grad_elem, v_elem in zip(grads, v) if grad_elem is not None
    ]

    # Second backprop
    grads_with_none = gradients(elemwise_products, xs)
    return_grads = [
        grad_elem if grad_elem is not None \
            else tf.zeros_like(x) \
        for x, grad_elem in zip(xs, grads_with_none)]

    return return_grads


def _AsList(x):
    return x if isinstance(x, (list, tuple)) else [x]


def gradient_reshape(ys, x, **kwargs):
    # Compute the partial derivatives of the input with respect to all
    # elements of `x`
    _gradients = tf.gradients(ys, x, **kwargs)[0]
    # for higher dimension, let's conver then into a vector
    _gradients = tf.reshape(_gradients, [-1])
    return _gradients


def hessians(ys, xs, name="hessians", colocate_gradients_with_ops=False,
             gate_gradients=False, aggregation_method=None):
    """Constructs the Hessian of sum of `ys` with respect to `x` in `xs`.
    `hessians()` adds ops to the graph to output the Hessian matrix of `ys`
    with respect to `xs`.    It returns a list of `Tensor` of length `len(xs)`
    where each tensor is the Hessian of `sum(ys)`. This function currently
    only supports evaluating the Hessian with respect to (a list of) one-
    dimensional tensors.
    The Hessian is a matrix of second-order partial derivatives of a scalar
    tensor (see https://en.wikipedia.org/wiki/Hessian_matrix for more details).
    Args:
        ys: A `Tensor` or list of tensors to be differentiated.
        xs: A `Tensor` or list of tensors to be used for differentiation.
        name: Optional name to use for grouping all the gradient ops together.
            defaults to 'hessians'.
        colocate_gradients_with_ops: See `gradients()` documentation for details.
        gate_gradients: See `gradients()` documentation for details.
        aggregation_method: See `gradients()` documentation for details.
    Returns:
        A list of Hessian matrices of `sum(y)` for each `x` in `xs`.
    Raises:
        LookupError: if one of the operations between `xs` and `ys` does not
            have a registered gradient function.
        ValueError: if the arguments are invalid or not supported. Currently,
            this function only supports one-dimensional `x` in `xs`.
    """
    # import time
    # start_time = time.time()

    xs = _AsList(xs)
    kwargs = {
            'colocate_gradients_with_ops': colocate_gradients_with_ops,
            'gate_gradients': gate_gradients,
            'aggregation_method': aggregation_method
    }
    # Compute a hessian matrix for each x in xs
    hessians = []
    for x in xs:
        # Check dimensions
        ndims = x.get_shape().ndims
        if ndims is None:
            raise ValueError('Cannot compute Hessian because the dimensionality of '
                                             'element number %d of `xs` cannot be determined' % i)
        #elif ndims != 1:
        #    raise ValueError('Computing hessians is currently only supported for '
        #                                     'one-dimensional tensors. Element number %d of `xs` has '
        #                                     '%d dimensions.' % (i, ndims))
        with ops.name_scope(name + '_first_derivative'):
            _gradients = gradient_reshape(ys, x, **kwargs)

            # Unpack the gradients into a list so we can take derivatives with
            # respect to each element
            _gradients = tf.unstack(_gradients, _gradients.get_shape()[0].value)
        row = []
        with ops.name_scope(name + '_second_derivative'):
            for xp in xs:
                # Compute the partial derivatives with respect to each element of the list
                _hess = [gradient_reshape(_gradient, xp, **kwargs) for _gradient in _gradients]
                #_hess = tf.gradients(_gradients, x, **kwargs)[0]
                # Pack the list into a matrix and add to the list of hessians
                row.append(tf.stack(_hess, name=name))
        hessians.append(row)
    # end_time = time.time()
    # print('{} s to compute hessian'.format(end_time - start_time))
    return hessians


def derivative_x1_x2(ys, xs1, xs2, name="Second_order_derivative",
                     colocate_gradients_with_ops=False,
            gate_gradients=False, aggregation_method=None):

    xs1 = _AsList(xs1)
    xs2 = _AsList(xs2)
    kwargs = {
      'colocate_gradients_with_ops': colocate_gradients_with_ops,
      'gate_gradients': gate_gradients,
      'aggregation_method': aggregation_method
    }
    # Compute a hessian matrix for each x in xs
    derivatives = []
    for i, x in enumerate(xs1):
        # Check dimensions
        ndims = x.get_shape().ndims
        if ndims is None:
            raise ValueError('Cannot compute Hessian because the dimensionality of '
                            'element number %d of `xs` cannot be determined' % i)
        #elif ndims != 1:
        #  raise ValueError('Computing hessians is currently only supported for '
        #                   'one-dimensional tensors. Element number %d of `xs` has '
        #                   '%d dimensions.' % (i, ndims))
        with ops.name_scope(name + '_first_derivative'):
            _gradients = gradient_reshape(ys, x, **kwargs)

            # Unpack the gradients into a list so we can take derivatives with
            # respect to each element
            _gradients = array_ops.unpack(_gradients, _gradients.get_shape()[0].value)

        _derivative = []
        for x2 in xs2:
            with ops.name_scope(name + '_second_derivative'):
                # Compute the partial derivatives with respect to each element of the list
                _hess = [gradient_reshape(_gradient, x2, **kwargs) for _gradient in _gradients]
                #_hess = tf.gradients(_gradients, x, **kwargs)[0]
                # Pack the list into a matrix and add to the list of hessians
            _derivative.append(array_ops.pack(_hess, name=name))
        derivatives.append(_derivative)

    return derivatives


def variable(name, shape, initializer):
    dtype = tf.float32
    var = tf.get_variable(
        name,
        shape,
        initializer=initializer,
        dtype=dtype)
    return var


def variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    dtype = tf.float32
    var = variable(
        name,
        shape,
        initializer=tf.truncated_normal_initializer(
            stddev=stddev,
            dtype=dtype))

    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var


def normalize_vector(v):
    """
    Takes in a vector in list form, concatenates it to form a single vector,
    normalizes it to unit length, then returns it in list form together with its norm.
    """
    norm_val = np.linalg.norm(np.concatenate(v))
    norm_v = [a / norm_val for a in v]
    return norm_v, norm_val


def operate_deep(op, var):
    # if isinstance(var, list):
    if type(var).__name__ == "list":
        return [operate_deep(op, x) for x in var]
    # if isinstance(var, tuple):
    if type(var).__name__ == "tuple":
        return tuple(operate_deep(op, x) for x in var)
    return op(var)


def operate_deep_2v(op, var1, var2):
    if isinstance(var1, list):
        return [operate_deep_2v(op, a, b) for a, b in zip(var1, var2)]
    if isinstance(var1, tuple):
        return tuple(operate_deep_2v(op, a, b) for a, b in zip(var1, var2))
    return op(var1, var2)

