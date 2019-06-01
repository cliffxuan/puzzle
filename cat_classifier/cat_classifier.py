#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""
import numpy as np
#  import pytest

# GRADED FUNCTION: propagate


def test_propagate():
    w = np.array([[1.], [2.]])
    b = 2.
    X = np.array([[1., 2., -1.], [3., 4., -3.2]])
    Y = np.array([[1, 0, 1]])
    grads, cost = propagate(w, b, X, Y)
    dw = np.array([[0.9984560146379561], [2.395072388486207]])

    assert np.array_equal(grads['dw'], dw)
    assert grads['db'] == 0.001455578136784208
    assert cost == 5.801545319394553


def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """
    s = 1 / (1 + np.exp(-z))
    return s


def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size
    (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """

    m = X.shape[1]

    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T, X) + b)
    cost = - (Y * np.log(A) + (1 - Y) * np.log(1 - A)).sum() / m

    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = np.dot(X, (A - Y).T) / m
    db = (A - Y).sum() / m

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw": dw, "db": db}

    return grads, cost
