#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""
import tensorflow as tf
import numpy as np


def compute_content_cost(a_C, a_G):
    """
    Computes the content cost

    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

    Returns:
    J_content -- scalar that you compute using equation 1 above.
    """

    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_C.get_shape().as_list()

    # Reshape a_C and a_G (≈2 lines)
    a_C_unrolled = tf.reshape(a_C, [n_H * n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, [n_H * n_W, n_C])

    # compute the cost with tensorflow (≈1 line)
    #  J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))
    J_content = tf.reduce_sum((a_C_unrolled - a_G_unrolled) ** 2) / (4 * n_H * n_W * n_C)
    return J_content


def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)

    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    GA = tf.matmul(A, tf.transpose(A))
    return GA


def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing
    style of the image S
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing
    style of the image G

    Returns:
    J_style_layer -- tensor representing a scalar value, style cost defined above by
    equation (2)
    """

    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_S.get_shape().as_list()

    # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
    a_S = tf.transpose(tf.reshape(a_S, [n_H * n_W, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, [n_H * n_W, n_C]))

    # Computing gram_matrices for both images S and G (≈2 lines)
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss (≈1 line)
    J_style_layer = tf.reduce_sum(tf.square(GS - GG)) / ((2 * n_H * n_C * n_W) ** 2)

    return J_style_layer


def test_compute_content_cost():
    with tf.Session():
        tf.set_random_seed(1)
        a_C = tf.random.normal([1, 4, 4, 3], mean=1, stddev=4)
        a_G = tf.random.normal([1, 4, 4, 3], mean=1, stddev=4)
        J_content = compute_content_cost(a_C, a_G)
        result = J_content.eval()
        print(f"J_content = {result}")
        assert result == 6.765592575073242


def test_gram_matrix():
    with tf.Session():
        tf.set_random_seed(1)
        A = tf.random_normal([3, 2], mean=1, stddev=4)
        GA = gram_matrix(A)
        result = GA.eval()
        assert result == np.array([[6.422305, -4.429122, -2.096682],
                                   [-4.429122, 19.465837, 19.563871],
                                   [-2.096682, 19.563871, 20.686462]])


def main():
    #  test_compute_content_cost()
    test_gram_matrix()


if __name__ == '__main__':
    main()
