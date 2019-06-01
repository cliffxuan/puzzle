#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PageRank (developed by Larry Page and Sergey Brin) revolutionized web search by
generating a ranked list of web pages based on the underlying connectivity of the web.
The PageRank algorithm is based on an ideal random web surfer who, when reaching a page,
goes to the next page by clicking on a link. The surfer has equal probability of
clicking any link on the page and, when reaching a page with no links, has equal
probability of moving to any other page by typing in its URL. In addition, the surfer
may occasionally choose to type in a random URL instead of following the links on a
page. The PageRank is the ranked order of the pages from the most to the least probable
page the surfer will be viewing.
"""
import numpy as np
import numpy.linalg as la
import pytest


@pytest.mark.parametrize("damping", [
    0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1
])
def test_page_rank_fast(damping):
    matrix = np.array([
        [0, 1 / 2, 1 / 3, 0, 0, 0],
        [1 / 3, 0, 0, 0, 1 / 2, 0],
        [1 / 3, 1 / 2, 0, 1, 0, 1 / 2],
        [1 / 3, 0, 1 / 3, 0, 1 / 2, 1 / 2],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1 / 3, 0, 0, 0]
    ])
    fast_rank = page_rank_fast(matrix, damping)

    rank = page_rank(matrix, damping)

    assert np.linalg.norm(fast_rank - rank) < 10 ** -10


def page_rank_fast(link_matrix: np.array, damping: float):
    """
    damping: damping parameter. probability of following a link is  damping
    and the probability of choosing a random website is therefore  1 − damping
    """
    dimension = link_matrix.shape[0]
    adjusted = (
        damping * link_matrix
        + (1 - damping) / dimension * np.ones([dimension, dimension])
    )
    eigen_values, eigen_vectors = la.eig(adjusted)
    order = np.absolute(eigen_values).argsort()[::-1]
    eigen_values = eigen_values[order]
    eigen_vectors = eigen_vectors[:, order]
    principle_eigen_vector = eigen_vectors[:, 0]
    probability = np.real(principle_eigen_vector / np.sum(principle_eigen_vector))
    return probability


def page_rank(link_matrix: np.array, damping: float):
    dimension = link_matrix.shape[0]
    adjusted = (
        damping * link_matrix
        + (1 - damping) / dimension * np.ones([dimension, dimension])
    )
    r = np.ones(dimension) / dimension
    i = 0
    while True:
        last_r = r
        i += 1
        r = adjusted @ r
        if la.norm(r - last_r) < 10 ** -10:
            print(str(i) + " iterations to convergence.")
            break
    return r


