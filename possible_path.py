#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
step 1:
number of possible pathes from bottom left of a m x n grid to top right.
each step forward can go either right or up, but not left or down.

step 2:
built on top of step 1, a cell may block access, e.g.
1 1 1
1 0 1
1 1 1
the 2nd item in the middle row is not allowed access.
"""
import typing

import pytest


# step 1
@pytest.mark.parametrize("m, n, expected", [
    (1, 1, 1),
    (1, 2, 1),
    (1, 3, 1),
    (2, 2, 2),
    (2, 3, 3),
    (3, 3, 6),
    (3, 4, 10),
    (4, 4, 20),
    (4, 5, 35),
    (5, 5, 70),
    (5, 6, 126),
    (6, 6, 252),
    (7, 7, 924),
    (32, 32, 465428353255261088),
    (64, 64, 6034934435761406706427864636568328000)
])
def test_possible_path(m, n, expected):
    #  assert possible_path(m, n) == expected
    assert possible_path_memorized(m, n) == expected


def possible_path(m, n):
    if m == 1 or n == 1:
        return 1
    return possible_path(m - 1, n) + possible_path(m, n - 1)


def possible_path_memorized(m, n, solutions=None):
    if solutions is None:
        solutions = {}
    if (m, n) in solutions:
        return solutions[(m, n)]
    if m == 1 or n == 1:
        return 1
    result = (
        possible_path_memorized(m - 1, n, solutions)
        + possible_path_memorized(m, n - 1, solutions)
    )
    solutions[(m, n)] = result
    return result


# step 2
@pytest.mark.parametrize("m, expected", [
    ([[1]], 1),
    ([[0]], 0),
    ([[1, 1], [1, 1]], 2),
    ([[1, 1, 1], [1, 1, 1], [1, 1, 1]], 6),
    ([[1, 1, 1], [1, 0, 1], [1, 1, 1]], 2),
    ([[1] * 7] * 7, 924),
    ([[1] * 32] * 32, 465428353255261088),
])
def test_possible_path_without_full_access(m, expected):
    #  assert possible_path_without_full_access(m) == expected
    assert possible_path_without_full_access_memorized(m) == expected


def possible_path_without_full_access(m: typing.List[typing.List[int]]):
    # no row or column
    if len(m) == 0 or len(m[0]) == 0:
        return 0
    # on no access
    if m[0][0] == 0:
        return 0
    # end
    if m == [[1]]:
        return 1
    right = possible_path_without_full_access([row[1:] for row in m])
    down = possible_path_without_full_access(m[1:])
    return right + down


def possible_path_without_full_access_memorized(
        m: typing.List[typing.List[int]],
        solutions: typing.Optional[typing.Dict] = None
):
    if solutions is None:
        solutions = {}
    key = tuple(tuple(row) for row in m)
    try:
        return solutions[key]
    except KeyError:
        pass
    if len(m) == 0 or len(m[0]) == 0:
        return 0
    if m[0][0] == 0:
        return 0
    if len(m) == 1 and len(m[0]) == 1 and m[0][0] == 1:
        return 1
    right = possible_path_without_full_access_memorized(
        [row[1:] for row in m], solutions
    )
    down = possible_path_without_full_access_memorized(
        m[1:], solutions
    )
    result = right + down
    solutions[key] = result
    return result
