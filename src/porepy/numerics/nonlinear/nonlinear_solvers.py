#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 11:21:54 2019

@author: eke001
"""
import logging
from inspect import signature
from typing import Callable, Optional

import numpy as np
import scipy.sparse.linalg as spla

import porepy as pp

__all__ = ["NewtonSolver", "scalar_newton", "scalar_bisection"]

# Module-wide logger
logger = logging.getLogger(__name__)

module_sections = ["numerics"]


class NewtonSolver:
    """Simple Newton solver for model classes."""

    def __init__(self, params=None):
        if params is None:
            params = {}

        default_options = {
            "max_iterations": 10,
            "nl_convergence_tol": 1e-10,
            "nl_divergence_tol": 1e5,
        }
        default_options.update(params)
        self.params = default_options

    def solve(self, model):
        model.before_newton_loop()

        iteration_counter = 0

        is_converged = False

        prev_sol = model.get_state_vector()
        init_sol = prev_sol
        errors = []
        error_norm = 1

        while iteration_counter <= self.params["max_iterations"] and not is_converged:
            logger.info(
                "Newton iteration number {} of {}".format(
                    iteration_counter, self.params["max_iterations"]
                )
            )

            # Re-discretize the nonlinear term
            model.before_newton_iteration()

            lin_tol = np.minimum(1e-4, error_norm)
            sol = self.iteration(model, lin_tol)

            model.after_newton_iteration(sol)

            error_norm, is_converged, is_diverged = model.check_convergence(
                sol, prev_sol, init_sol, self.params
            )
            prev_sol = sol
            errors.append(error_norm)

            if is_diverged:
                model.after_newton_failure(sol, errors, iteration_counter)
            elif is_converged:
                model.after_newton_convergence(sol, errors, iteration_counter)

            iteration_counter += 1

        if not is_converged:
            model.after_newton_failure(sol, errors, iteration_counter)

        return error_norm, is_converged, iteration_counter

    def iteration(self, model, lin_tol):
        """A single Newton iteration.

        Right now, this is a single line, however, we keep it as a separate function
        to prepare for possible future introduction of more advanced schemes.
        """

        # Assemble and solve
        sol = model.assemble_and_solve_linear_system(lin_tol)

        return sol


def _wrap_function_active_arg(fun):
    if len(signature(fun).parameters) == 1:

        def wrapper(x, active):
            return fun(x)

        return wrapper
    else:
        return fun


def scalar_newton(
    x0: np.ndarray,
    fun: Callable,
    tol: float = 1e-12,
    max_iter: int = 30,
    active: Optional[np.ndarray] = None,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
) -> np.ndarray:

    if active is None:
        active = np.ones(x0.size, dtype=bool)
    if lower is None:
        lower = np.full(x0.size, -np.inf)
    if upper is None:
        upper = np.full(x0.size, np.inf)

    fun = _wrap_function_active_arg(fun)

    counter = 0
    x = x0
    while counter <= max_iter:
        x_ad = pp.ad.initAdArrays(x)
        f = fun(x_ad, active)
        if np.all(np.max(np.abs(f.val)) < tol):
            return x

        dx = spla.spsolve(f.jac, -f.val)
        x += dx

        x = np.clip(x, a_min=lower, a_max=upper)

        counter += 1

    raise ValueError("Maximum number of iterations exceeded")


def scalar_bisection(
    fun: Callable,
    lower: np.ndarray,
    upper: np.ndarray,
    tol: float = 1e-12,
    max_iter: int = 100,
    active: Optional[np.ndarray] = None,
) -> np.ndarray:
    counter = 0

    fl = fun(lower)
    fu = fun(upper)
    if np.any(fl * fu > 0):
        raise ValueError(
            "Function values at interval bounds should have different sign"
        )

    upper = upper.astype(float)
    lower = lower.astype(float)

    while counter <= max_iter:
        middle = 0.5 * (upper + lower)
        fm = fun(middle)

        same_as_upper = np.sign(fm * fu) > 0
        same_as_lower = np.logical_not(same_as_upper)

        upper[same_as_upper] = middle[same_as_upper].copy()
        lower[same_as_lower] = middle[same_as_lower].copy()

        fl = fun(lower)
        fu = fun(upper)
        if np.max(np.abs(fl - fu)) < tol:
            assert np.all(middle >= lower)
            assert np.all(middle <= upper)
            return middle

        counter += 1

    raise ValueError("Maximum number of iterations reached")
