""" Module contains tests of nonlinear solvers.
"""
import numpy as np
import porepy as pp

import pytest


# Functions and intervals used for testing Newton and bisection
# when applied to scalar problems (single equation)
f_1 = (
    lambda x: np.power(x, 3) - 0.5 * x,
    np.array([-1 / np.sqrt(2), 0, 1 / np.sqrt(2)]),
)
f_2 = (lambda x: np.power(x, 2) - 1, np.array([-1, 1]))
f_3 = (lambda x: np.power(x, 2) + 1, np.array([]))
interval_1 = None
interval_2 = [-2, 2]
interval_3 = [0, 2]
interval_4 = [2, 4]


@pytest.mark.parametrize("fun_root", [f_1, f_2, f_3])
@pytest.mark.parametrize(
    "interval",
    [
        interval_1,
        interval_2,
        interval_3,
        interval_4,
    ],
)
def test_scalar_newton(fun_root, interval):
    """Test of Newton's method when applied to a single scalar
    equation (e.g. vectorization is not invoked).
    """
    fun, root = fun_root

    if interval is None:
        upper = None
        lower = None
    else:
        lower = np.array([interval[0]])
        upper = np.array([interval[1]])

    x0 = np.array([0.5])

    tol = 1e-12

    # If the problem is known to have no roots, or the roots are
    # outside the given interval, an error should be raised.
    no_root = root.size == 0
    root_outside = (
        root.size > 0
        and interval is not None
        and (np.all(root < lower[0]) or np.all(root > upper[0]))
    )

    if no_root or root_outside:
        pytest.raises(
            ValueError,
            pp.nonlinear.scalar_newton,
            x0,
            fun,
            lower=lower,
            upper=upper,
            tol=tol,
        )
    else:
        x = pp.nonlinear.scalar_newton(x0, fun, tol=tol, lower=lower, upper=upper)

        assert np.min(np.abs(fun(x[0]) - fun(root))) < tol


@pytest.mark.parametrize("fun_root", [f_1, f_2, f_3])
@pytest.mark.parametrize("interval", [interval_2, interval_3, interval_4])
def test_scalar_bisection(fun_root, interval):
    """Test the bisection method when applied to a single scalar equation."""
    fun, root = fun_root

    lower = np.array([interval[0]])
    upper = np.array([interval[1]])

    tol = 1e-12

    # If the problem is known to have no roots, the roots are
    # outside the given interval or the function value is the same on the upper and
    # lower bounds, an error should be raised.
    no_root = root.size == 0
    no_sign_change = np.any(np.sign(fun(lower) * fun(upper)) > 0)
    root_outside = np.all(root < interval[0]) or np.all(root > interval[1])

    if no_root or no_sign_change or root_outside:
        pytest.raises(
            ValueError,
            pp.nonlinear.scalar_bisection,
            fun,
            lower=lower,
            upper=upper,
            tol=tol,
        )
    else:
        x = pp.nonlinear.scalar_bisection(fun, lower=lower, upper=upper, tol=tol)

        assert np.min(np.abs(fun(x[0]) - fun(root))) < tol


# Functions with two unknowns used for testing Newton and bisection when applied
# to systems of decoupled equations. Also intervals with lower and upper bounds
# for when the function is applied.
f_4 = (
    lambda x: (x ** np.array([3, 2])) - np.array([0, 1]),
    [np.array([0]), np.array([-1, 1])],
)
f_5 = (
    lambda x: (x ** np.array([3, 2])) + np.array([0, 1]),
    [np.array([0]), np.array([])],
)
interval_5 = [np.array([-2, -2]), np.array([2, 2])]
interval_6 = [np.array([-2, 0]), np.array([2, 2])]
interval_7 = [np.array([-2, 2]), np.array([2, 4])]


@pytest.mark.parametrize("fun_root", [f_4, f_5])
@pytest.mark.parametrize(
    "interval",
    [
        interval_5,
        interval_6,
        interval_7,
    ],
)
def test_scalar_newton_decoupled_systems(fun_root, interval):
    """Test of Newton's method when applied to two equations, decoupled."""
    fun, root = fun_root

    if interval is None:
        upper = None
        lower = None
    else:
        lower = interval[0]
        upper = interval[1]

    x0 = np.full(2, 0.5)

    tol = 1e-12

    # If the problem is known to have no roots, or the roots are
    # outside the given interval, an error should be raised.
    no_root = any([r.size == 0 for r in root])
    root_outside = False
    if not no_root:
        for i in range(x0.size):
            r = root[i]
            if np.all(r < lower[i]) or np.all(r > upper[i]):
                root_outside = True

    if no_root or root_outside:
        pytest.raises(
            ValueError,
            pp.nonlinear.scalar_newton,
            x0,
            fun,
            lower=lower,
            upper=upper,
            tol=tol,
        )
    else:
        x = pp.nonlinear.scalar_newton(x0, fun, tol=tol, lower=lower, upper=upper)

        for i in range(x.size):
            assert np.min(np.abs(fun(x[i]) - fun(root[i]))) < tol


@pytest.mark.parametrize("fun_root", [f_4, f_5])
@pytest.mark.parametrize("interval", [interval_5, interval_6, interval_7])
def test_scalar_bisection_decoupled_systems(fun_root, interval):
    """Test of the bisection method when applied to two equations, decoupled."""

    fun, root = fun_root

    lower = interval[0]
    upper = interval[1]

    tol = 1e-12

    # If the problem is known to have no roots, the roots are
    # outside the given interval or the function value is the same on the upper and
    # lower bounds, an error should be raised.
    no_root = any([r.size == 0 for r in root])
    no_sign_change = np.any(np.sign(fun(lower) * fun(upper)) > 0)
    root_outside = False
    if not no_root:
        for i in range(lower.size):
            r = root[i]
            if np.all(r < lower[i]) or np.all(r > upper[i]):
                root_outside = True

    if no_root or no_sign_change or root_outside:
        pytest.raises(
            ValueError,
            pp.nonlinear.scalar_bisection,
            fun,
            lower=lower,
            upper=upper,
            tol=tol,
        )
    else:
        x = pp.nonlinear.scalar_bisection(fun, lower=lower, upper=upper, tol=tol)

        for i in range(x.size):
            assert np.min(np.abs(fun(x[i]) - fun(root[i]))) < tol
