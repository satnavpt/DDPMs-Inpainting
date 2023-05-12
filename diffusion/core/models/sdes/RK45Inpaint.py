# Adapted from https://github.com/scipy/scipy/blob/v1.10.1/scipy/integrate/_ivp/rk.py#L280-L390

import numpy as np
from scipy.integrate._ivp.base import OdeSolver
from scipy.integrate._ivp.common import (
    validate_max_step,
    validate_tol,
    select_initial_step,
    norm,
    warn_extraneous,
    validate_first_step,
)
from scipy.integrate._ivp.rk import RkDenseOutput
from core.models.model_utils import from_flattened_numpy, to_flattened_numpy
import torch
from functools import partial

SAFETY = 0.9
MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.


def rk_step(fun, t, y, f, h, A, B, C, K, cond, mask, shape, device, sde):
    """Perform a single Runge-Kutta step.

    This function computes a prediction of an explicit Runge-Kutta method and
    also estimates the error of a less accurate method.

    Notation for Butcher tableau is as in [1]_.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t : float
        Current time.
    y : ndarray, shape (n,)
        Current state.
    f : ndarray, shape (n,)
        Current value of the derivative, i.e., ``fun(x, y)``.
    h : float
        Step to use.
    A : ndarray, shape (n_stages, n_stages)
        Coefficients for combining previous RK stages to compute the next
        stage. For explicit methods the coefficients at and above the main
        diagonal are zeros.
    B : ndarray, shape (n_stages,)
        Coefficients for combining RK stages for computing the final
        prediction.
    C : ndarray, shape (n_stages,)
        Coefficients for incrementing time for consecutive RK stages.
        The value for the first stage is always zero.
    K : ndarray, shape (n_stages + 1, n)
        Storage array for putting RK stages here. Stages are stored in rows.
        The last row is a linear combination of the previous rows with
        coefficients

    Returns
    -------
    y_new : ndarray, shape (n,)
        Solution at t + h computed with a higher accuracy.
    f_new : ndarray, shape (n,)
        Derivative ``fun(t + h, y_new)``.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4.
    """

    K[0] = f
    for s, (a, c) in enumerate(zip(A[1:], C[1:]), start=1):
        dy = np.dot(K[:s].T, a[:s]) * h
        K[s] = fun(t + c * h, y + dy)

    # new for colour
    ytest = from_flattened_numpy(y, shape).to(device).type(torch.float32)
    vec_t = torch.ones(ytest.shape[0], device=ytest.device) * (t + h)
    #

    y_new = y + h * np.dot(K[:-1].T, B)
    f_new = fun(t + h, y_new)

    # new for colour
    y_out = from_flattened_numpy(y_new, shape).to(device).type(torch.float32)
    masked_data_mean, std = sde.marginal_prob(cond, vec_t)
    masked_data = masked_data_mean + torch.randn_like(y_out) * std[:, None, None, None]
    y_out = (y_out) * (mask) + masked_data * (1.0 - mask)
    y_out = to_flattened_numpy(y_out)

    K[-1] = f_new

    return y_out, f_new


class RungeKutta(OdeSolver):
    """Base class for explicit Runge-Kutta methods."""

    C: np.ndarray = NotImplemented
    A: np.ndarray = NotImplemented
    B: np.ndarray = NotImplemented
    E: np.ndarray = NotImplemented
    P: np.ndarray = NotImplemented
    order: int = NotImplemented
    error_estimator_order: int = NotImplemented
    n_stages: int = NotImplemented

    def __init__(
        self,
        fun,
        t0,
        y0,
        t_bound,
        max_step=np.inf,
        rtol=1e-3,
        atol=1e-6,
        cond=None,
        mask=None,
        shape=None,
        device=None,
        sde=None,
        vectorized=False,
        first_step=None,
        **extraneous
    ):
        warn_extraneous(extraneous)
        fun = partial(fun, cond=cond, mask=mask)
        super().__init__(fun, t0, y0, t_bound, vectorized, support_complex=True)
        self.y_old = None
        self.max_step = validate_max_step(max_step)
        self.rtol, self.atol = validate_tol(rtol, atol, self.n)

        self.cond = cond
        self.mask = mask
        self.shape = shape
        self.device = device
        self.sde = sde

        self.f = self.fun(self.t, self.y)
        if first_step is None:
            self.h_abs = select_initial_step(
                self.fun,
                self.t,
                self.y,
                self.f,
                self.direction,
                self.error_estimator_order,
                self.rtol,
                self.atol,
            )
        else:
            self.h_abs = validate_first_step(first_step, t0, t_bound)
        self.K = np.empty((self.n_stages + 1, self.n), dtype=self.y.dtype)
        self.error_exponent = -1 / (self.error_estimator_order + 1)
        self.h_previous = None

    def _estimate_error(self, K, h):
        return np.dot(K.T, self.E) * h

    def _estimate_error_norm(self, K, h, scale):
        return norm(self._estimate_error(K, h) / scale)

    def _step_impl(self):
        t = self.t
        y = self.y

        max_step = self.max_step
        rtol = self.rtol
        atol = self.atol

        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)

        if self.h_abs > max_step:
            h_abs = max_step
        elif self.h_abs < min_step:
            h_abs = min_step
        else:
            h_abs = self.h_abs

        step_accepted = False
        step_rejected = False

        while not step_accepted:
            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP

            h = h_abs * self.direction
            t_new = t + h

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound

            h = t_new - t
            h_abs = np.abs(h)

            y_new, f_new = rk_step(
                self.fun,
                t,
                y,
                self.f,
                h,
                self.A,
                self.B,
                self.C,
                self.K,
                self.cond,
                self.mask,
                self.shape,
                self.device,
                self.sde,
            )

            scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
            # scalef = from_flattened_numpy(scale, self.shape).to(self.device).type(torch.float32)
            # mask = get_colourisation_mask(scalef)

            error_norm = self._estimate_error_norm(self.K, h, scale)
            print(error_norm)

            if error_norm < 1:
                if error_norm == 0:
                    factor = MAX_FACTOR
                else:
                    factor = min(MAX_FACTOR, SAFETY * error_norm**self.error_exponent)

                if step_rejected:
                    factor = min(1, factor)

                h_abs *= factor

                step_accepted = True
            else:
                h_abs *= max(MIN_FACTOR, SAFETY * error_norm**self.error_exponent)
                step_rejected = True
                print("step rejected")

        self.h_previous = h
        self.y_old = y

        self.t = t_new
        self.y = y_new

        self.h_abs = h_abs
        self.f = f_new

        return True, None

    def _dense_output_impl(self):
        Q = self.K.T.dot(self.P)
        return RkDenseOutput(self.t_old, self.t, self.y_old, Q)


class RK45Inpaint(RungeKutta):
    """Explicit Runge-Kutta method of order 5(4).

    This uses the Dormand-Prince pair of formulas [1]_. The error is controlled
    assuming accuracy of the fourth-order method accuracy, but steps are taken
    using the fifth-order accurate formula (local extrapolation is done).
    A quartic interpolation polynomial is used for the dense output [2]_.

    Can be applied in the complex domain.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(t, y)``.
        Here ``t`` is a scalar, and there are two options for the ndarray ``y``:
        It can either have shape (n,); then ``fun`` must return array_like with
        shape (n,). Alternatively it can have shape (n, k); then ``fun``
        must return an array_like with shape (n, k), i.e., each column
        corresponds to a single column in ``y``. The choice between the two
        options is determined by `vectorized` argument (see below).
    t0 : float
        Initial time.
    y0 : array_like, shape (n,)
        Initial state.
    t_bound : float
        Boundary time - the integration won't continue beyond it. It also
        determines the direction of the integration.
    first_step : float or None, optional
        Initial step size. Default is ``None`` which means that the algorithm
        should choose.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float and array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
        relative accuracy (number of correct digits), while `atol` controls
        absolute accuracy (number of correct decimal places). To achieve the
        desired `rtol`, set `atol` to be smaller than the smallest value that
        can be expected from ``rtol * abs(y)`` so that `rtol` dominates the
        allowable error. If `atol` is larger than ``rtol * abs(y)`` the
        number of correct digits is not guaranteed. Conversely, to achieve the
        desired `atol` set `rtol` such that ``rtol * abs(y)`` is always smaller
        than `atol`. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing array_like with shape (n,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.
    vectorized : bool, optional
        Whether `fun` is implemented in a vectorized fashion. Default is False.

    Attributes
    ----------
    n : int
        Number of equations.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : ndarray
        Current state.
    t_old : float
        Previous time. None if no steps were made yet.
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    nfev : int
        Number evaluations of the system's right-hand side.
    njev : int
        Number of evaluations of the Jacobian. Is always 0 for this solver as it does not use the Jacobian.
    nlu : int
        Number of LU decompositions. Is always 0 for this solver.

    References
    ----------
    .. [1] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
           formulae", Journal of Computational and Applied Mathematics, Vol. 6,
           No. 1, pp. 19-26, 1980.
    .. [2] L. W. Shampine, "Some Practical Runge-Kutta Formulas", Mathematics
           of Computation,, Vol. 46, No. 173, pp. 135-150, 1986.
    """

    order = 5
    error_estimator_order = 4
    n_stages = 6
    C = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1])
    A = np.array(
        [
            [0, 0, 0, 0, 0],
            [1 / 5, 0, 0, 0, 0],
            [3 / 40, 9 / 40, 0, 0, 0],
            [44 / 45, -56 / 15, 32 / 9, 0, 0],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
        ]
    )
    B = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84])
    E = np.array(
        [-71 / 57600, 0, 71 / 16695, -71 / 1920, 17253 / 339200, -22 / 525, 1 / 40]
    )
    # Corresponds to the optimum value of c_6 from [2]_.
    P = np.array(
        [
            [
                1,
                -8048581381 / 2820520608,
                8663915743 / 2820520608,
                -12715105075 / 11282082432,
            ],
            [0, 0, 0, 0],
            [
                0,
                131558114200 / 32700410799,
                -68118460800 / 10900136933,
                87487479700 / 32700410799,
            ],
            [
                0,
                -1754552775 / 470086768,
                14199869525 / 1410260304,
                -10690763975 / 1880347072,
            ],
            [
                0,
                127303824393 / 49829197408,
                -318862633887 / 49829197408,
                701980252875 / 199316789632,
            ],
            [
                0,
                -282668133 / 205662961,
                2019193451 / 616988883,
                -1453857185 / 822651844,
            ],
            [0, 40617522 / 29380423, -110615467 / 29380423, 69997945 / 29380423],
        ]
    )
