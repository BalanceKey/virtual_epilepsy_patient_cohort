"""
Model taken from the TVB library available on github.com/the-virtual-brain/tvb-root
the Epileptor model has been reduced to 3 state variables: x_1, y_1, z

Here the stimuli causes an accumulation and increase in m, and when m crosses a threshold
then x0 becomes exitable and it travels up to the limit cycle until m goes back under threshold
Here m evolves at the same speed as z.
"""

from tvb.simulator.lab import *
from tvb.simulator.models.base import ModelNumbaDfun
from tvb.basic.neotraits.api import NArray, List, Range, Final, HasTraits, Attr
from numba import guvectorize, float64, vectorize
import numpy


@vectorize([float64(float64, float64)])
def heaviside_impl(x1, x2):
    if x1 < 0:
        return 0.0
    elif x1 > 0:
        return 1.0
    else:
        return x2


@guvectorize([(float64[:],) * 18], '(n),(m)' + ',()' * 15 + '->(n)', nopython=True)
def _numba_dfun(y, c_pop, x0, Iext, a, b, tt, Kvf, c, d, r, r2, Ks, Kf, modification, Istim, threshold, ydot):
    "Gufunc for Hindmarsh-Rose-Jirsa Epileptor model equations."

    c_pop1 = c_pop[0]  # x1
    c_pop2 = c_pop[1]  # z
    c_pop3 = c_pop[2]  # m

    # population 1
    if y[0] < 0.0:
        ydot[0] = - a[0] * y[0] ** 2 + b[0] * y[0]
    else:
        # ydot[0] = slope[0] + 0.6 * (y[2] - 4.0) ** 2
        ydot[0] = y[3] + 0.6 * (y[2] - 4.0) ** 2
    ydot[0] = tt[0] * (y[1] - y[2] + Iext[0] + Istim[0] * 5 + Kvf[0] * c_pop1 + ydot[0] * y[0])
    ydot[1] = tt[0] * (c[0] - d[0] * y[0] ** 2 - y[1])

    # energy
    if y[2] < 0.0:
        ydot[2] = - 0.1 * y[2] ** 7
    else:
        ydot[2] = 0.0

    H = heaviside_impl(y[6] - threshold[0], 1.0)

    if modification[0]:
        h = x0[0] + H + 3 / (1 + numpy.exp(-(y[0] + 0.5) / 0.1))
    else:
        h = 4 * (y[0] - x0[0] - H) + ydot[2]

    ydot[2] = tt[0] * (r[0] * (h - y[2] + Ks[0] * c_pop1))

    ydot[3] = tt[0] * r2[0] * (-0.3 * y[3] + abs(Istim[0]) * 20 + Kf[0] * c_pop3)


class EpileptorStim(ModelNumbaDfun):
    r"""
    The Epileptor is a composite neural mass model of six dimensions which
    has been crafted to model the phenomenology of epileptic seizures.
    (see [Jirsaetal_2014]_)

    Equations and default parameters are taken from [Jirsaetal_2014]_.

          +------------------------------------------------------+
          |                         Table 1                      |
          +----------------------+-------------------------------+
          |        Parameter     |           Value               |
          +======================+===============================+
          |         I_rest1      |              3.1              |
          +----------------------+-------------------------------+
          |         I_rest2      |              0.45             |
          +----------------------+-------------------------------+
          |         r            |            0.00035            |
          +----------------------+-------------------------------+
          |         x_0          |             -1.6              |
          +----------------------+-------------------------------+
          |         slope        |              0.0              |
          +----------------------+-------------------------------+
          |             Integration parameter                    |
          +----------------------+-------------------------------+
          |           dt         |              0.1              |
          +----------------------+-------------------------------+
          |  simulation_length   |              4000             |
          +----------------------+-------------------------------+
          |                    Noise                             |
          +----------------------+-------------------------------+
          |         nsig         | [0., 0., 0., 1e-3, 1e-3, 0.]  |
          +----------------------+-------------------------------+
          |              Jirsa et al. 2014                       |
          +------------------------------------------------------+


    .. figure :: img/Epileptor_01_mode_0_pplane.svg
        :alt: Epileptor phase plane

    .. [Jirsaetal_2014] Jirsa, V. K.; Stacey, W. C.; Quilichini, P. P.;
        Ivanov, A. I.; Bernard, C. *On the nature of seizure dynamics.* Brain,
        2014.


    Variables of interest to be used by monitors: -y[0] + y[3]

        .. math::
            \dot{x_{1}} &=& y_{1} - f_{1}(x_{1}, x_{2}) - z + I_{ext1} \\
            \dot{y_{1}} &=& c - d x_{1}^{2} - y{1} \\
            \dot{z} &=&
            \begin{cases}
            r(4 (x_{1} - x_{0}) - z-0.1 z^{7}) & \text{if } x<0 \\
            r(4 (x_{1} - x_{0}) - z) & \text{if } x \geq 0
            \end{cases} \\
            \dot{x_{2}} &=& -y_{2} + x_{2} - x_{2}^{3} + I_{ext2} + 0.002 g - 0.3 (z-3.5) \\
            \dot{y_{2}} &=& 1 / \tau (-y_{2} + f_{2}(x_{2}))\\
            \dot{g} &=& -0.01 (g - 0.1 x_{1})

    where:
        .. math::
            f_{1}(x_{1}, x_{2}) =
            \begin{cases}
            a x_{1}^{3} - b x_{1}^2 & \text{if } x_{1} <0\\
            -(slope - x_{2} + 0.6(z-4)^2) x_{1} &\text{if }x_{1} \geq 0
            \end{cases}

    and:

        .. math::
            f_{2}(x_{2}) =
            \begin{cases}
            0 & \text{if } x_{2} <-0.25\\
            a_{2}(x_{2} + 0.25) & \text{if } x_{2} \geq -0.25
            \end{cases}

    Note Feb. 2017: the slow permittivity variable can be modify to account for the time
    difference between interictal and ictal states (see [Proixetal_2014]).

    .. [Proixetal_2014] Proix, T.; Bartolomei, F; Chauvel, P; Bernard, C; Jirsa, V.K. *
        Permittivity coupling across brain regions determines seizure recruitment in
        partial epilepsy.* J Neurosci 2014, 34:15009-21.

    """

    a = NArray(
        label=":math:`a`",
        default=numpy.array([1.0]),
        doc="Coefficient of the cubic term in the first state variable")

    b = NArray(
        label=":math:`b`",
        default=numpy.array([3.0]),
        doc="Coefficient of the squared term in the first state variabel")

    c = NArray(
        label=":math:`c`",
        default=numpy.array([1.0]),
        doc="Additive coefficient for the second state variable, \
        called :math:`y_{0}` in Jirsa paper")

    d = NArray(
        label=":math:`d`",
        default=numpy.array([5.0]),
        doc="Coefficient of the squared term in the second state variable")

    r = NArray(
        label=":math:`r`",
        domain=Range(lo=0.0, hi=0.001, step=0.00005),
        default=numpy.array([0.00035]),
        doc="Temporal scaling in the third state variable, \
        called :math:`1/\\tau_{0}` in Jirsa paper")

    r2 = NArray(
        label=":math:`r`",
        domain=Range(lo=0.0, hi=0.001, step=0.00005),
        default=numpy.array([0.00035]),
        doc="Temporal scaling in the fourth state variable m")

    s = NArray(
        label=":math:`s`",
        default=numpy.array([4.0]),
        doc="Linear coefficient in the third state variable")

    x0 = NArray(
        label=":math:`x_0`",
        domain=Range(lo=-3.0, hi=-1.0, step=0.1),
        default=numpy.array([-1.6]),
        doc="Epileptogenicity parameter")

    threshold = NArray(
        label=":math:`threshold`",
        domain=Range(lo=0.0, hi=10.0, step=0.1),
        default=numpy.array([1.8]),
        doc="Accumulation threshold for m")

    Iext = NArray(
        label=":math:`I_{ext}`",
        domain=Range(lo=1.5, hi=5.0, step=0.1),
        default=numpy.array([3.1]),
        doc="External input current to the first population")

    # slope = NArray(
    #     label=":math:`slope`",
    #     domain=Range(lo=-16.0, hi=6.0, step=0.1),
    #     default=numpy.array([0.]),
    #     doc="Linear coefficient in the first state variable")

    Kvf = NArray(
        label=":math:`K_{vf}`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=4.0, step=0.5),
        doc="Coupling scaling on a very fast time scale.")

    Kf = NArray(
        label=":math:`K_{f}`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=4.0, step=0.5),
        doc="Correspond to the coupling scaling on a fast time scale.")

    Ks = NArray(
        label=":math:`K_{s}`",
        default=numpy.array([0.0]),
        domain=Range(lo=-4.0, hi=4.0, step=0.1),
        doc="Permittivity coupling, that is from the fast time scale toward the slow time scale")

    tt = NArray(
        label=":math:`K_{tt}`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.001, hi=10.0, step=0.001),
        doc="Time scaling of the whole system")

    modification = NArray(
        dtype=bool,
        label=":math:`modification`",
        default=numpy.array([False]),
        doc="When modification is True, then use nonlinear influence on z. \
        The default value is False, i.e., linear influence.")

    timer = NArray(
        label=":math:`timer`",
        default=numpy.array([0.]),
        doc="#TODO improve duration of x0(t) in nr of steps when it travels to the limit cycle, how long it can stay")

    Istim = NArray(
        label=":math:`I_{ext}`",
        domain=Range(lo=0, hi=50.0, step=0.1),
        default=numpy.array([0.]),
        doc="External input current from the stimuli applied to mysterious variables. ~BD")

    # n_stim = NArray(
    #     label=":math:`n_stim`",
    #     default=numpy.array([0.]),
    #     doc="Counter for the number of stimulations applied to the model")

    state_variable_range = Final(
        default={
            "x1": numpy.array([-2., 1.]),
            "y1": numpy.array([-20., 2.]),
            "z": numpy.array([2.0, 5.0]),
            "m": numpy.array([-16.0, 6.0])
        },
        label="State variable ranges [lo, hi]",
        doc="Typical bounds on state variables in the Epileptor model.")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=('x1', 'y1', 'z', 'm'),
        default=('x1', 'z'),
        doc="Quantities of the Epileptor available to monitor.",
    )

    state_variables = ('x1', 'y1', 'z', 'm')
    _nvar = 4
    cvar = numpy.array([0], dtype=numpy.int32)  # should these not be constant Attr's?

    # cvar.setflags(write=False)  # todo review this
    # ON = True

    # def _numpy_dfun(self, state_variables, coupling, local_coupling=0.0,
    #                 array=numpy.array, where=numpy.where, concat=numpy.concatenate):
    #
    #     y = state_variables
    #     ydot = numpy.empty_like(state_variables)
    #
    #     Iext = self.Iext + local_coupling * y[0]
    #     c_pop1 = coupling[0, :]
    #     c_pop2 = coupling[1, :]
    #     c_pop3 = coupling[2, :]
    #
    #     # population 1
    #     if_ydot0 = - self.a * y[0] ** 2 + self.b * y[0]
    #     #else_ydot0 = self.slope + 0.6 * (y[2] - 4.0) ** 2
    #     else_ydot0 =  y[3] + 0.6 * (y[2] - 4.0) ** 2
    #     ydot[0] = self.tt * (y[1] - y[2] + Iext + self.Istim * 5 + self.Kvf * c_pop1 + where(y[0] < 0., if_ydot0, else_ydot0) * y[0])
    #     ydot[1] = self.tt * (self.c - self.d * y[0] ** 2 - y[1])
    #
    #     # energy
    #     if_ydot2 = - 0.1 * y[2] ** 7
    #     else_ydot2 = 0
    #     if self.modification:
    #         h = self.x0 + 3. / (1. + numpy.exp(- (y[0] + 0.5) / 0.1))
    #     else:
    #         h = 4 * (y[0] - self.x0) + where(y[2] < 0., if_ydot2, else_ydot2)
    #     ydot[2] = self.tt * (self.r * (h - y[2] + self.Ks * c_pop1))
    #
    #     # dot_m(t) variable definition here
    #     ydot[3] = self.tt * (self.r2*(-y[3] + abs(self.Istim)*15 + self.Kf * c_pop3))
    #
    #     if self.n_stim[0] > 0 and y[3] > self.threshold:
    #     # if y[3] > 2:
    #         self.x0 = numpy.array([-1.2])
    #     elif self.n_stim[0] > 0 :
    #     # else:
    #         self.x0 = numpy.array([-2.2])
    #
    #     return ydot

    def dfun(self, x, c, local_coupling=0.0, stimulus=0.0):
        r"""
        Computes the derivatives of the state variables of the Epileptor
        with respect to time.

        Implementation note: we expect this version of the Epileptor to be used
        in a vectorized manner. Concretely, y has a shape of (6, n) where n is
        the number of nodes in the network. An consequence is that
        the original use of if/else is translated by calculated both the true
        and false forms and mixing them using a boolean mask.

        Variables of interest to be used by monitors: -y[0] + y[3]

            .. math::
                \dot{x_{1}} &=& y_{1} - f_{1}(x_{1}, x_{2}) - z + I_{ext1} \\
                \dot{y_{1}} &=& c - d x_{1}^{2} - y{1} \\
                \dot{z} &=&
                \begin{cases}
                r(4 (x_{1} - x_{0}) - z-0.1 z^{7}) & \text{if } x<0 \\
                r(4 (x_{1} - x_{0}) - z) & \text{if } x \geq 0
                \end{cases} \\
                \dot{x_{2}} &=& -y_{2} + x_{2} - x_{2}^{3} + I_{ext2} + 0.002 g - 0.3 (z-3.5) \\
                \dot{y_{2}} &=& 1 / \tau (-y_{2} + f_{2}(x_{2}))\\
                \dot{g} &=& -0.01 (g - 0.1 x_{1})

        where:
            .. math::
                f_{1}(x_{1}, x_{2}) =
                \begin{cases}
                a x_{1}^{3} - b x_{1}^2 & \text{if } x_{1} <0\\
                -(slope - x_{2} + 0.6(z-4)^2) x_{1} &\text{if }x_{1} \geq 0
                \end{cases}

        and:
            .. math::
                f_{2}(x_{2}) =
                \begin{cases}
                0 & \text{if } x_{2} <-0.25\\
                a_{2}(x_{2} + 0.25) & \text{if } x_{2} \geq -0.25
                \end{cases}

        """
        if isinstance(stimulus, numpy.ndarray):  # and stimulus.any() > 0.0:

            self.Istim = stimulus[0, :, 0]  # stimulus shape (nr_var, nr_regions, 1)

        elif isinstance(stimulus, float) and stimulus != 0.0:
            print("Error in stimulus argument in 3D epileptor model.")
            return

        x_ = x.reshape(x.shape[:-1]).T  # x shape (nr_var, nr_regions, 1); x_ shape (nr_regions, nr_var)
        c_ = c.reshape(c.shape[:-1]).T  # c shape (nr_cvar, nr_regions, 1); c_ shape (nr_regions, nr_cvar)
        Iext = self.Iext + local_coupling * x[0, :, 0]

        deriv = _numba_dfun(x_, c_,
                            self.x0, Iext, self.a, self.b, self.tt, self.Kvf,
                            self.c, self.d, self.r, self.r2, self.Ks, self.Kf, self.modification, self.Istim,
                            self.threshold)
        return deriv.T[..., numpy.newaxis]