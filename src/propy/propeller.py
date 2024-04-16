from abc import ABC, abstractmethod
from dataclasses import dataclass, replace


from numpy import pi, ndarray, array, linspace, inf
from scipy.optimize import root_scalar, minimize
from scipy.interpolate import CubicSpline


@dataclass
class Propeller(ABC):
    """
    Fields
    ------
    blades: int
        The amount of blades on the propeller
    diameter: float
        The diameter of the propeller in [m], must be >0.
    area_ratio: float
        The expanded area ratio of the propeller, which is defined as the ratio of the expanded blade area and the
        disk-area of the propeller.
    pd_ratio: float
        The ratio between the pitch [m] and the diameter [m] of the propeller.
    """
    blades:     int = 4
    diameter:   float = 1.0
    area_ratio: float = 0.5
    pd_ratio:   float = 0.8

    def eta(self, j):
        """
        Efficiency curve of this propeller

        This function calculates the efficiency curve of this propeller as a function of the advance ratio. The advance
        ratio can be defined as a single point or an array of points. When the advance ratio is outside the valid
        range of this propeller (kt < 0), a NaN is returned.

        The advance ratio is defined as:
            j = v / n / d

        The efficiency is defined as:
            eta = t * v / 2 / pi / q / n

        Where:
            - v: speed of the vessel [m/s]
            - n: rotation speed [1/s] or [Hz]
            - d: diameter of the propeller [m]
            - t: thrust of the propeller [N]
            - q: torque of the motor [Nm]

        Parameters
        ----------
        j : float or array-like
            The advance-ratio's where the efficiency should be calculated

        Returns
        -------
        float or array-like
            The efficiency of the propeller
        """
        kt = self.kt(j)
        kq = self.kq(j)
        res = j * float('NaN')

        eta_valid = j < self.j_max
        if isinstance(j, ndarray):
            res[eta_valid] = kt[eta_valid] * j[eta_valid] / 2 / pi / kq[eta_valid]
        elif eta_valid:
            res = kt * j / 2 / pi / kq

        return res

    def minimum_area_ratio(self, thrust, immersion, single_screw=False, rho=1025):
        """
        Calculate the minimum required expanded area ratio to prevent cavitation

        In general, a lower expanded area ratio will lead to a more efficient propeller. Unfortunately there's a limit
        where the propeller will start cavitating. This limit is roughly approximated by Keller's formula [1].

        References
        ----------
            [1] J. auf dem Keller, Enige Aspecten bij het ontwerpen van Scheepsschroeven, Schip en werf, 1966

        Parameters
        ----------
        thrust: float
            The maximum expected thrust [N].
        immersion: float
            The minimum expected immersion below the waterline [m].
        single_screw: bool
            True when the ship has a single screw and this has a non-smooth wake.
        rho: float
            The density of the fluid [kg/m^3].

        Returns
        -------
        float
            An estimation of the minimum expanded area ratio of the propeller to prevent cavitation.
        """
        min_area_ratio = (1.3 + 0.3 * self.blades) * thrust / self.diameter**2 / (1e5 + rho * 9.81 * immersion - 1700)
        if single_screw:
            min_area_ratio += 0.2
        return min_area_ratio

    def optimize_for_t_v(self, thrust, velocity, n_max=1e300, q_max=1e300, diameter_max=inf, immersion=inf,
                         single_screw=False, rho=1025):
        """
        Optimize the propeller for a given thrust and velocity

        This propeller will be optimized for a given amount of thrust and velocity. The optimization will attempt to
        optimize the efficiency of the propeller at a given velocity. The result will be returned in the form of a new
        propeller of the same type, but with the following fields optimized:
         - diameter: The total diameter of the propeller
         - area_ratio: The expanded area ratio
         - pd_ratio: The pitch/diameter ratio
        The optimization occurs via the method explained in [1].

        References
        ----------
        .. [1] G. Kuiper, The Wageningen propeller series, MARIN Publication 92-001, 1992

        Parameters
        ----------
        thrust: float
            The required thrust for this working point [N]
        velocity: float
            The velocity of the propeller [m/s]
        n_max: float
            The maximum rotational speed [1/s] or [Hz]
        q_max: float
            The maximum required torque [Nm]
        diameter_max: float
            The maximum allowed diameter [m]
        immersion: float
            The minimum expected immersion below the waterline [m].
        single_screw: bool
            True when the ship has a single screw and this has a non-smooth wake.
        rho: float
            The density of the fluid [kg/m^3].

        Returns
        -------
        Propeller
            A new propeller with optimum diameter, area_ratio and pd_ratio
        """
        def losses(x):
            p = replace(self, diameter=x[0], area_ratio=x[1], pd_ratio=x[2])
            ktj2 = thrust / rho / velocity ** 2 / x[0] ** 2
            return 1 - p.eta(p._find_j_for_ktj2(ktj2))

        def cavitation_margin(x):
            p = replace(self, diameter=x[0], area_ratio=x[1], pd_ratio=x[2])
            min_ear = p.minimum_area_ratio(thrust, immersion, single_screw=single_screw, rho=rho)
            return x[1] - min_ear

        def n_margin(x):
            p = replace(self, diameter=x[0], area_ratio=x[1], pd_ratio=x[2])
            _, n, _, _, _, _ = p.calculate_operating_point(thrust, velocity, rho=rho)
            return n_max - n

        def q_margin(x):
            p = replace(self, diameter=x[0], area_ratio=x[1], pd_ratio=x[2])
            q, _, _, _, _, _ = p.calculate_operating_point(thrust, velocity, rho=rho)
            return q_max - q

        # noinspection PyTypeChecker
        opt_res = minimize(
            fun=losses,
            x0=array([
                self.diameter,
                self.area_ratio,
                self.pd_ratio]
            ),
            bounds=[
                (0, min(2*immersion, diameter_max)),
                (self.area_ratio_min, self.area_ratio_max),
                (self.pd_ratio_min, self.pd_ratio_max)
            ],
            constraints=[
                {'type': 'ineq', 'fun': cavitation_margin},
                {'type': 'ineq', 'fun': n_margin},
                {'type': 'ineq', 'fun': q_margin},
            ]
        )

        prop = replace(self, diameter=opt_res.x[0], area_ratio=opt_res.x[1], pd_ratio=opt_res.x[2])

        return prop

    def optimize_for_t_n(self, thrust, n, velocity, immersion, single_screw=False, rho=1025):
        """
        Optimize the propeller for a given thrust and rotation speed

        This propeller will be optimized for a given amount of thrust and rotation speed. The optimization will attempt
        to optimize the efficiency of the propeller at a given velocity. The result will be returned in the form of a
        new propeller of the same type, but with the following fields optimized:
         - diameter: The diameter of the propeller
         - area_ratio: The expanded area ratio
         - pd_ratio: The pitch/diameter ratio
        The optimization occurs via the method explained in [1].

        References
        ----------
            [1] G. Kuiper, The Wageningen propeller series, MARIN Publication 92-001, 1992

        Parameters
        ----------
        thrust: float
            The required thrust for this working point [N]
        n: float
            The rotation speed of the propeller [1/s] or [Hz]
        velocity: float
            The velocity of the propeller [m/s]
        immersion: float
            The minimum expected immersion below the waterline [m].
        single_screw: bool
            True when the ship has a single screw and this has a non-smooth wake.
        rho: float
            The density of the fluid [kg/m^3].

        Returns
        -------
        Propeller
            A new propeller with optimum diameter, area_ratio and pd_ratio
        """
        ktj4 = thrust * n**2 / rho / velocity**4

        def losses(x):
            p = replace(self, area_ratio=x[0], pd_ratio=x[1])
            return 1 - p.eta(p._find_j_for_ktj4(ktj4))

        def cavitation_margin(x):
            p = replace(self, area_ratio=x[0], pd_ratio=x[1])
            d = velocity / n / p._find_j_for_ktj4(ktj4)
            p = replace(p, diameter=d)
            min_ear = p.minimum_area_ratio(thrust, immersion, single_screw=single_screw, rho=rho)
            return x[0] - min_ear

        def immersion_margin(x):
            p = replace(self, area_ratio=x[0], pd_ratio=x[1])
            d = velocity / n / p._find_j_for_ktj4(ktj4)
            return immersion - d/2

        # noinspection PyTypeChecker
        opt_res = minimize(
            fun=losses,
            x0=array([self.area_ratio, self.pd_ratio]),
            bounds=[(self.area_ratio_min, self.area_ratio_max),
                    (self.pd_ratio_min, self.pd_ratio_max)],
            constraints=[
                {'type': 'ineq', 'fun': cavitation_margin},
                {'type': 'ineq', 'fun': immersion_margin}
            ]
        )

        prop = replace(self, area_ratio=opt_res.x[0], pd_ratio=opt_res.x[1])
        diameter = velocity / n / prop._find_j_for_ktj4(ktj4)
        prop = replace(prop, diameter=diameter)

        return prop

    def optimize_for_p_n(self, power, n, velocity, immersion, single_screw=False, rho=1025):
        """
        Optimize the propeller for a given motor power and rotation speed

        This propeller will be optimized for a given amount of motor power and rotation speed. The optimization will
        attempt to optimize the efficiency of the propeller at a given velocity. The result will be returned in the form
        of a new propeller of the same type, but with the following fields optimized:
         - diameter: The diameter of the propeller
         - area_ratio: The expanded area ratio
         - pd_ratio: The pitch/diameter ratio
        The optimization occurs via the method explained in [1].

        References
        ----------
            [1] G. Kuiper, The Wageningen propeller series, MARIN Publication 92-001, 1992

        Parameters
        ----------
        power: float
            The motor power for this working point [N]
        n: float
            The rotation speed of the propeller [1/s] or [Hz]
        velocity: float
            The velocity of the propeller [m/s]
        immersion: float
            The minimum expected immersion below the waterline [m].
        single_screw: bool
            True when the ship has a single screw and this has a non-smooth wake.
        rho: float
            The density of the fluid [kg/m^3].

        Returns
        -------
        Propeller
            A new propeller with optimum diameter, area_ratio and pd_ratio
        """
        kqj5 = power * n**2 / 2 / pi / rho / velocity**5

        def losses(x):
            p = replace(self, area_ratio=x[0], pd_ratio=x[1])
            return 1 - p.eta(p._find_j_for_kqj5(kqj5))

        def cavitation_margin(x):
            p = replace(self, area_ratio=x[0], pd_ratio=x[1])
            j = p._find_j_for_kqj5(kqj5)
            d = velocity / n / j
            p = replace(p, diameter=d)
            thrust = p.kt(j) * rho * n**2 * d**4
            min_ear = p.minimum_area_ratio(thrust, immersion, single_screw=single_screw, rho=rho)
            return x[0] - min_ear

        def immersion_margin(x):
            p = replace(self, area_ratio=x[0], pd_ratio=x[1])
            d = velocity / n / p._find_j_for_kqj5(kqj5)
            return immersion - d/2

        # noinspection PyTypeChecker
        opt_res = minimize(
            fun=losses,
            x0=array([self.area_ratio, self.pd_ratio]),
            bounds=[(self.area_ratio_min, self.area_ratio_max),
                    (self.pd_ratio_min, self.pd_ratio_max)],
            constraints=[
                {'type': 'ineq', 'fun': cavitation_margin},
                {'type': 'ineq', 'fun': immersion_margin}
            ]
        )

        prop = replace(self, area_ratio=opt_res.x[0], pd_ratio=opt_res.x[1])
        diameter = velocity / n / prop._find_j_for_kqj5(kqj5)
        prop = replace(prop, diameter=diameter)

        return prop

    def __post_init__(self):
        if not (self.diameter > 0):
            raise ValueError(f'Diameter (= {self.diameter}) must be > 0')

        if not (isinstance(self.blades, int)):
            raise TypeError(f'The amount of blades (= {self.blades}) must be an integer')

        if not (self.blades >= self.blades_min):
            raise ValueError(f'Amount of blades (= {self.blades}) must be >= {self.blades_min}')

        if not (self.blades <= self.blades_max):
            raise ValueError(f'Amount of blades (= {self.blades}) must be <= {self.blades_max}')

        if not (self.area_ratio >= self.area_ratio_min):
            raise ValueError(f'Area ratio (= {self.area_ratio}) must be >= {self.area_ratio_min}')

        if not (self.area_ratio <= self.area_ratio_max):
            raise ValueError(f'Area ratio (= {self.area_ratio}) must be <= {self.area_ratio_max}')

        if not (self.pd_ratio >= self.pd_ratio_min):
            raise ValueError(f'Pitch/Diameter ratio (= {self.pd_ratio}) must be >= {self.pd_ratio_min}')

        if not (self.pd_ratio <= self.pd_ratio_max):
            raise ValueError(f'Pitch/Diameter ratio (= {self.pd_ratio}) must be <= {self.pd_ratio_max}')

    @property
    @abstractmethod
    def j_max(self):
        """The maximum valid advance-ratio of this propeller"""
        pass

    @property
    @abstractmethod
    def blades_min(self):
        """The minimum amount of blades of this propeller type"""
        pass

    @property
    @abstractmethod
    def blades_max(self):
        """The maximum amount of blades of this propeller type"""
        pass

    @property
    @abstractmethod
    def area_ratio_min(self):
        """The minimum expanded area ratio of this propeller type"""
        pass

    @property
    @abstractmethod
    def area_ratio_max(self):
        """The maximum expanded area ratio of this propeller type"""
        pass

    @property
    @abstractmethod
    def pd_ratio_min(self):
        """The minimum pitch/diameter ratio of this propeller type"""
        pass

    @property
    @abstractmethod
    def pd_ratio_max(self):
        """The maximum pitch/diameter ratio of this propeller type"""
        pass

    @abstractmethod
    def kt(self, j):
        """
        Thrust coefficient of the propeller

        This function calculates the thrust coefficient curve of this propeller as a function of the advance ratio.
        The advance ratio can be defined as a single point or an array of points.

        The advance ratio is defined as:
            j = v / n / d

        The thrust coefficient is defined as:
            kt = t / rho / n^2 / d^4

        Where:
            - v: speed of the vessel [m/s]
            - n: rotation speed [1/s] or [Hz]
            - d: diameter of the propeller [m]
            - t: thrust of the propeller [N]
            - rho: density of the fluid [kg/m^3]

        Parameters
        ----------
        j : float or array-like
            The advance-ratio's where the thrust coefficient should be calculated

        Returns
        -------
        float or array-like
            The thrust coefficients of the propeller
        """
        pass

    @abstractmethod
    def kq(self, j):
        """
        Torque coefficient of the propeller

        This function calculates the torque coefficient curve of this propeller as a function of the advance ratio.
        The advance ratio can be defined as a single point or an array of points.

        The advance ratio is defined as:
            j = v / n / d

        The thrust coefficient is defined as:
            kq = q / rho / n^2 / d^5

        Where:
            - v: speed of the vessel [m/s]
            - n: rotation speed [1/s] or [Hz]
            - d: diameter of the propeller [m]
            - q: torque of the propeller [Nm]
            - rho: density of the fluid [kg/m^3]

        Parameters
        ----------
        j : float or array-like
            The advance-ratio's where the thrust coefficient should be calculated

        Returns
        -------
        float or array-like
            The torque coefficients of the propeller
        """
        pass

    def kt_inv(self, kt):
        """
        The inverse function of the kt polynomial (vectorized)

        Calculates j as a function of a given kt. This function differs from kt_inv_single in that this approximation is
        made using a cubic spline interpolation. This way, it's very fast for calculating array's of kt's.

        Parameters
        ----------
        kt: float or array-like
            The thrust coefficients

        Returns
        -------
        j: float or array-like
            The advance ratio's
        """
        if not hasattr(self, '_kt_inv'):
            j = linspace(-0.1, self.j_max+0.1, 100)
            j = j[::-1]
            kts = self.kt(j)
            self._kt_inv = CubicSpline(kts, j)
        return self._kt_inv(kt)

    def kt_inv_single(self, kt):
        """
        The inverse function of the kt polynomial (single)

        Calculates j as a function of a given kt. This function differs from kt_inv in that this approximation is
        made using a root finding algorithm. This way, it's more precise, but only a single value can be calculated at a
        time.

        Parameters
        ----------
        kt: float
            The thrust coefficient

        Returns
        -------
        j: float
            The advance ratio
        """
        return root_scalar(
            f=lambda j: self.kt(j) - kt,
            bracket=[-1e-9, self.j_max],
            x0=1e-3
        ).root

    def kq_inv(self, kq):
        """
        The inverse function of the kq polynomial (vectorized)

        Calculates j as a function of a given kq. This function differs from kq_inv_single in that this approximation is
        made using a cubic spline interpolation. This way, it's very fast for calculating array's of kq's.

        Parameters
        ----------
        kq: float or array-like
            The torque coefficients

        Returns
        -------
        j: float or array-like
            The advance ratio's
        """
        if not hasattr(self, '_kq_inv'):
            j = linspace(-0.1, self.j_max+0.1, 100)
            j = j[::-1]
            kqs = self.kq(j)
            self._kq_inv = CubicSpline(kqs, j)
        return self._kq_inv(kq)

    def kq_inv_single(self, kq):
        """
        The inverse function of the kq polynomial (single)

        Calculates j as a function of a given kq. This function differs from kq_inv in that this approximation is
        made using a root finding algorithm. This way, it's more precise, but only a single value can be calculated at a
        time.

        Parameters
        ----------
        kq: float
            The torque coefficient

        Returns
        -------
        j: float
            The advance ratio
        """
        return root_scalar(
            f=lambda j: self.kq(j) - kq,
            bracket=[-1e-9, self.j_max],
            x0=1e-3
        ).root

    def calculate_operating_point(self, thrust, velocity, rho=1025.0):
        ktj2 = thrust / rho / velocity**2 / self.diameter**2
        j = self._find_j_for_ktj2(ktj2)
        kt = ktj2 * j**2
        kq = self.kq(j)
        eta = kt * j / 2 / pi / kq
        n = velocity / j / self.diameter
        torque = kq * rho * n**2 * self.diameter**5
        return torque, n, j, kt, kq, eta

    def _find_j_for_ktj2(self, ktj2):
        return root_scalar(
            f=lambda j: self.kt(j)/j**2 - ktj2,
            bracket=[1e-9, self.j_max],
            x0=0.8*self.j_max
        ).root

    def _find_j_for_ktj4(self, ktj4):
        return root_scalar(
            f=lambda j: self.kt(j)/j**4 - ktj4,
            bracket=[1e-9, self.j_max],
            x0=0.8*self.j_max
        ).root

    def _find_j_for_kqj5(self, kqj5):
        return root_scalar(
            f=lambda j: self.kq(j)/j**5 - kqj5,
            bracket=[1e-9, self.j_max],
            x0=0.8*self.j_max
        ).root
