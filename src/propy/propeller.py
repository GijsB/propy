from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import ClassVar

from numpy import pi, ndarray, array, inf
from scipy.optimize import root_scalar, minimize


@dataclass(frozen=True)
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

    blades_min:     ClassVar[int] = -1
    blades_max:     ClassVar[int] = -1
    area_ratio_min: ClassVar[float] = float('NaN')
    area_ratio_max: ClassVar[float] = float('NaN')
    pd_ratio_min:   ClassVar[float] = float('NaN')
    pd_ratio_max:   ClassVar[float] = float('NaN')

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

    def optimize_efficiency_for_t_v(self, thrust, velocity, n_max=None, q_max=None, diameter_max=None, immersion=None,
                                    single_screw=False, tip_speed_max=None, rho=1025):
        """
        Optimize the efficiency of a propeller for a given thrust and velocity

        This propeller will be optimized for a given amount of thrust and velocity. The result will be returned in the
        form of a new propeller of the same type, but with the following fields optimized:
         - diameter: The total diameter of the propeller
         - area_ratio: The expanded area ratio
         - pd_ratio: The pitch/diameter ratio
        The optimization is inspired by [1].

        References
        ----------
           [1] G. Kuiper, The Wageningen propeller series, MARIN Publication 92-001, 1992

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
        tip_speed_max: float
            The maximum allowed tip speed [m/s]
        rho: float
            The density of the fluid [kg/m^3].

        Returns
        -------
        Propeller
            A new propeller with optimum diameter, area_ratio and pd_ratio
        """
        def losses(x):
            p = self.new(blades=self.blades, diameter=x[0], area_ratio=x[1], pd_ratio=x[2])
            ktj2 = thrust / rho / velocity ** 2 / p.diameter**2
            return 1 - p.eta(p._find_j_for_ktj2(ktj2))

        def cavitation_margin(x):
            p = self.new(blades=self.blades, diameter=x[0], area_ratio=x[1], pd_ratio=x[2])
            min_ear = p.minimum_area_ratio(thrust, immersion, single_screw=single_screw, rho=rho)
            return x[1] - min_ear

        def n_margin(x):
            p = self.new(blades=self.blades, diameter=x[0], area_ratio=x[1], pd_ratio=x[2])
            _, n, _, _, _, _ = p.calculate_operating_point(thrust, velocity, rho=rho)
            return n_max - n

        def q_margin(x):
            p = self.new(blades=self.blades, diameter=x[0], area_ratio=x[1], pd_ratio=x[2])
            q, _, _, _, _, _ = p.calculate_operating_point(thrust, velocity, rho=rho)
            return q_max - q

        def tip_speed(x):
            p = self.new(blades=self.blades, diameter=x[0], area_ratio=x[1], pd_ratio=x[2])
            _, n, _, _, _, _ = p.calculate_operating_point(thrust, velocity, rho=rho)
            return tip_speed_max - p.diameter * n * pi

        constraints = []
        if immersion:
            constraints.append({'type': 'ineq', 'fun': cavitation_margin})
            if diameter_max:
                diameter_max = min(diameter_max, immersion*2)
            else:
                diameter_max = immersion*2
        else:
            if diameter_max is None:
                diameter_max = inf
        if n_max:
            constraints.append({'type': 'ineq', 'fun': n_margin})
        if q_max:
            constraints.append({'type': 'ineq', 'fun': q_margin})
        if tip_speed_max:
            constraints.append({'type': 'ineq', 'fun': tip_speed})

        # noinspection PyTypeChecker
        opt_res = minimize(
            fun=losses,
            x0=array([
                self.diameter,
                self.area_ratio,
                self.pd_ratio]
            ),
            bounds=[
                (1e-3, diameter_max),
                (self.area_ratio_min, self.area_ratio_max),
                (self.pd_ratio_min, self.pd_ratio_max)
            ],
            constraints=constraints
        )

        if not opt_res.success:
            print(f'Optimization unsuccessful, reason: {opt_res.message}')

        return self.new(blades=self.blades, diameter=opt_res.x[0], area_ratio=opt_res.x[1], pd_ratio=opt_res.x[2])

    def optimize_eff_with_extremes(self, nom_point, constraint_points=(), n_max=None, q_max=None, diameter_max=None,
                                   immersion=None, single_screw=False, tip_speed_max=None, rho=1025):
        def losses(x, thrust, velocity):
            p = self.new(blades=self.blades, diameter=x[0], area_ratio=x[1], pd_ratio=x[2])
            ktj2 = thrust / rho / velocity ** 2 / p.diameter**2
            return 1 - p.eta(p._find_j_for_ktj2(ktj2))

        def cavitation_margin(x, thrust, velocity):
            p = self.new(blades=self.blades, diameter=x[0], area_ratio=x[1], pd_ratio=x[2])
            min_ear = p.minimum_area_ratio(thrust, immersion, single_screw=single_screw, rho=rho)
            return x[1] - min_ear

        def n_margin(x, thrust, velocity):
            p = self.new(blades=self.blades, diameter=x[0], area_ratio=x[1], pd_ratio=x[2])
            _, n, _, _, _, _ = p.calculate_operating_point(thrust, velocity, rho=rho)
            return n_max - n

        def q_margin(x, thrust, velocity):
            p = self.new(blades=self.blades, diameter=x[0], area_ratio=x[1], pd_ratio=x[2])
            q, _, _, _, _, _ = p.calculate_operating_point(thrust, velocity, rho=rho)
            return q_max - q

        def tip_speed(x, thrust, velocity):
            p = self.new(blades=self.blades, diameter=x[0], area_ratio=x[1], pd_ratio=x[2])
            _, n, _, _, _, _ = p.calculate_operating_point(thrust, velocity, rho=rho)
            return tip_speed_max - p.diameter * n * pi

        constraint_points = [nom_point] + list(constraint_points)

        constraints = []
        for constraint_point in constraint_points:
            if immersion:
                constraints.append({'type': 'ineq', 'fun': cavitation_margin, 'args': constraint_point})
                if diameter_max:
                    diameter_max = min(diameter_max, immersion * 2)
                else:
                    diameter_max = immersion * 2
            else:
                if diameter_max is None:
                    diameter_max = inf
            if n_max:
                constraints.append({'type': 'ineq', 'fun': n_margin, 'args': constraint_point})
            if q_max:
                constraints.append({'type': 'ineq', 'fun': q_margin, 'args': constraint_point})
            if tip_speed_max:
                constraints.append({'type': 'ineq', 'fun': tip_speed, 'args': constraint_point})

        # noinspection PyTypeChecker
        opt_res = minimize(
            fun=lambda x: losses(x, nom_point[0], nom_point[1]),
            x0=array([
                self.diameter,
                self.area_ratio,
                self.pd_ratio]
            ),
            bounds=[
                (1e-3, diameter_max),
                (self.area_ratio_min, self.area_ratio_max),
                (self.pd_ratio_min, self.pd_ratio_max)
            ],
            constraints=constraints
        )

        if not opt_res.success:
            print(f'Optimization unsuccessful, reason: {opt_res.message}')

        return self.new(blades=self.blades, diameter=opt_res.x[0], area_ratio=opt_res.x[1], pd_ratio=opt_res.x[2])

    def optimize_bollard_thrust_for_q_n(self, q, n, diameter_max=None, immersion=None, single_screw=False, rho=1025):
        def neg_thrust(x):
            p = self.new(blades=self.blades, diameter=self.diameter, area_ratio=x[0], pd_ratio=x[1])
            d = (q / rho / n**2 / p.kq_max) ** (1 / 5)
            t = p.kt_max * rho * n**2 * d**4
            return -t

        def cavitation_margin(x):
            p = self.new(blades=self.blades, diameter=self.diameter, area_ratio=x[0], pd_ratio=x[1])
            d = (q / rho / n ** 2 / p.kq_max) ** (1 / 5)
            t = p.kt_max * rho * n ** 2 * d ** 4
            p = self.new(blades=p.blades, diameter=d, area_ratio=p.area_ratio, pd_ratio=p.pd_ratio)
            min_ear = p.minimum_area_ratio(t, immersion, single_screw=single_screw, rho=rho)
            return x[1] - min_ear

        constraints = []
        if immersion:
            constraints.append({'type': 'ineq', 'fun': cavitation_margin})
            if diameter_max:
                diameter_max = min(diameter_max, immersion * 2)
            else:
                diameter_max = immersion * 2

        def diameter_margin(x):
            p = self.new(blades=self.blades, diameter=self.diameter, area_ratio=x[0], pd_ratio=x[1])
            d = (q / rho / n ** 2 / p.kq_max) ** (1 / 5)
            return diameter_max - d

        if diameter_max:
            constraints.append({'type': 'ineq', 'fun': diameter_margin})

        # noinspection PyTypeChecker
        opt_res = minimize(
            fun=neg_thrust,
            x0=array([
                self.area_ratio,
                self.pd_ratio]
            ),
            bounds=[
                (self.area_ratio_min, self.area_ratio_max),
                (self.pd_ratio_min, self.pd_ratio_max)
            ],
            constraints=constraints
        )

        if not opt_res.success:
            print(f'Optimization unsuccessful, reason: {opt_res.message}')

        p = self.new(blades=self.blades, diameter=self.diameter, area_ratio=opt_res.x[0], pd_ratio=opt_res.x[1])
        diameter = (q / rho / n ** 2 / p.kq_max) ** (1 / 5)
        return self.new(blades=p.blades, diameter=diameter, area_ratio=p.area_ratio, pd_ratio=p.pd_ratio)

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
    def kt(self):
        """
        Thrust coefficient of the propeller

        This function returns a callable to calculate the thrust coefficient curve of this propeller as a function of
        the advance ratio. The advance ratio can be defined as a single point or an array of points.

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

    @property
    @abstractmethod
    def kq(self):
        """
        Torque coefficient of the propeller

        This function returns a callable to calculate the torque coefficient curve of this propeller as a function of
        the advance ratio. The advance ratio can be defined as a single point or an array of points.

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
        The inverse function of the kt polynomial (single)

        Calculates j as a function of a given kt. This is achieved using a root-finding algorithm. This way, it's more
        precise, but only a single value can be calculated at a time.

        Parameters
        ----------
        kt: float
            The thrust coefficient

        Returns
        -------
        j: float
            The advance ratio
        """
        if kt >= self.kt_max:
            return 0
        elif kt <= self.kq_min:
            return self.j_max
        else:
            return root_scalar(
                f=lambda j: self.kt(j) - kt,
                bracket=[0, self.j_max],
                x0=self.j_max * (kt - self.kt_max) / (self.kt_min - self.kt_max),
                rtol=1e-15, xtol=1e-15
            ).root

    def kq_inv(self, kq):
        """
        The inverse function of the kq polynomial (single)

        Calculates j as a function of a given kq. This is achieved using a root-finding algorithm. This way, it's
        precise, but only a single value can be calculated at a time.

        Parameters
        ----------
        kq: float
            The torque coefficient

        Returns
        -------
        j: float
            The advance ratio
        """
        if kq >= self.kq_max:
            return 0
        elif kq <= self.kq_min:
            return self.j_max
        else:
            return root_scalar(
                f=lambda j: self.kq(j) - kq,
                bracket=[0, self.j_max],
                x0=self.j_max * (kq - self.kq_max) / (self.kq_min - self.kq_max),
                rtol=1e-15, xtol=1e-15
            ).root

    def calculate_operating_point(self, thrust, velocity, rho=1025.0):
        """
        Calculate an operating point for this propeller.

        Parameters
        ----------
        thrust: float
            The required thrust [N]

        velocity: float
            The velocity of the incomming flow [m/s]

        rho: float (optional)
            The density of the fluid [kg/m^3]

        Returns
        -------
        torque: float
            The required torque on the shaft [Nm]

        n: float
            The required rotational speed of the propeller [Hz]

        j: float
            The advance ratio of this working point [-]

        kt: float
            The thrust coefficient of this working point [-]

        kq: float
            The torque coefficient of this working point [-]

        eta: float
            The efficiency of the propeller at this working point [-]
        """
        ktj2 = thrust / rho / velocity**2 / self.diameter**2
        j = self._find_j_for_ktj2(ktj2)
        kt = ktj2 * j**2
        kq = self.kq(j)
        eta = kt * j / 2 / pi / kq
        n = velocity / j / self.diameter
        torque = kq * rho * n**2 * self.diameter**5
        return torque, n, j, kt, kq, eta

    def calculate_operating_points(self, thrusts, velocities, rho=1025.0):
        """
        Calculate an operating points for this propeller.

        This is a vectorized version of the calculate_operating_point function.

        Parameters
        ----------
        thrusts: array-like
            The required thrusts [N]

        velocities: array-like
            The velocities of the incomming flow [m/s]

        rho: float (optional)
            The density of the fluid [kg/m^3]

        Returns
        -------
        torques: array-like
            The required torques on the shaft [Nm]

        ns: array-like
            The required rotational speed of the propeller [Hz]

        js: array-like
            The advance ratios of these working points [-]

        kts: array-like
            The thrust coefficients of these working points [-]

        kqs: array-like
            The torque coefficients of these workings point [-]

        etas: float
            The efficiencies of the propeller at these working points [-]
        """
        ktj2s = thrusts / rho / velocities**2 / self.diameter**2
        js = array([self._find_j_for_ktj2(ktj2) for ktj2 in ktj2s])
        kts = ktj2s * js**2
        kqs = self.kq(js)
        etas = kts * js / 2 / pi / kqs
        ns = velocities / js / self.diameter
        torques = kqs * rho * ns**2 * self.diameter**5
        return torques, ns, js, kts, kqs, etas

    @property
    @abstractmethod
    def j_max(self):
        """The maximum valid advance-ratio of this propeller"""
        pass

    @property
    def kt_max(self):
        return self.kt(0)

    @property
    def kt_min(self):
        return 0

    @property
    def kq_max(self):
        return self.kq(0)

    @property
    def kq_min(self):
        return self.kq(self.j_max)

    def _find_j_for_ktj2(self, ktj2):
        return root_scalar(
            f=lambda j: self.kt(j)/j**2 - ktj2,
            bracket=[1e-9, self.j_max],
            x0=0.8*self.j_max
        ).root

    @classmethod
    @lru_cache
    def new(cls, *args, **kwargs):
        return cls(*args, **kwargs)
