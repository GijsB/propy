from abc import ABC, abstractmethod
from collections.abc import Iterable, Callable
from dataclasses import dataclass
from functools import lru_cache, cached_property
from typing import ClassVar, Self
from numpy import pi, array, atan2, cos, sin, sqrt, logical_and, broadcast_arrays, atleast_1d, float64
from numpy.linalg import solve
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import root_scalar, minimize


@dataclass(frozen=True)
class WorkingPoint:
    thrust:         float | ArrayLike = 0
    speed:          float | ArrayLike = 0
    immersion:      float = float('inf')
    rho:            float = 1025
    single_screw:   bool = False


@dataclass(frozen=True)
class PerformancePoint:
    torque:         NDArray[float64]
    rotation_speed: NDArray[float64]
    j:              NDArray[float64]
    kt:             NDArray[float64]
    kq:             NDArray[float64]
    eta:            NDArray[float64]


@dataclass(frozen=True)
class WorkingPoint4Q:
    rotation_speed: float | ArrayLike = 0
    speed:          float | ArrayLike = 0
    rho:            float = 1025


@dataclass(frozen=True)
class PerformancePoint4Q:
    torque:         NDArray[float64]
    thrust:         NDArray[float64]
    beta:           NDArray[float64]
    ct:             NDArray[float64]
    cq:             NDArray[float64]


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

    def find_performance(self, speed, thrust, rho=1025.) -> PerformancePoint:
        j = self.find_j(speed, thrust, rho)
        kt = self.kt(j)
        kq = self.kq(j)
        n = speed / j / self.diameter

        return PerformancePoint(
            j=j,
            kt=kt,
            kq=kq,
            eta=kt * j / 2 / pi / kq,
            torque=kq * rho * n ** 2 * self.diameter ** 5,
            rotation_speed=n,
        )

    def find_j(self, speed, thrust, rho=1025.) -> NDArray[float64]:
        speed, thrust = broadcast_arrays(*atleast_1d(speed, thrust))
        ktj2 = thrust / rho / speed ** 2 / self.diameter ** 2
        return self._find_j_for_ktj2s(ktj2)

    def _find_j_for_ktj2(self, ktj2: float) -> float:
        return float(root_scalar(
            f=lambda j: self.kt(j) / j ** 2 - ktj2,
            bracket=[1e-9, self.j_max],
            x0=0.8 * self.j_max
        ).root)

    def _find_j_for_ktj2s(self, ktj2s: ArrayLike) -> NDArray[float64]:
        return array([self._find_j_for_ktj2(ktj2) for ktj2 in ktj2s])

    def optimize(self,
                 objective: Callable[[Self], float],
                 constraints: Iterable[Callable[[Self], float]] = (),
                 optimizer: Callable = minimize,
                 diameter_min: float = 1e-3,
                 diameter_max: float = float('inf'),
                 verbose: bool = False,) -> Self:

        @dataclass(frozen=True)
        class ConstraintFunction:
            base: Propeller
            func: Callable

            def __call__(self, x):
                return self.func(self.base.new(self.base.blades, *x))

        opt_res = optimizer(
            fun=lambda x: objective(self.new(self.blades, *x)),
            x0=array([
                self.diameter,
                self.area_ratio,
                self.pd_ratio]
            ),
            bounds=[
                (diameter_min, diameter_max),
                (self.area_ratio_min, self.area_ratio_max),
                (self.pd_ratio_min, self.pd_ratio_max)
            ],
            constraints=[{'type': 'ineq',
                          'fun': ConstraintFunction(self, cfun)} for cfun in constraints]
        )

        if verbose:
            print(opt_res)

        return self.new(self.blades, *opt_res.x)

    @classmethod
    @lru_cache
    def new(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def losses(self, speed, thrust, rho=1025.) -> float:
        pp = self.find_performance(speed, thrust, rho=rho)
        return 1 - pp.eta

    def cavitation_margin(self, thrust, immersion, rho=1025.0, single_screw=False) -> float:
        min_area_ratio = ((1.3 + 0.3 * self.blades) * thrust / self.diameter ** 2 /
                          (1e5 + rho * 9.81 * immersion - 1700))
        if single_screw:
            min_area_ratio += 0.2
        return (self.area_ratio - min_area_ratio) / self.area_ratio_max

    def rotation_speed_margin(self, wp: WorkingPoint, rotation_speed_max: float) -> float:
        pp = self.find_performance(wp.speed, wp.thrust, wp.rho)
        return (rotation_speed_max - pp.rotation_speed) / rotation_speed_max

    def torque_margin(self, wp: WorkingPoint, torque_max: float) -> float:
        pp = self.find_performance(wp.speed, wp.thrust, wp.rho)
        return (torque_max - pp.torque) / torque_max

    def tip_speed_margin(self, wp: WorkingPoint, tip_speed_max: float) -> float:
        pp = self.find_performance(wp.speed, wp.thrust, wp.rho)
        return (tip_speed_max - self.diameter * pi * pp.rotation_speed) / tip_speed_max

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
    def kt(self) -> Callable[[float | ArrayLike], float | ArrayLike]:
        """
        Thrust coefficient of the propeller

        This function returns a callable to calculate the thrust coefficient curve of this propeller as a function of
        the advance ratio. The advance ratio can be defined as a single point or an array of points.

        The advance ratio is defined as:
            j = speed / rotation_speed / d

        The thrust coefficient is defined as:
            kt = thrust / rho / rotation_speed^2 / d^4

        Where:
            - speed: speed of the vessel [m/s]
            - rotation_speed: rotation speed [1/s] or [Hz]
            - d: diameter of the propeller [m]
            - thrust: thrust of the propeller [N]
            - rho: density of the fluid [kg/m^3]

        Returns
        -------
        Function(j)
            A callable that calculates the thrust coefficient of the propeller as a function of the advance ratio.
        """
        pass

    @property
    @abstractmethod
    def kq(self) -> Callable[[float | ArrayLike], float | ArrayLike]:
        """
        Torque coefficient of the propeller

        This function returns a callable to calculate the torque coefficient curve of this propeller as a function of
        the advance ratio. The advance ratio can be defined as a single point or an array of points.

        The advance ratio is defined as:
            j = speed / rotation_speed / d

        The torque coefficient is defined as:
            kq = torque / rho / rotation_speed^2 / d^5

        Where:
            - speed: speed of the vessel [m/s]
            - rotation_speed: rotation speed [1/s] or [Hz]
            - d: diameter of the propeller [m]
            - torque: torque of the propeller [Nm]
            - rho: density of the fluid [kg/m^3]

        Returns
        -------
        Function(j)
            A callable that calculates the torque coefficient of the propeller as a function of the advance ratio.
        """
        pass

    def eta(self, j):
        return self.kt(j) * j / 2 / pi / self.kq(j)

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

    @dataclass(frozen=True)
    class FourQuadrantFunction:
        amplitude: float
        phase: float

        def __call__(self, beta: float | ArrayLike) -> float | ArrayLike:
            return self.amplitude * sin(beta + self.phase)

    @cached_property
    def ct(self) -> FourQuadrantFunction:
        """Fit the 1-quadrant behaviour the propeller onto a  4-quadrant function and return the resulting function.

        With the 4-quadrant behaviour of a propeller, the thrust and torque can be calculated for every load angle. This
        means the propeller can also be used for generating and reversing cases. The result of this function is a very
        rough approximation of the actual behaviour, which cannot be determined from the 1-quadrant data exclusively. It
        should therefore not be relied upon for accuracy.

        The load angle is defined as:
            beta = atan(speed / 0.7 / pi / rotation_speed / diameter)
            beta = atan(j / 0.7 / pi)

        The 4-quadrant thrust coefficient is defined as:
            ct = 8 * thrust / (speed^2 + (0.7 * pi * rotation_speed * diameter)^2) / pi / rho / diameter^2
            ct = 8 * kt / pi / (j^2 + 0.7^2 * pi^2)

        Where:
            - j: is the advance ratio (speed / rotation_speed / diameter)
            - kt: is the 1-quadrant thrust coefficient
            - speed: speed of the vessel [m/s]
            - rotation_speed: rotation speed [1/s] or [Hz]
            - d: diameter of the propeller [m]
            - thrust: thrust of the propeller [N]
            - rho: density of the fluid [kg/m^3]

        Returns
        -------
        FourQuadrantFunction(beta)
            A function that returns the thrust coefficient of the propeller as a function the load angle beta
        """

        # The load angle at the maximum J (where kt=0)
        beta_max = atan2(self.j_max, 0.7 * pi)
        ct_min = self.kt_min * 8 / pi / (self.j_max**2 + 0.7**2 * pi**2)

        # The thrust coefficient at J=0 (and thus beta=0)
        beta_min = 0
        ct_max = self.kt_max * 8 / pi / (0.7**2 * pi**2)

        # Linearly fit the ct(beta) function on these two points
        (a_c, ), (a_s, ) = solve(
            [[cos(beta_min), sin(beta_min)],
             [cos(beta_max), sin(beta_max)]],
            [[ct_max],
             [ct_min]]
        )

        return Propeller.FourQuadrantFunction(
            amplitude=float(sqrt(a_c**2 + a_s**2)),
            phase=float(atan2(a_c, a_s))
        )

    @cached_property
    def cq(self) -> FourQuadrantFunction:
        """Fit the 1-quadrant behaviour the propeller onto a  4-quadrant function and return the resulting function.

        With the 4-quadrant behaviour of a propeller, the thrust and torque can be calculated for every load angle. This
        means the propeller can also be used for generating and reversing cases. The result of this function is a very
        rough approximation of the actual behaviour, which cannot be determined from the 1-quadrant data exclusively. It
        should therefore not be relied upon for accuracy.

        The load angle is defined as:
            beta = atan(speed / 0.7 / pi / rotation_speed / diameter)
            beta = atan(j / 0.7 / pi)

        The 4-quadrant torque coefficient is defined as:
            cq = 8 * torque / (speed^2 + (0.7 * pi * rotation_speed * diameter)^2) / pi / rho / diameter^3
            cq = 8 * kq / pi / (j^2 + 0.7^2 * pi^2)

        Where:
            - j: is the advance ratio (speed / rotation_speed / diameter)
            - kq: is the 1-quadrant torque coefficient
            - speed: speed of the vessel [m/s]
            - rotation_speed: rotation speed [1/s] or [Hz]
            - d: diameter of the propeller [m]
            - thrust: thrust of the propeller [N]
            - rho: density of the fluid [kg/m^3]

        Returns
        -------
        FourQuadrantFunction(beta)
            A function that returns the torque coefficient of the propeller as a function the load angle beta
        """

        # The load angle at the maximum J (where kq != 0)
        beta_max = atan2(self.j_max, 0.7 * pi)
        cq_min = self.kq_min * 8 / pi / (self.j_max**2 + 0.7**2 * pi**2)

        # The torque coefficient at J=0 (and thus beta=0)
        beta_min = 0
        cq_max = self.kq_max * 8 / pi / (0.7 ** 2 * pi ** 2)

        # Linearly fit the ct(beta) function on these two points
        (a_c,), (a_s,) = solve(
            [[cos(beta_min), sin(beta_min)],
             [cos(beta_max), sin(beta_max)]],
            [[cq_max],
             [cq_min]]
        )

        return Propeller.FourQuadrantFunction(
            amplitude=float(sqrt(a_c ** 2 + a_s ** 2)),
            phase=float(atan2(a_c, a_s))
        )

    def find_performance_4q(self, wp: WorkingPoint4Q) -> PerformancePoint4Q:
        """
        Calculate the 4-quadrant performance of this propeller at a given speed. When the workingpoint turns out to be
        in the 1-quadrant area, the more accurate 1-quadrant model is used.

        Parameters
        ----------
        wp
            The 4 quadrant working point defining the (rotation-) speed.

        Returns
        -------
            The performance at the given workingpoint.
        """

        # Cast working point data to arrays
        rotation_speed, speed = broadcast_arrays(*atleast_1d(wp.rotation_speed, wp.speed))

        # Assume 4-quadrant working point at first
        beta = atan2(speed, 0.7 * pi * rotation_speed * self.diameter)
        ct, cq = self.ct(beta), self.cq(beta)
        thrust = ct * (speed**2 + (0.7 * pi * rotation_speed * self.diameter)**2) * pi * wp.rho * self.diameter**2 / 8
        torque = cq * (speed**2 + (0.7 * pi * rotation_speed * self.diameter)**2) * pi * wp.rho * self.diameter**3 / 8

        # Substitute more accurate 1-quadrant data if it's available
        is_in_first_quadrant = logical_and(speed > 0, speed < (self.j_max * rotation_speed * self.diameter))
        j = speed[is_in_first_quadrant] / rotation_speed[is_in_first_quadrant] / self.diameter
        kt, kq = self.kt(j), self.kq(j)
        thrust[is_in_first_quadrant] = kt * wp.rho * rotation_speed[is_in_first_quadrant]**2 * self.diameter**4
        torque[is_in_first_quadrant] = kq * wp.rho * rotation_speed[is_in_first_quadrant]**2 * self.diameter**5

        return PerformancePoint4Q(
            beta=beta,
            ct=ct,
            cq=cq,
            thrust=thrust,
            torque=torque
        )

    @property
    @abstractmethod
    def j_max(self) -> float:
        """The maximum valid advance-ratio of this propeller"""
        pass

    @property
    def kt_max(self) -> float:
        return self.kt(0)

    @property
    def kt_min(self) -> float:
        return 0

    @property
    def kq_max(self) -> float:
        return self.kq(0)

    @property
    def kq_min(self) -> float:
        return self.kq(self.j_max)
