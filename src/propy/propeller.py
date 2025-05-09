from abc import ABC, abstractmethod
from collections.abc import Iterable, Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import ClassVar, Self
from numpy import pi, array
from numpy.typing import ArrayLike
from scipy.optimize import root_scalar, minimize


@dataclass(frozen=True)
class WorkingPoint:
    thrust:         float | ArrayLike   = 0
    speed:          float | ArrayLike   = 0
    immersion:      float   = float('inf')
    rho:            float   = 1025
    single_screw:   bool    = False


@dataclass(frozen=True)
class PerformancePoint:
    torque:         float | ArrayLike
    rotation_speed: float | ArrayLike
    j:              float | ArrayLike
    kt:             float | ArrayLike
    kq:             float | ArrayLike
    eta:            float | ArrayLike


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

    def find_performance(self, wp: WorkingPoint) -> PerformancePoint:
        j = self.find_j(wp)
        kt = self.kt(j)
        kq = self.kq(j)
        n = wp.speed / j / self.diameter

        return PerformancePoint(
            j= j,
            kt= kt,
            kq= kq,
            eta=kt * j / 2 / pi / kq,
            torque=kq * wp.rho * n ** 2 * self.diameter ** 5,
            rotation_speed= n,
        )

    def find_j(self, wp: WorkingPoint):
        ktj2 = wp.thrust / wp.rho / wp.speed ** 2 / self.diameter ** 2
        try:
            return self._find_j_for_ktj2(ktj2)
        except (TypeError, ValueError):
            return self._find_j_for_ktj2s(ktj2)

    def _find_j_for_ktj2(self, ktj2: float) -> float:
        return float(root_scalar(
            f=lambda j: self.kt(j) / j ** 2 - ktj2,
            bracket=[1e-9, self.j_max],
            x0=0.8 * self.j_max
        ).root)

    def _find_j_for_ktj2s(self, ktj2s: ArrayLike) -> ArrayLike:
        return array([root_scalar(
            f=lambda j: self.kt(j) / j ** 2 - ktj2,
            bracket=[1e-9, self.j_max],
            x0=0.8 * self.j_max
        ).root for ktj2 in ktj2s])

    def optimize(self,
                 objective: Callable[[Self], float],
                 constraints: Iterable[Callable[[Self], float]] = (),
                 optimizer: Callable = minimize,
                 diameter_min : float = 1e-3,
                 diameter_max : float = float('inf'),
                 verbose: bool = True,) -> Self:

        @dataclass(frozen=True)
        class ConstraintFunction:
            base: Propeller
            func: Callable

            def __call__(self, x):
                return self.func(self.base.new(self.base.blades, *x))

        opt_res = optimizer(
            fun = lambda x: objective(self.new(self.blades, *x)),
            x0 = array([
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

    def losses(self, wp: WorkingPoint) -> float:
        pp = self.find_performance(wp)
        return 1 - pp.eta

    def cavitation_margin(self, wp: WorkingPoint) -> float:
        min_area_ratio = ((1.3 + 0.3 * self.blades) * wp.thrust / self.diameter ** 2 /
                          (1e5 + wp.rho * 9.81 * wp.immersion - 1700))
        if wp.single_screw:
            min_area_ratio += 0.2
        return (self.area_ratio - min_area_ratio) / self.area_ratio_max

    def rotation_speed_margin(self, wp: WorkingPoint, rotation_speed_max: float) -> float:
        pp = self.find_performance(wp)
        return (rotation_speed_max - pp.rotation_speed) / rotation_speed_max

    def torque_margin(self, wp: WorkingPoint, torque_max: float) -> float:
        pp = self.find_performance(wp)
        return (torque_max - pp.torque) / torque_max

    def tip_speed_margin(self, wp: WorkingPoint, tip_speed_max: float) -> float:
        pp = self.find_performance(wp)
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
    def kt(self):
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
            j = speed / rotation_speed / d

        The thrust coefficient is defined as:
            kq = torque / rho / rotation_speed^2 / d^5

        Where:
            - speed: speed of the vessel [m/s]
            - rotation_speed: rotation speed [1/s] or [Hz]
            - d: diameter of the propeller [m]
            - torque: torque of the propeller [Nm]
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