from abc import ABC, abstractmethod
from collections.abc import Iterable, Callable
from dataclasses import dataclass
from functools import lru_cache, cached_property
from typing import ClassVar, Self, Any, TypeVar
from math import cos, sin, sqrt, atan2, pi
from numpy import float64, array, zeros_like
from numpy import atan2 as atan2_v
from numpy import sin as sin_v
from numpy.typing import NDArray
from numpy.linalg import solve
from scipy.optimize import root_scalar, minimize


ScalarOrArray = TypeVar('ScalarOrArray', float, NDArray[float64])


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

    # Class housekeeping
    @classmethod
    @lru_cache
    def new(cls, *args: Any, **kwargs: Any) -> Self:
        return cls(*args, **kwargs)

    def __post_init__(self) -> None:
        if not (self.diameter > 0):
            raise ValueError(f'Diameter (= {self.diameter}) must be > 0')

        if not isinstance(self.blades, int):
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
    def j_max(self) -> float:
        """The maximum valid advance-ratio of this propeller"""
        pass

    @property
    def j_min(self) -> float:
        """The minimum valid advance-ratio of this propeller"""
        return 0.0

    @property
    def kt_max(self) -> float:
        return self.kt(self.j_min)

    @property
    def kt_min(self) -> float:
        return 0

    @property
    def kq_max(self) -> float:
        return self.kq(self.j_min)

    @property
    def kq_min(self) -> float:
        return self.kq(self.j_max)

    # Basic model as a function of the advance ratio (j)
    @property
    @abstractmethod
    def kt(self) -> Callable[[ScalarOrArray], ScalarOrArray]:
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
            A callable that calculates the thrust coefficient of the propeller as a function of the advance ratio.
        """
        pass

    @property
    @abstractmethod
    def kq(self) -> Callable[[ScalarOrArray], ScalarOrArray]:
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
            A callable that calculates the torque coefficient of the propeller as a function of the advance ratio.
        """
        pass

    def eta(self, j: ScalarOrArray) -> ScalarOrArray:
        return self.kt(j) * j / 2 / pi / self.kq(j)

    # Basic 4-quadrant model as a function of the advance angle (beta)
    @dataclass(frozen=True)
    class FourQuadrantFunction:
        amplitude: float
        phase: float

        def __call__(self, beta: ScalarOrArray) -> ScalarOrArray:
            result: ScalarOrArray = self.amplitude * sin_v(beta + self.phase)
            return result

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
            A function that returns the thrust coefficient of the propeller as a function the load angle beta
        """

        # The load angle at the maximum J (where kt=0)
        beta_max = atan2(self.j_max, 0.7 * pi)
        ct_min = self.kt_min * 8 / pi / (self.j_max**2 + 0.7**2 * pi**2)

        # The thrust coefficient at j_min
        beta_min = atan2(self.j_min, 0.7 * pi)
        ct_max = self.kt_max * 8 / pi / (0.7**2 * pi**2)

        # Linearly fit the ct(beta) function on these two points
        (a_c, ), (a_s, ) = solve(
            [[cos(beta_min), sin(beta_min)],
             [cos(beta_max), sin(beta_max)]],
            [[ct_max],
             [ct_min]]
        )

        return Propeller.FourQuadrantFunction(
            amplitude=sqrt(a_c**2 + a_s**2),
            phase=atan2(a_c, a_s)
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
            A function that returns the torque coefficient of the propeller as a function the load angle beta
        """

        # The load angle at the maximum J (where kq != 0)
        beta_max = atan2(self.j_max, 0.7 * pi)
        cq_min = self.kq_min * 8 / pi / (self.j_max**2 + 0.7**2 * pi**2)

        # The torque coefficient at J=0 (and thus beta=0)
        beta_min = atan2(self.j_min, 0.7 * pi)
        cq_max = self.kq_max * 8 / pi / (0.7 ** 2 * pi ** 2)

        # Linearly fit the ct(beta) function on these two points
        (a_c,), (a_s,) = solve(
            [[cos(beta_min), sin(beta_min)],
             [cos(beta_max), sin(beta_max)]],
            [[cq_max],
             [cq_min]]
        )

        return Propeller.FourQuadrantFunction(
            amplitude=sqrt(a_c ** 2 + a_s ** 2),
            phase=atan2(a_c, a_s)
        )

    # Inverse propeller model
    def find_j_for_vt(
            self,
            speed: float,
            thrust: float,
            rho: float = 1025.0
    ) -> float:
        """
        Calculate the advance ratio given the speed and thrust.

        Parameters
        ----------
        speed
            The speed of in flow into the propeller [m/s]
        thrust
            The thrust produced by the propeller [N]
        rho
            The density of the water [kg/m^3], defaults to 1025 kg/m^3

        Returns
        -------
            The advance ratio of the propeller at the given work-point [-]
        """
        ktj2 = thrust / rho / speed ** 2 / self.diameter ** 2
        return root_scalar(
            f=lambda j: self.kt(j) / j ** 2 - ktj2,
            bracket=(self.j_min + 1e-9, self.j_max)
        ).root

    def find_j_for_vt_vec(
            self,
            speed: NDArray[float64],
            thrust: NDArray[float64],
            rho: float = 1025.0) -> NDArray[float64]:
        """
        Calculate the advance ratio given arrays of the speed and thrust.

        Parameters
        ----------
        speed
            The speed of in flow into the propeller [m/s]
        thrust
            The thrust produced by the propeller [N]
        rho
            The density of the water [kg/m^3], defaults to 1025 kg/m^3

        Returns
        -------
            The advance ratio of the propeller at the given work-point [-]
        """
        return array([self.find_j_for_vt(s, t, rho=rho) for s, t in zip(speed, thrust)])

    def find_j_for_vn(
            self,
            speed: float,
            rotation_speed: float
    ) -> float:
        """
        Calculate the advance ratio given the speed and rotation rate.

        Parameters
        ----------
        speed
            The speed of in flow into the propeller [m/s]
        rotation_speed
            The rate at which the propeller is rotating [Hz]

        Returns
        -------
            The advance ratio of the propeller at the given work-point [-]
        """
        return speed / rotation_speed / self.diameter

    def find_j_for_vn_vec(
            self,
            speed: NDArray[float64],
            rotation_speed: NDArray[float64]
    ) -> NDArray[float64]:
        """
        Calculate the advance ratio given arrays of the speed and rotation rate.

        Parameters
        ----------
        speed
            The speed of in flow into the propeller [m/s]
        rotation_speed
            The rate at which the propeller is rotating [Hz]

        Returns
        -------
            The advance ratio of the propeller at the given work-point [-]
        """
        return speed / rotation_speed / self.diameter

    def find_beta_for_vn(
            self,
            speed: float,
            rotation_speed: float
    ) -> float:
        """
        Calculate the advance angle of the propeller given the speed and rotation rate.

        Parameters
        ----------
        speed
            The speed of in flow into the propeller [m/s]
        rotation_speed
            The rate at which the propeller is rotating [Hz]

        Returns
        -------
            The advance angle of the propeller at the given work-point [rad]
        """
        return atan2(speed, 0.7 * pi * rotation_speed * self.diameter)

    def find_beta_for_vn_vec(
            self,
            speed: NDArray[float64],
            rotation_speed: NDArray[float64]
    ) -> NDArray[float64]:
        """
        Calculate the advance angle of the propeller given arrays of the speed and rotation rate.

        Parameters
        ----------
        speed
            The speed of in flow into the propeller [m/s]
        rotation_speed
            The rate at which the propeller is rotating [Hz]

        Returns
        -------
            The advance angle of the propeller at the given work-point [rad]
        """
        return atan2_v(speed, 0.7 * pi * rotation_speed * self.diameter)

    def find_tq_for_vn(
            self,
            speed: float,
            rotation_speed: float,
            rho: float = 1025.0
    ) -> tuple[float, float]:
        """
        Calculate the thrust and torque for a given speed and rotation rate.

        Parameters
        ----------
        speed
            The speed of in flow into the propeller [m/s]
        rotation_speed
            The rate at which the propeller is rotating [Hz]
        rho
            The density of the water [kg/m^3], defaults to 1025 kg/m^3

        Returns
        -------
        tuple[float, float]
            The thrust [N] and torque [Nm] at the given work-point
        """
        if (self.j_min * rotation_speed * self.diameter) < speed < (self.j_max * rotation_speed * self.diameter):
            # Use more accurate 1-quadrant data if it's applicable
            j = self.find_j_for_vn(speed, rotation_speed)
            kt, kq = self.kt(j), self.kq(j)
            thrust = kt * rho * rotation_speed ** 2 * self.diameter ** 4
            torque = kq * rho * rotation_speed ** 2 * self.diameter ** 5
        else:
            # Fall back to the 4-quadrant model
            beta = self.find_beta_for_vn(speed, rotation_speed)
            ct, cq = self.ct(beta), self.cq(beta)
            thrust = ct * (speed**2 + (0.7 * pi * rotation_speed * self.diameter)**2) * pi * rho * self.diameter**2 / 8
            torque = cq * (speed**2 + (0.7 * pi * rotation_speed * self.diameter)**2) * pi * rho * self.diameter**3 / 8
        return thrust, torque

    def find_tq_for_vn_vec(
            self,
            speed: NDArray[float64],
            rotation_speed: NDArray[float64],
            rho: float = 1025.0
    ) -> tuple[NDArray[float64], NDArray[float64]]:
        """
        Calculate arrays of thrust and torque for a given speed and rotation rate.

        Parameters
        ----------
        speed
            The speed of in flow into the propeller [m/s]
        rotation_speed
            The rate at which the propeller is rotating [Hz]
        rho
            The density of the water [kg/m^3], defaults to 1025 kg/m^3

        Returns
        -------
        tuple[NDArray, NDArray]
            The thrust [N] and torque [Nm] at the given work-point
        """
        is_1q = (((self.j_min * rotation_speed * self.diameter) < speed) &
                 (speed < (self.j_max * rotation_speed * self.diameter)))

        j = zeros_like(is_1q, dtype=float64)
        j[is_1q] = self.find_j_for_vn_vec(speed[is_1q], rotation_speed[is_1q])
        j[~is_1q] = self.find_j_for_vn_vec(speed[~is_1q], rotation_speed[~is_1q])

        kt = zeros_like(is_1q, dtype=float64)
        kt[is_1q] = self.kt(j[is_1q])
        kt[~is_1q] = self.ct(j[~is_1q])

        kq = zeros_like(is_1q, dtype=float64)
        kq[is_1q] = self.kq(j[is_1q])
        kq[~is_1q] = self.cq(j[~is_1q])

        thrust = zeros_like(is_1q, dtype=float64)
        thrust[is_1q] = kt[is_1q] * rho * rotation_speed[is_1q] ** 2 * self.diameter ** 4
        thrust[~is_1q] = (kt[~is_1q] * pi * rho * self.diameter**2 / 8 *
                          (speed[~is_1q]**2 + (0.7 * pi * rotation_speed[~is_1q] * self.diameter)**2))

        torque = zeros_like(is_1q, dtype=float64)
        torque[is_1q] = kq[is_1q] * rho * rotation_speed[is_1q] ** 2 * self.diameter ** 5
        torque[~is_1q] = (kq[~is_1q] * pi * rho * self.diameter ** 3 / 8 *
                          (speed[~is_1q] ** 2 + (0.7 * pi * rotation_speed[~is_1q] * self.diameter) ** 2))

        return thrust, torque

    def find_nq_for_vt(
            self,
            speed: float,
            thrust: float,
            rho: float = 1025.0
    ) -> tuple[float, float]:
        """
        Calculate rotation speed and torque for a given speed and thrust.

        Parameters
        ----------
        speed
            The speed of in flow into the propeller [m/s]
        thrust
            The thrust produced by the propeller [N]
        rho
            The density of the water [kg/m^3], defaults to 1025 kg/m^3

        Returns
        -------
        tuple[float, float]
            The rotation-rate [Hz] and torque [Nm] at the given work-point
        """
        j = self.find_j_for_vt(speed, thrust, rho)
        kq = self.kq(j)
        rotation_speed = speed / j / self.diameter
        torque = kq * rho * rotation_speed ** 2 * self.diameter ** 5
        return rotation_speed, torque

    def find_nq_for_vt_vec(
            self,
            speed: NDArray[float64],
            thrust: NDArray[float64],
            rho: float = 1025.0
    ) -> tuple[NDArray[float64], NDArray[float64]]:
        """
        Calculate arrays of rotation speed and torque for a given speed and thrust.

        Parameters
        ----------
        speed
            The speed of in flow into the propeller [m/s]
        thrust
            The thrust produced by the propeller [N]
        rho
            The density of the water [kg/m^3], defaults to 1025 kg/m^3

        Returns
        -------
        tuple[NDArray, NDArray]
            The rotation-rate [Hz] and torque [Nm] at the given work-point
        """
        j = self.find_j_for_vt_vec(speed, thrust, rho)
        kq = self.kq(j)
        rotation_speed = speed / j / self.diameter
        torque = kq * rho * rotation_speed ** 2 * self.diameter ** 5
        return rotation_speed, torque

    # Optimisation methods
    def optimize(
            self,
            objective: Callable[[Self], float],
            constraints: Iterable[Callable[["Propeller"], float]] = (),
            diameter_min: float = 0.03,
            diameter_max: float = float('inf'),
            verbose: bool = False
    ) -> Self:

        @dataclass(frozen=True)
        class ConstraintFunction:
            base: Propeller
            func: Callable[[Propeller], float]

            def __call__(self, x: Any) -> float:
                x = (float(arg) for arg in x)
                return self.func(self.base.new(self.base.blades, *x))

        def objective_function(x: Any) -> float:
            x = (float(arg) for arg in x)
            return objective(self.new(self.blades, *x))

        # noinspection PyTypeChecker
        opt_res = minimize(
            fun=objective_function,
            x0=(
                self.diameter,
                self.area_ratio,
                self.pd_ratio,
            ),
            bounds=(
                (diameter_min, diameter_max),
                (self.area_ratio_min, self.area_ratio_max),
                (self.pd_ratio_min, self.pd_ratio_max),
            ),
            constraints=[{'type': 'ineq', 'fun': ConstraintFunction(self, cfun)} for cfun in constraints]
        )

        if verbose:
            print(opt_res)

        if not opt_res.success:
            raise RuntimeError(opt_res.message)

        return self.new(self.blades, *(float(arg) for arg in opt_res.x))

    def losses(self, speed: float, thrust: float, rho: float = 1025.) -> float:
        j = self.find_j_for_vt(speed, thrust, rho=rho)
        return 1 - self.eta(j)

    def cavitation_margin(self,
                          thrust: float,
                          immersion: float,
                          rho: float = 1025.0,
                          single_screw: bool = False) -> float:
        min_area_ratio = ((1.3 + 0.3 * self.blades) * thrust / self.diameter ** 2 /
                          (1e5 + rho * 9.81 * immersion - 1700))
        if single_screw:
            min_area_ratio += 0.2
        return (self.area_ratio - min_area_ratio) / self.area_ratio_max

    def rotation_speed_margin(self,
                              speed: float,
                              thrust: float,
                              rotation_speed_max: float,
                              rho: float = 1025.0) -> float:
        n, q = self.find_nq_for_vt(speed, thrust, rho=rho)
        return (rotation_speed_max - n) / rotation_speed_max

    def torque_margin(self, speed: float, thrust: float, torque_max: float, rho: float = 1025.0) -> float:
        n, q = self.find_nq_for_vt(speed, thrust, rho=rho)
        return (torque_max - q) / torque_max

    def tip_speed_margin(self, speed: float, thrust: float, tip_speed_max: float, rho: float = 1025.0) -> float:
        n, q = self.find_nq_for_vt(speed, thrust, rho=rho)
        return (tip_speed_max - self.diameter * pi * n) / tip_speed_max
