from abc import ABC, abstractmethod
from collections.abc import Iterable, Callable
from dataclasses import dataclass
from functools import lru_cache, cached_property
from typing import ClassVar, Self, Any, TypeVar
from math import cos, sin, sqrt, atan2, pi
from numpy import float64, zeros_like, linspace, array, ndarray
from numpy import atan2 as atan2_v
from numpy import sin as sin_v
from numpy.typing import NDArray
from numpy.linalg import solve
from scipy.interpolate import make_interp_spline
from scipy.optimize import root_scalar

from propy.optimization import slsqp, PropFunctionWrapper, OptimizationMethod


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

    @property
    @abstractmethod
    def area_ratio_min(self) -> float:
        pass

    @property
    @abstractmethod
    def area_ratio_max(self) -> float:
        pass

    @property
    @abstractmethod
    def pd_ratio_min(self) -> float:
        pass

    @property
    @abstractmethod
    def pd_ratio_max(self) -> float:
        pass

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

    @cached_property
    def j_max(self) -> float:
        """The maximum valid advance-ratio of this propeller"""
        return root_scalar(
            f=lambda j: float(self.kt(j)),
            bracket=(self.j_min, self.pd_ratio*1.5),
            xtol=1e-15, rtol=1e-15
        ).root

    @property
    def j_min(self) -> float:
        """The minimum valid advance-ratio of this propeller"""
        return 0.0

    @property
    def kt_max(self) -> float:
        return float(self.kt(self.j_min))

    @property
    def kt_min(self) -> float:
        return 0.0

    @property
    def kq_max(self) -> float:
        return float(self.kq(self.j_min))

    @property
    def kq_min(self) -> float:
        return float(self.kq(self.j_max))

    # Basic model as a function of the advance ratio (j)
    @cached_property
    @abstractmethod
    def kt(self) -> Callable[[ScalarOrArray], NDArray[float64]]:
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

    @cached_property
    @abstractmethod
    def kq(self) -> Callable[[ScalarOrArray], NDArray[float64]]:
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

    def eta(self, j: ScalarOrArray) -> NDArray[float64]:
        return array(self.kt(j) * j / 2 / pi / self.kq(j), dtype=float64)

    # Basic 4-quadrant model as a function of the advance angle (beta)
    @dataclass(frozen=True)
    class FourQuadrantFunction:
        amplitude: float
        phase: float

        def __call__(self, beta: ScalarOrArray) -> NDArray[float64]:
            return array(self.amplitude * sin_v(beta + self.phase), dtype=float64)

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
    @cached_property
    def kt_inv(self) -> Callable[[ScalarOrArray], NDArray[float64]]:
        """
        The inverse function of the kt polynomial

        This function returns a callable that calculates j as a function of a given kt. This is achieved using a
        spline interpolator. The interpolator is only generated once, after which it's cached for your convenience.

        Parameters of the callable
        --------------------------
        kt: float
            The thrust coefficient

        Callable returns
        ----------------
        j: float
            The advance ratio
        """
        j = linspace(self.j_max, self.j_min, 50)
        kt = self.kt(j)
        return make_interp_spline(kt, j, k=3)

    @cached_property
    def kq_inv(self) -> Callable[[ScalarOrArray], NDArray[float64]]:
        """
        The inverse function of the kq polynomial

        This function returns a callable that calculates j as a function of a given kq. This is achieved using a
        spline interpolator. The interpolator is only generated once, after which it's cached for your convenience.

        Parameters of the callable
        --------------------------
        kq: float
            The torque coefficient

        Callable returns
        ----------------
        j: float
            The advance ratio
        """
        j = linspace(self.j_max, self.j_min, 200)
        kq = self.kq(j)
        return make_interp_spline(kq, j, k=4)
    
    @cached_property
    def ktj2_inv(self) -> Callable[[ScalarOrArray], NDArray[float64]]:
        j_min = max(1e-30, self.j_min)
        j = linspace(self.j_max, j_min, 300)
        ktj2 = self.kt(j) / j**2
        return make_interp_spline(ktj2, j, k=4)
    
    @cached_property
    def kqj2_inv(self) -> Callable[[ScalarOrArray], NDArray[float64]]:
        j_min = max(1e-30, self.j_min)
        j = linspace(self.j_max, j_min, 300)
        kqj2 = self.kq(j) / j**2
        return make_interp_spline(kqj2, j, k=4)

    def find_j_for_vt(
            self,
            speed: ScalarOrArray,
            thrust: ScalarOrArray,
            rho: float = 1025.0
    ) -> NDArray[float64]:
        """
        Calculate the advance ratio givencast(Callable[[ScalarOrArray], ScalarOrArray], the speed and thrust.

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
        return self.ktj2_inv(ktj2)

    def find_j_for_vq(
            self,
            speed: ScalarOrArray,
            torque: ScalarOrArray,
            rho: float = 1025.0
    ) -> NDArray[float64]:
        """
        Calculate the advance ratio given the speed and torque

        Parameters
        ----------
        speed
            The speed of in flow into the propeller [m/s]
        torque
            The torque load on the propeller [Nm]
        rho
            The density of the water [kg/m^3], defaults to 1025 kg/m^3

        Returns
        -------
            The advance ratio of the propeller at the given work-point [-]
        """
        kqj2 = torque / rho / speed**2 / self.diameter**3
        return self.kqj2_inv(kqj2)

    def find_j_for_vn(
            self,
            speed: ScalarOrArray,
            rotation_speed: ScalarOrArray
    ) -> ScalarOrArray:
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
            The advance ratio of the propeller at the given work-point [-]. NOTE: The return type of this function is
            different from all similar functions. To prevent slowdown from explicitly casting to an array, the return
            type is equal to the input types.
        """
        return speed / rotation_speed / self.diameter
    
    def find_j_for_nq(
            self,
            rotation_speed: ScalarOrArray,
            torque: ScalarOrArray,
            rho: float = 1025.0
    ) -> NDArray[float64]:
        """
        Calculate the advance ratio given the rotation rate and torque.

        Parameters
        ----------
        rotation_speed
            The rate at which the propeller is rotating [Hz]
        torque
            The torque load on the propeller [Nm]

        Returns
        -------
            The advance ratio of the propeller at the given work-point [-]
        """
        kq = torque / rho / rotation_speed**2 / self.diameter**5
        return self.kq_inv(kq)
    
    def find_j_for_nt(
            self,
            rotation_speed: ScalarOrArray,
            thrust: ScalarOrArray,
            rho: float = 1025.0
    ) -> NDArray[float64]:
        """
        Calculate the advance ratio given the rotation rate and thrust.

        Parameters
        ----------
        rotation_speed
            The rate at which the propeller is rotating [Hz]
        thrust
            The thrust produced by the propeller [N]

        Returns
        -------
            The advance ratio of the propeller at the given work-point [-]
        """
        kt = thrust / rho / rotation_speed**2 / self.diameter**4
        return self.kt_inv(kt)
    
    def find_beta_for_vn(
            self,
            speed: ScalarOrArray,
            rotation_speed: ScalarOrArray
    ) -> ScalarOrArray:
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
            The advance angle of the propeller at the given work-point [rad]. NOTE: The return type of this function is
            different from all similar functions. To prevent slowdown from explicitly casting to an array, the return
            type is equal to the input types.
        """
        if isinstance(speed, ndarray):
            return atan2_v(speed, 0.7 * pi * rotation_speed * self.diameter)
            
        return atan2(speed, 0.7 * pi * rotation_speed * self.diameter)

    def find_tq_for_vn(
            self,
            speed: ScalarOrArray,
            rotation_speed: ScalarOrArray,
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
        speed_arr: NDArray[float64] = array(speed, dtype=float64)
        rotation_speed_arr: NDArray[float64] = array(rotation_speed, dtype=float64)

        is_1q = (((self.j_min * rotation_speed_arr * self.diameter) < speed_arr) &
                 (speed_arr < (self.j_max * rotation_speed_arr * self.diameter)))

        j = zeros_like(is_1q, dtype=float64)
        j[is_1q] = self.find_j_for_vn(speed_arr[is_1q], rotation_speed_arr[is_1q])
        j[~is_1q] = self.find_beta_for_vn(speed_arr[~is_1q], rotation_speed_arr[~is_1q])

        kt = zeros_like(is_1q, dtype=float64)
        kt[is_1q] = self.kt(j[is_1q])
        kt[~is_1q] = self.ct(j[~is_1q])

        kq = zeros_like(is_1q, dtype=float64)
        kq[is_1q] = self.kq(j[is_1q])
        kq[~is_1q] = self.cq(j[~is_1q])

        thrust = zeros_like(is_1q, dtype=float64)
        thrust[is_1q] = kt[is_1q] * rho * rotation_speed_arr[is_1q] ** 2 * self.diameter ** 4
        thrust[~is_1q] = (kt[~is_1q] * pi * rho * self.diameter**2 / 8 *
                          (speed_arr[~is_1q]**2 + (0.7 * pi * rotation_speed_arr[~is_1q] * self.diameter)**2))

        torque = zeros_like(is_1q, dtype=float64)
        torque[is_1q] = kq[is_1q] * rho * rotation_speed_arr[is_1q] ** 2 * self.diameter ** 5
        torque[~is_1q] = (kq[~is_1q] * pi * rho * self.diameter ** 3 / 8 *
                          (speed_arr[~is_1q] ** 2 + (0.7 * pi * rotation_speed_arr[~is_1q] * self.diameter) ** 2))

        return thrust, torque

    def find_nq_for_vt(
            self,
            speed: ScalarOrArray,
            thrust: ScalarOrArray,
            rho: float = 1025.0
    ) -> tuple[NDArray[float64], NDArray[float64]]:
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

    def find_vt_for_nq(
        self,
        rotation_speed: ScalarOrArray,
        torque: ScalarOrArray,
        rho: float = 1025.0
    ) -> tuple[NDArray[float64], NDArray[float64]]:
        """
        Calculate speed and thrust for a given rotation speed and torque.

        Parameters
        ----------
        rotation_speed
            The rate at which the propeller is rotating [Hz]
        torque
            The torque load on the propeller [Nm]
        rho
            The density of the water [kg/m^3], defaults to 1025 kg/m^3

        Returns
        -------
        tuple[float, float]
            The speed [m/s] and the thrust [N] of the propeller
        """
        j = self.find_j_for_nq(rotation_speed=rotation_speed, torque=torque, rho=rho)
        speed = j * rotation_speed * self.diameter
        thrust = self.kt(j) * rho * rotation_speed**2 * self.diameter**4
        return speed, thrust
    
    def find_vq_for_nt(
        self,
        rotation_speed: ScalarOrArray,
        thrust: ScalarOrArray,
        rho: float = 1025
    ) -> tuple[NDArray[float64], NDArray[float64]]:
        """
        Calculate speed and torque for a given rotation speed and thrust.

        Parameters
        ----------
        rotation_speed
            The rate at which the propeller is rotating [Hz]
        thrust
            The thrust produced by the propeller [N]
        rho
            The density of the water [kg/m^3], defaults to 1025 kg/m^3

        Returns
        -------
        tuple[float, float]
            The speed [m/s] and the torque [Nm] of the propeller
        """
        j = self.find_j_for_nt(rotation_speed=rotation_speed, thrust=thrust, rho=rho)
        speed = j * rotation_speed * self.diameter
        torque = self.kq(j) * rho * rotation_speed**2 * self.diameter**5
        return speed, torque

    def find_nt_for_vq(
        self,
        speed: ScalarOrArray,
        torque: ScalarOrArray,
        rho: float = 1025
    ) -> tuple[NDArray[float64], NDArray[float64]]:
        """
        Calculate rotation speed and thrust for a given speed and torque.

        Parameters
        ----------
        speed
            The speed of in flow into the propeller [m/s]
        torque
            The torque load on the propeller [Nm]
        rho
            The density of the water [kg/m^3], defaults to 1025 kg/m^3

        Returns
        -------
        tuple[float, float]
            The rotation speed [Hz] and the thrust [N] of the propeller
        """
        kqj2 = torque / rho / speed**2 / self.diameter**3
        j = self.kqj2_inv(kqj2)
        rotation_speed = speed / j / self.diameter
        thrust = self.kt(j) * rho * rotation_speed**2 * self.diameter**4
        return rotation_speed, thrust

    # Optimisation methods
    def optimize(
            self,
            objective: Callable[["Propeller"], float],
            constraints: Iterable[Callable[["Propeller"], float]] = (),
            method: OptimizationMethod = slsqp,
            diameter_min: float = 0.03,
            diameter_max: float = 30.0,
            verbose: bool = False
    ) -> Self:
        """
        Optimize the parameters of this propeller as to minimize the objective under the given constraints.

        The most common use-case of this function is to minimize the losses of a propeller on a given working point.
        It is usually relevant to take some constraints into account, for example a maximum torque or a tip-speed limit
        to prevent cavitation.

        Parameters
        ----------
        objective
            The function which needs to be minimized. The optimizer will call this function multiple times to check
            the quality of a certain propeller. The objective function needs to have 1 argument: a propeller and it
            must return a floating point number. This is commonly achieved using a lambda function, see the example
            below.
        constraints
            An itterable of constraint functions. The optimizer will call these functions multiple times to check the
            validity of a certain propeller. The constrain functions need to have 1 argument: a propeller and it must
            return a floating point number. The constrain is considered ok when the return value is >= 0.
        method
            The optimization method to use. Although it is possible to write a custom method, it is most convenient to
            use the default optimization methods defined in propy.optimization.
        diameter_min
            The minimum allowed propeller diameter. When this is chosen very low (in the order of millimeters), some
            optimizations can fail due to numerical instabilities.
        diameter_max
            The maximum allowed propeller diameter, this is usually relevant when designing within a limited volume
            claim.
        verbose
            Print the progress statements of the optimizer.
        
        Returns
        -------
        Propeller
            The resulting propeller from the optimization

        Raises
        ------
        RuntimeError
            When the optimization process exits in with an unsuccesful result. This can happen when the combination
            of constraints is infeasable for this type of propeller.

        Examples
        --------
        The code below demonstrates how the `optimize` function can be used to minimize the losses of a 3-bladed
        propeller when a limit on the torque needs to be taken into account.

        >>> from propy import WageningenBPropeller
        >>>
        >>> speed = 5
        >>> thrust = 1000
        >>> torque_limit = 60
        >>>
        >>> prop = WageningenBPropeller(
        ...     blades=3,
        ... ).optimize(
        ...     objective=lambda p: p.losses(speed, thrust),
        ...     constraints=[
        ...         lambda p: p.torque_margin(speed, thrust, torque_limit)
        ...     ]
        ... )
        >>> prop
        WageningenBPropeller(blades=3, diameter=0.379..., area_ratio=0.3..., pd_ratio=0.917...)
        """

        args = method(
            objective=PropFunctionWrapper(self, objective),
            constraints=tuple(PropFunctionWrapper(self, constraint) for constraint in constraints),
            bounds=(
                (diameter_min, self.diameter, diameter_max),
                (self.area_ratio_min, self.area_ratio, self.area_ratio_max),
                (self.pd_ratio_min, self.pd_ratio, self.pd_ratio_max)
            ),
            verbose=verbose
        )

        return self.new(self.blades, *args)

    def losses(self, speed: float, thrust: float, rho: float = 1025.) -> float:
        """
        Calculate the (relative) losses of the propeller at a certain working point.

        This function can be very convenient to use as an objective for the optimize function.

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
        float
            The relative losses in the propeller, equal to 1 - efficiency.
        """

        j = self.find_j_for_vt(speed, thrust, rho=rho)
        return 1 - float(self.eta(j))

    def cavitation_margin(
            self,
            thrust: float,
            immersion: float,
            rho: float = 1025.0,
            single_screw: bool = False
    ) -> float:
        """
        Calculate the (normalized) minimum area ratio to prevent cavitaion according to the Keller criterion.

        The result is normalized relative to the maximum area ratio for this propeller type. This way, the optimizer
        weighs all the constraints in a similar way.

        Parameters
        ----------
        thrust
            The thrust produced by the propeller [N]
        immersion
            The depth at where the propeller operates [m]
        rho
            The density of the water [kg/m^3], defaults to 1025 kg/m^3
        single_screw
            When true, the minimum area-ration is higher due to a different flow-field.

        Returns
        -------
        float
            The normalized maximum area ratio, is >= 0 when it satisfies the constraint.
        """

        min_area_ratio = ((1.3 + 0.3 * self.blades) * thrust / self.diameter ** 2 /
                          (1e5 + rho * 9.81 * immersion - 1700))
        if single_screw:
            min_area_ratio += 0.2
        return (self.area_ratio - min_area_ratio) / self.area_ratio_max

    def rotation_speed_margin(
            self,
            speed: float,
            thrust: float,
            rotation_speed_max: float,
            rho: float = 1025.0
    ) -> float:
        """
        Calculate the (normalized) required rotation speed, can be a constraint to prevent driveshaft vibrations.

        The result is normalized relative to the given maximum rotation speed. This way, the optimizer weighs all the
        constraints in a similar way.

        Parameters
        ----------
        speed
            The speed of in flow into the propeller [m/s]
        thrust
            The thrust produced by the propeller [N]
        rotation_speed_max
            The maximum allowed rotation speed for this constraint [Hz]
        rho
            The density of the water [kg/m^3], defaults to 1025 kg/m^3

        Returns
        -------
        float
            The normalized rotation speed, is >= 0 when it satisfies the constraint.
        """

        n, _ = self.find_nq_for_vt(speed, thrust, rho=rho)
        return (rotation_speed_max - float(n)) / rotation_speed_max

    def torque_margin(self, speed: float, thrust: float, torque_max: float, rho: float = 1025.0) -> float:
        """
        Calculate the (normalized) required torqeu, can be a constraint to prevent gears from breaking.

        The result is normalized relative to the given maximum torque. This way, the optimizer weighs all the
        constrains in a similar way.

        Parameters
        ----------
        speed
            The speed of in flow into the propeller [m/s]
        thrust
            The thrust produced by the propeller [N]
        torque_max
            The maximum allowed torque for this constraint [Nm]
        rho
            The density of the water [kg/m^3], defaults to 1025 kg/m^3

        Returns
        -------
        float
            The normalized torque, is >= 0 when it satisfies the constraint.
        """

        _, q = self.find_nq_for_vt(speed, thrust, rho=rho)
        return (torque_max - float(q)) / torque_max

    def tip_speed_margin(self, speed: float, thrust: float, tip_speed_max: float, rho: float = 1025.0) -> float:
        """
        Calculate the (normalized) tip-speed, can be a constraint to prevent cavitaion.

        The result is normalized relative to the given maximum tip speed. This way, the optimizer weighs all the
        constraints in a similar way.

        Parameters
        ----------
        speed
            The speed of in flow into the propeller [m/s]
        thrust
            The thrust produced by the propeller [N]
        tip_speed_max
            The maximum allowed tip speed for this constraint [m/s]
        rho
            The density of the water [kg/m^3], defaults to 1025 kg/m^3

        Returns
        -------
        float
            The normalized tip speed, is >= 0 when it satisfies the constraint.
        """

        n, _ = self.find_nq_for_vt(speed, thrust, rho=rho)
        return (tip_speed_max - self.diameter * pi * float(n)) / tip_speed_max
