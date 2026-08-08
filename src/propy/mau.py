from propy.propeller import Propeller, ScalarOrArray

from dataclasses import dataclass
from typing import ClassVar, Callable
from functools import cached_property

from numpy.polynomial.polynomial import Polynomial
from numpy.typing import NDArray
from numpy import float64, array


@dataclass(frozen=True)
class MAUPropeller(Propeller):
    """

    References
    ----------
        [1] J. Chun Suh, C. Sup Lee, Polynomial representation for MAU-Propeller Open Water Characteristics
    """

    blades: int = 4

    blades_min: ClassVar[int] = 3
    blades_max: ClassVar[int] = 6

    @property
    def area_ratio_min(self) -> float:
        match self.blades:
            case 3:
                return 0.35
            case 4:
                return 0.4
            case 5:
                return 0.5
            case 6:
                return 0.55
            case _:
                return float('NaN')

    @property
    def area_ratio_max(self) -> float:
        match self.blades:
            case 3:
                return 0.50
            case 4:
                return 0.70
            case 5:
                return 0.80
            case 6:
                return 0.85
            case _:
                return float('NaN')
            
    @property
    def pd_ratio_min(self) -> float:
        match self.blades:
            case 3 | 5:
                return 0.4
            case 4 | 6:
                return 0.5
            case _:
                return float('NaN')
    
    @property
    def pd_ratio_max(self) -> float:
        match self.blades:
            case 3:
                return 1.2
            case 4 | 5:
                return 1.6
            case 6:
                return 1.5
            case _:
                return float('NaN')
    
    @cached_property
    def kt(self) -> Callable[[ScalarOrArray], NDArray[float64]]:
        return lambda j: array(j)

    @cached_property
    def kq(self) -> Callable[[ScalarOrArray], NDArray[float64]]:
        return lambda j: array(j)

    @property
    def _kt3_35(self) -> NDArray[float64]:
        return array([
            - 0.0674536 * self.pd_ratio**0 +
            + 0.5892164 * self.pd_ratio**1 +
            - 0.1971395 * self.pd_ratio**2 +
            + 0.0521675 * self.pd_ratio**3,

            - 0.0751015 * self.pd_ratio**0 +
            - 0.3569108 * self.pd_ratio**1 +
            + 0.2951303 * self.pd_ratio**2 +
            - 0.0967409 * self.pd_ratio**3,

            - 0.3041660 * self.pd_ratio**0 +
            + 0.0901521 * self.pd_ratio**1 +
            + 0.3234639 * self.pd_ratio**2 +
            - 0.1886770 * self.pd_ratio**3,

            + 0.1063169 * self.pd_ratio**0 +
            - 0.1536279 * self.pd_ratio**1 +
            - 0.0067481 * self.pd_ratio**2 +
            + 0.0488185 * self.pd_ratio**3,
        ], dtype=float64)

    @property 
    def _kq3_35(self) -> NDArray[float64]:
        return array([
            - 0.00176065 * self.pd_ratio**0 +
            + 0.02144957 * self.pd_ratio**1 +
            + 0.01563138 * self.pd_ratio**2 +
            + 0.01464783 * self.pd_ratio**3,

            - 0.00183090 * self.pd_ratio**0 +
            - 0.01321039 * self.pd_ratio**1 +
            - 0.02596498 * self.pd_ratio**2 +
            + 0.00833566 * self.pd_ratio**3,

            - 0.00895205 * self.pd_ratio**0 +
            + 0.02653115 * self.pd_ratio**1 +
            - 0.00510880 * self.pd_ratio**2 +
            - 0.00761559 * self.pd_ratio**3,

            - 0.03270370 * self.pd_ratio**0 +
            + 0.02510493 * self.pd_ratio**1 +
            - 0.00052158 * self.pd_ratio**2 +
            - 0.00142337 * self.pd_ratio**3,
        ], dtype=float64)
