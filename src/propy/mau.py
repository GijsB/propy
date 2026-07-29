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
