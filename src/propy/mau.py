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

    @property
    def _kt3_50(self) -> NDArray[float64]:
        return array([
            - 0.0687378 * self.pd_ratio**0 +
            + 0.5457044 * self.pd_ratio**1 +
            - 0.0578749 * self.pd_ratio**2 +
            - 0.0038145 * self.pd_ratio**3,

            - 0.1212302 * self.pd_ratio**0 +
            - 0.2609504 * self.pd_ratio**1 +
            - 0.0095168 * self.pd_ratio**2 +
            + 0.1021598 * self.pd_ratio**3,

            - 0.2465525 * self.pd_ratio**0 +
            - 0.0942721 * self.pd_ratio**1 +
            + 0.5995814 * self.pd_ratio**2 +
            - 0.3804743 * self.pd_ratio**3,

            + 0.1727268 * self.pd_ratio**0 +
            - 0.2307054 * self.pd_ratio**1 +
            + 0.0085561 * self.pd_ratio**2 +
            + 0.0854031 * self.pd_ratio**3,
        ], dtype=float64)
    
    @property
    def _kq3_50(self) -> NDArray[float64]:
        return array([
            - 0.00065957 * self.pd_ratio**0 +
            + 0.01089628 * self.pd_ratio**1 +
            + 0.03104104 * self.pd_ratio**2 +
            + 0.01546352 * self.pd_ratio**3,

            + 0.00339191 * self.pd_ratio**0 +
            - 0.01137696 * self.pd_ratio**1 +
            - 0.03246998 * self.pd_ratio**2 +
            + 0.00237390 * self.pd_ratio**3,

            - 0.01072531 * self.pd_ratio**0 +
            - 0.01646474 * self.pd_ratio**1 +
            + 0.02046463 * self.pd_ratio**2 +
            - 0.00298331 * self.pd_ratio**3,

            - 0.03558418 * self.pd_ratio**0 +
            + 0.08969163 * self.pd_ratio**1 +
            - 0.07184621 * self.pd_ratio**2 +
            + 0.01870359 * self.pd_ratio**3,
        ], dtype=float64)

    @property
    def _kt4_40(self) -> NDArray[float64]:
        return array([
            - 0.0001474 * self.pd_ratio**0 +
            + 0.4187610 * self.pd_ratio**1 +
            + 0.0350059 * self.pd_ratio**2 +
            - 0.0438544 * self.pd_ratio**3,

            - 0.2900881 * self.pd_ratio**0 +
            + 0.2356777 * self.pd_ratio**1 +
            - 0.3545542 * self.pd_ratio**2 +
            + 0.1559505 * self.pd_ratio**3,

            - 0.2644724 * self.pd_ratio**0 +
            + 0.2546326 * self.pd_ratio**1 +
            + 0.0299011 * self.pd_ratio**2 +
            - 0.0762877 * self.pd_ratio**3,

            - 0.0381288 * self.pd_ratio**0 +
            - 0.0327118 * self.pd_ratio**1 +
            + 0.0370220 * self.pd_ratio**2 +
            + 0.0012230 * self.pd_ratio**3,
        ], dtype=float64)
    
    @property
    def _kq4_40(self) -> NDArray[float64]:
        return array([
            + 0.00324114 * self.pd_ratio**0 +
            + 0.00025192 * self.pd_ratio**1 +
            + 0.05960882 * self.pd_ratio**2 +
            - 0.00739601 * self.pd_ratio**3,

            + 0.00203276 * self.pd_ratio**0 +
            - 0.00108519 * self.pd_ratio**1 +
            - 0.05641272 * self.pd_ratio**2 +
            + 0.02353945 * self.pd_ratio**3,

            - 0.03698804 * self.pd_ratio**0 +
            + 0.06398395 * self.pd_ratio**1 +
            - 0.00303819 * self.pd_ratio**2 +
            - 0.01376306 * self.pd_ratio**3,

            - 0.02878678 * self.pd_ratio**0 +
            + 0.00903704 * self.pd_ratio**1 +
            - 0.00312009 * self.pd_ratio**2 +
            + 0.00402354 * self.pd_ratio**3,
        ], dtype=float64)

    @property
    def _kt4_55(self) -> NDArray[float64]:
        return array([
            - 0.0560577 * self.pd_ratio**0 +
            + 0.5440128 * self.pd_ratio**1 +
            - 0.0131719 * self.pd_ratio**2 +
            - 0.0313747 * self.pd_ratio**3,

            - 0.1937437 * self.pd_ratio**0 +
            - 0.1383095 * self.pd_ratio**1 +
            - 0.0009202 * self.pd_ratio**2 +
            + 0.0495503 * self.pd_ratio**3,

            - 0.3207175 * self.pd_ratio**0 +
            + 0.2505594 * self.pd_ratio**1 +
            - 0.0267239 * self.pd_ratio**2 +
            - 0.0331890 * self.pd_ratio**3,

            + 0.1415830 * self.pd_ratio**0 +
            - 0.2201395 * self.pd_ratio**1 +
            + 0.1222395 * self.pd_ratio**2 +
            - 0.0206896 * self.pd_ratio**3,
        ], dtype=float64)
    
    @property
    def _kq4_55(self) -> NDArray[float64]:
        return array([
            + 0.01129100 * self.pd_ratio**0 +
            - 0.03487319 * self.pd_ratio**1 +
            + 0.10633678 * self.pd_ratio**2 +
            - 0.01919152 * self.pd_ratio**3,

            - 0.01389722 * self.pd_ratio**0 +
            + 0.04873902 * self.pd_ratio**1 +
            - 0.11009982 * self.pd_ratio**2 +
            + 0.03767585 * self.pd_ratio**3,

            - 0.01521688 * self.pd_ratio**0 +
            - 0.05244194 * self.pd_ratio**1 +
            + 0.09886425 * self.pd_ratio**2 +
            - 0.03868991 * self.pd_ratio**3,

            + 0.00740959 * self.pd_ratio**0 +
            - 0.01423152 * self.pd_ratio**1 +
            - 0.00556094 * self.pd_ratio**2 +
            + 0.00650457 * self.pd_ratio**3,
        ], dtype=float64)