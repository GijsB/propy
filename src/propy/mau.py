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

    @property
    def _kt4_70(self) -> NDArray[float64]:
        return array([
            - 0.0646424 * self.pd_ratio**0 +
            + 0.4922363 * self.pd_ratio**1 +
            + 0.1272241 * self.pd_ratio**2 +
            - 0.0827072 * self.pd_ratio**3,

            - 0.1571431 * self.pd_ratio**0 +
            - 0.1229877 * self.pd_ratio**1 +
            - 0.1564544 * self.pd_ratio**2 +
            + 0.1377097 * self.pd_ratio**3,

            - 0.3534932 * self.pd_ratio**0 +
            + 0.0304649 * self.pd_ratio**1 +
            + 0.2333862 * self.pd_ratio**2 +
            - 0.1334871 * self.pd_ratio**3,

            + 0.2717910 * self.pd_ratio**0 +
            - 0.2950153 * self.pd_ratio**1 +
            + 0.1176184 * self.pd_ratio**2 +
            - 0.0068707 * self.pd_ratio**3,
        ], dtype=float64)
    
    @property
    def _kq4_70(self) -> NDArray[float64]:
        return array([
            + 0.00509697 * self.pd_ratio**0 +
            - 0.02036556 * self.pd_ratio**1 +
            + 0.10108860 * self.pd_ratio**2 +
            - 0.01652730 * self.pd_ratio**3,

            + 0.01334815 * self.pd_ratio**0 +
            - 0.03587294 * self.pd_ratio**1 +
            - 0.04250861 * self.pd_ratio**2 +
            + 0.02397980 * self.pd_ratio**3,

            - 0.04510605 * self.pd_ratio**0 +
            + 0.04482430 * self.pd_ratio**1 +
            - 0.01413650 * self.pd_ratio**2 +
            - 0.00653318 * self.pd_ratio**3,

            + 0.02385343 * self.pd_ratio**0 +
            - 0.04462985 * self.pd_ratio**1 +
            + 0.03040715 * self.pd_ratio**2 +
            - 0.00511894 * self.pd_ratio**3,
        ], dtype=float64)

    @property
    def _kt5_50(self) -> NDArray[float64]:
        return array([
            - 0.0074234 * self.pd_ratio**0 +
            + 0.4604075 * self.pd_ratio**1 +
            + 0.0506650 * self.pd_ratio**2 +
            - 0.0488094 * self.pd_ratio**3,

            - 0.1981725 * self.pd_ratio**0 +
            - 0.0879429 * self.pd_ratio**1 +
            + 0.0368657 * self.pd_ratio**2 +
            - 0.0012424 * self.pd_ratio**3,

            - 0.3964577 * self.pd_ratio**0 +
            + 0.3914554 * self.pd_ratio**1 +
            - 0.1279939 * self.pd_ratio**2 +
            + 0.0139456 * self.pd_ratio**3,

            + 0.0367620 * self.pd_ratio**0 +
            - 0.0790636 * self.pd_ratio**1 +
            + 0.0396169 * self.pd_ratio**2 +
            - 0.0063409 * self.pd_ratio**3,
        ], dtype=float64)
    
    @property
    def _kq5_50(self) -> NDArray[float64]:
        return array([
            + 0.01077106 * self.pd_ratio**0 +
            - 0.02771245 * self.pd_ratio**1 +
            + 0.10378129 * self.pd_ratio**2 +
            - 0.02213732 * self.pd_ratio**3,

            - 0.00082556 * self.pd_ratio**0 +
            - 0.01661583 * self.pd_ratio**1 +
            - 0.02247197 * self.pd_ratio**2 +
            + 0.00752553 * self.pd_ratio**3,

            - 0.00952693 * self.pd_ratio**0 +
            - 0.02635570 * self.pd_ratio**1 +
            + 0.04782063 * self.pd_ratio**2 +
            - 0.01425715 * self.pd_ratio**3,

            - 0.01052510 * self.pd_ratio**0 +
            + 0.00565588 * self.pd_ratio**1 +
            - 0.01184727 * self.pd_ratio**2 +
            + 0.00468082 * self.pd_ratio**3,
        ], dtype=float64)

    @property
    def _kt5_65(self) -> NDArray[float64]:
        return array([
            - 0.0764546 * self.pd_ratio**0 +
            + 0.6294264 * self.pd_ratio**1 +
            - 0.0654519 * self.pd_ratio**2 +
            - 0.0077443 * self.pd_ratio**3,

            - 0.1345568 * self.pd_ratio**0 +
            - 0.3376999 * self.pd_ratio**1 +
            + 0.2200446 * self.pd_ratio**2 +
            - 0.0477772 * self.pd_ratio**3,

            - 0.3899990 * self.pd_ratio**0 +
            + 0.3320150 * self.pd_ratio**1 +
            - 0.0182617 * self.pd_ratio**2 +
            - 0.0382110 * self.pd_ratio**3,

            + 0.0888334 * self.pd_ratio**0 +
            - 0.1624223 * self.pd_ratio**1 +
            + 0.0642455 * self.pd_ratio**2 +
            + 0.0015954 * self.pd_ratio**3,
        ], dtype=float64)
    
    @property
    def _kq5_65(self) -> NDArray[float64]:
        return array([
            - 0.00178930 * self.pd_ratio**0 +
            + 0.01940973 * self.pd_ratio**1 +
            + 0.04949018 * self.pd_ratio**2 +
            + 0.00213476 * self.pd_ratio**3,

            + 0.02362586 * self.pd_ratio**0 +
            - 0.13757970 * self.pd_ratio**1 +
            + 0.11819780 * self.pd_ratio**2 +
            - 0.04281347 * self.pd_ratio**3,

            + 0.00279677 * self.pd_ratio**0 +
            - 0.01523687 * self.pd_ratio**1 +
            - 0.00897452 * self.pd_ratio**2 +
            + 0.01077238 * self.pd_ratio**3,

            - 0.00550960 * self.pd_ratio**0 +
            - 0.00174664 * self.pd_ratio**1 +
            + 0.00408763 * self.pd_ratio**2 +
            - 0.00172860 * self.pd_ratio**3,
        ], dtype=float64)

    @property
    def _kt5_80(self) -> NDArray[float64]:
        return array([
            - 0.0128748 * self.pd_ratio**0 +
            + 0.3192149 * self.pd_ratio**1 +
            + 0.3379987 * self.pd_ratio**2 +
            - 0.1450585 * self.pd_ratio**3,

            - 0.1388527 * self.pd_ratio**0 +
            - 0.2423269 * self.pd_ratio**1 +
            - 0.0076956 * self.pd_ratio**2 +
            + 0.0553135 * self.pd_ratio**3,

            - 0.4703999 * self.pd_ratio**0 +
            + 0.2855924 * self.pd_ratio**1 +
            + 0.1192634 * self.pd_ratio**2 +
            - 0.0997479 * self.pd_ratio**3,

            + 0.3010865 * self.pd_ratio**0 +
            - 0.4185277 * self.pd_ratio**1 +
            + 0.1686530 * self.pd_ratio**2 +
            - 0.0092374 * self.pd_ratio**3,
        ], dtype=float64)
    
    @property
    def _kq5_80(self) -> NDArray[float64]:
        return array([
            + 0.00729184 * self.pd_ratio**0 +
            - 0.02298344 * self.pd_ratio**1 +
            + 0.10367990 * self.pd_ratio**2 +
            - 0.01521706 * self.pd_ratio**3,

            + 0.02803990 * self.pd_ratio**0 +
            - 0.14434922 * self.pd_ratio**1 +
            + 0.12027810 * self.pd_ratio**2 +
            - 0.04279097 * self.pd_ratio**3,

            - 0.00306121 * self.pd_ratio**0 +
            - 0.03035455 * self.pd_ratio**1 +
            - 0.00355549 * self.pd_ratio**2 +
            + 0.01225184 * self.pd_ratio**3,

            + 0.03271438 * self.pd_ratio**0 +
            - 0.05411986 * self.pd_ratio**1 +
            + 0.03452750 * self.pd_ratio**2 +
            - 0.00889198 * self.pd_ratio**3,
        ], dtype=float64)

    @property
    def _kt6_55(self) -> NDArray[float64]:
        return array([
            - 0.0790642 * self.pd_ratio**0 +
            + 0.6869545 * self.pd_ratio**1 +
            - 0.1329584 * self.pd_ratio**2 +
            - 0.0019329 * self.pd_ratio**3,

            - 0.1298732 * self.pd_ratio**0 +
            - 0.4702193 * self.pd_ratio**1 +
            + 0.4109637 * self.pd_ratio**2 +
            - 0.0907269 * self.pd_ratio**3,

            - 0.3184723 * self.pd_ratio**0 +
            + 0.6379156 * self.pd_ratio**1 +
            - 0.6429782 * self.pd_ratio**2 +
            + 0.2211519 * self.pd_ratio**3,

            - 0.2713717 * self.pd_ratio**0 +
            + 0.4058155 * self.pd_ratio**1 +
            - 0.1455594 * self.pd_ratio**2 +
            - 0.0103519 * self.pd_ratio**3,
        ], dtype=float64)
    
    @property
    def _kq6_55(self) -> NDArray[float64]:
        return array([
            - 0.01138489 * self.pd_ratio**0 +
            + 0.05125740 * self.pd_ratio**1 +
            + 0.02834479 * self.pd_ratio**2 +
            - 0.00244448 * self.pd_ratio**3,

            + 0.03597150 * self.pd_ratio**0 +
            - 0.18900802 * self.pd_ratio**1 +
            + 0.17367129 * self.pd_ratio**2 +
            - 0.05419557 * self.pd_ratio**3,

            - 0.00148579 * self.pd_ratio**0 +
            + 0.08660195 * self.pd_ratio**1 +
            - 0.14112461 * self.pd_ratio**2 +
            + 0.05826705 * self.pd_ratio**3,

            - 0.08011267 * self.pd_ratio**0 +
            + 0.09144876 * self.pd_ratio**1 +
            - 0.02563117 * self.pd_ratio**2 +
            - 0.00392796 * self.pd_ratio**3,
        ], dtype=float64)

    @property
    def _kt6_70(self) -> NDArray[float64]:
        return array([
            - 0.0916528 * self.pd_ratio**0 +
            + 0.6896042 * self.pd_ratio**1 +
            - 0.1115190 * self.pd_ratio**2,

            - 0.1835845 * self.pd_ratio**0 +
            - 0.2973475 * self.pd_ratio**1 +
            + 0.1695376 * self.pd_ratio**2,

            - 0.3017883 * self.pd_ratio**0 +
            + 0.3339410 * self.pd_ratio**1 +
            - 0.1404525 * self.pd_ratio**2,

            - 0.0987808 * self.pd_ratio**0 +
            + 0.1158054 * self.pd_ratio**1 +
            - 0.0310880 * self.pd_ratio**2,
        ], dtype=float64)
    
    @property
    def _kq6_70(self) -> NDArray[float64]:
        return array([
            - 0.1217118 * self.pd_ratio**0 +
            + 0.4646745 * self.pd_ratio**1 +
            + 0.3533158 * self.pd_ratio**2,

            + 0.2420133 * self.pd_ratio**0 +
            - 0.9746163 * self.pd_ratio**1 +
            + 0.3077717 * self.pd_ratio**2,

            - 0.4501002 * self.pd_ratio**0 +
            + 0.8654783 * self.pd_ratio**1 +
            - 0.3687229 * self.pd_ratio**2,

            - 0.2523096 * self.pd_ratio**0 +
            + 0.0380946 * self.pd_ratio**1 +
            + 0.0371469 * self.pd_ratio**2,
        ], dtype=float64)