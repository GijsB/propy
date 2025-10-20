from propy.propeller import Propeller, ScalarOrArray

from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar, Callable, cast
from math import sqrt

from numpy.polynomial.polynomial import Polynomial
from numpy import roots, isreal


@dataclass(frozen=True)
class GawnBurrillPropeller(Propeller):
    """

    References
    ----------
        [1] D. Radojcic, Mathematical Model of Segmental Section Propeller Series for Open-Water and Cavitating
        Conditions Applicable in CAD.
    """
    blades: int = 3

    blades_min: ClassVar[int] = 3
    blades_max: ClassVar[int] = 3
    area_ratio_min: ClassVar[float] = 0.34 * 0.5 * (2.75 + 0.5 / 3)
    area_ratio_max: ClassVar[float] = 0.34 * 1.1 * (2.75 + 1.1 / 3)
    pd_ratio_min: ClassVar[float] = 0.8
    pd_ratio_max: ClassVar[float] = 1.8

    @cached_property
    def j_max(self) -> float:
        # Cast to a Polynomial object because we know this to be true for a GawnBurrill propeller
        kt = cast(Polynomial, self.kt)
        kt_root = roots(kt.coef[::-1])
        kt_root = kt_root[(kt_root > 0) & (kt_root < 2.0)]
        kt_root = min(kt_root)
        assert isreal(kt_root)
        return float(kt_root.real)

    @property
    def j_min(self) -> float:
        return self.area_ratio / 2

    @property
    def kt(self) -> Callable[[ScalarOrArray], ScalarOrArray]:
        area_ratio = (sqrt(0.935**2 + 4 * 0.113 * self.area_ratio) - 0.935) / 2 / 0.1133
        return cast(
            Callable[[ScalarOrArray], ScalarOrArray],
            Polynomial(symbol='J', coef=[
                + 0.1193852 * 10 ** +0 * area_ratio ** 0 * self.pd_ratio ** 0 +
                + 0.3493294 * 10 ** +0 * area_ratio ** 0 * self.pd_ratio ** 1 +
                - 0.1341679 * 10 ** +0 * area_ratio ** 1 * self.pd_ratio ** 0 +
                + 0.2970728 * 10 ** +0 * area_ratio ** 1 * self.pd_ratio ** 2 +
                - 4.0801660 * 10 ** -3 * area_ratio ** 1 * self.pd_ratio ** 6 +
                - 1.1364520 * 10 ** -3 * area_ratio ** 2 * self.pd_ratio ** 6,

                - 0.6574682 * 10 ** +0 * area_ratio ** 0 * self.pd_ratio ** 0 +
                + 0.4119366 * 10 ** +0 * area_ratio ** 0 * self.pd_ratio ** 1 +
                - 0.1991927 * 10 ** +0 * area_ratio ** 0 * self.pd_ratio ** 2 +
                + 0.2628839 * 10 ** +0 * area_ratio ** 1 * self.pd_ratio ** 0 +
                - 0.5217023 * 10 ** +0 * area_ratio ** 1 * self.pd_ratio ** 1 +
                + 4.1542010 * 10 ** -3 * area_ratio ** 1 * self.pd_ratio ** 6,

                + 5.8630510 * 10 ** -2 * area_ratio ** 0 * self.pd_ratio ** 2,

                - 1.1077350 * 10 ** -2 * area_ratio ** 0 * self.pd_ratio ** 2 +
                + 6.1525800 * 10 ** -2 * area_ratio ** 2 * self.pd_ratio ** 1 +
                - 2.4708400 * 10 ** -2 * area_ratio ** 2 * self.pd_ratio ** 2
            ])
        )

    @property
    def kq(self) -> Callable[[ScalarOrArray], ScalarOrArray]:
        area_ratio = (sqrt(0.935**2 + 4 * 0.113 * self.area_ratio) - 0.935) / 2 / 0.1133
        return cast(
            Callable[[ScalarOrArray], ScalarOrArray],
            Polynomial(symbol='J', coef=[
                + 1.5411660 * 10 ** -3 * area_ratio ** 0 * self.pd_ratio ** 0 +
                - 4.3706150 * 10 ** -2 * area_ratio ** 0 * self.pd_ratio ** 1 +
                + 8.5367470 * 10 ** -2 * area_ratio ** 0 * self.pd_ratio ** 2 +
                - 4.8650630 * 10 ** -2 * area_ratio ** 1 * self.pd_ratio ** 1 +
                + 8.5299550 * 10 ** -2 * area_ratio ** 1 * self.pd_ratio ** 2,

                + 0.1091688 * 10 ** +0 * area_ratio ** 0 * self.pd_ratio ** 0 +
                - 9.5121630 * 10 ** -2 * area_ratio ** 0 * self.pd_ratio ** 2 +
                + 5.4960340 * 10 ** -2 * area_ratio ** 1 * self.pd_ratio ** 0 +
                - 0.1062500 * 10 ** +0 * area_ratio ** 1 * self.pd_ratio ** 1,

                - 0.3102420 * 10 ** +0 * area_ratio ** 0 * self.pd_ratio ** 0 +
                + 0.2490295 * 10 ** +0 * area_ratio ** 0 * self.pd_ratio ** 1 +
                - 9.3203070 * 10 ** -3 * area_ratio ** 0 * self.pd_ratio ** 2 +
                - 3.1517560 * 10 ** -3 * area_ratio ** 2 * self.pd_ratio ** 2,

                + 0.1547428 * 10 ** +0 * area_ratio ** 0 * self.pd_ratio ** 0 +
                - 0.1594602 * 10 ** +0 * area_ratio ** 0 * self.pd_ratio ** 1 +
                + 3.2878050 * 10 ** -2 * area_ratio ** 0 * self.pd_ratio ** 2 +
                + 1.1010230 * 10 ** -2 * area_ratio ** 2 * self.pd_ratio ** 0,
            ])
        )
