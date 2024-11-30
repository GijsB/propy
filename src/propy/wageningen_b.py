from propy.propeller import Propeller

from dataclasses import dataclass
from numpy.polynomial.polynomial import Polynomial
from numpy import roots, isreal


@dataclass
class WageningenBPropeller(Propeller):
    """
    A propeller of the Wageningen-B type. This propeller is defined by 2 polynomials for the thrust- and
    torque-coefficients. The polynomials are defined in [1].

    References
    ----------
        [1] M. M. Bernitsas, D. Ray and P. Kinley: Kt, Kq and efficiency curves for the wageningen b-series propellers,
        Department of Naval Architecture and Marine Engineering, University of Michigan. May 1981.

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

    def __post_init__(self):
        super().__post_init__()

        self.kt = self._calc_kt_pol()
        self.kq = self._calc_kq_pol()
        kt_root = roots(self.kt.coef[::-1])
        kt_root = kt_root[(kt_root > 0) & (kt_root < 1.6)]
        kt_root = min(kt_root)
        assert isreal(kt_root)
        self._j_max = kt_root
        self.kt = self.kt.convert(domain=[0, self.j_max], window=[0, self.j_max])
        self.kq = self.kq.convert(domain=[0, self.j_max], window=[0, self.j_max])

    def _calc_kq_pol(self):
        return Polynomial(symbol='J', coef=[
              0.0037936800 * self.pd_ratio**0 * self.area_ratio**0 * self.blades**0 +
              0.0158960000 * self.pd_ratio**0 * self.area_ratio**2 * self.blades**0 +
            - 0.0001843000 * self.pd_ratio**0 * self.area_ratio**2 * self.blades**2 +
              0.0051369600 * self.pd_ratio**1 * self.area_ratio**0 * self.blades**1 +
            - 0.0408811000 * self.pd_ratio**1 * self.area_ratio**1 * self.blades**0 +
            - 0.0502782000 * self.pd_ratio**1 * self.area_ratio**2 * self.blades**0 +
              0.0034477800 * self.pd_ratio**2 * self.area_ratio**0 * self.blades**0 +
              0.1885610000 * self.pd_ratio**2 * self.area_ratio**1 * self.blades**0 +
            - 0.0269403000 * self.pd_ratio**2 * self.area_ratio**1 * self.blades**1 +
              0.0015533400 * self.pd_ratio**2 * self.area_ratio**1 * self.blades**2 +
              0.0126803000 * self.pd_ratio**2 * self.area_ratio**2 * self.blades**1 +
              0.0161886000 * self.pd_ratio**3 * self.area_ratio**1 * self.blades**0 +
            - 0.0397722000 * self.pd_ratio**3 * self.area_ratio**2 * self.blades**0 +
            - 0.0004253990 * self.pd_ratio**3 * self.area_ratio**2 * self.blades**2 +
            - 0.0003139120 * self.pd_ratio**6 * self.area_ratio**0 * self.blades**1 +
            - 0.0014212100 * self.pd_ratio**6 * self.area_ratio**1 * self.blades**1 +
              0.0003026830 * self.pd_ratio**6 * self.area_ratio**1 * self.blades**2 +
            - 0.0035002400 * self.pd_ratio**6 * self.area_ratio**2 * self.blades**0 +
              0.0033426800 * self.pd_ratio**6 * self.area_ratio**2 * self.blades**1 +
            - 0.0004659000 * self.pd_ratio**6 * self.area_ratio**2 * self.blades**2,
            - 0.0037087100 * self.pd_ratio**0 * self.area_ratio**0 * self.blades**1 +
              0.0002695510 * self.pd_ratio**0 * self.area_ratio**1 * self.blades**2 +
              0.0471729000 * self.pd_ratio**0 * self.area_ratio**2 * self.blades**0 +
            - 0.0038363700 * self.pd_ratio**0 * self.area_ratio**2 * self.blades**1 +
            - 0.0322410000 * self.pd_ratio**1 * self.area_ratio**0 * self.blades**0 +
              0.0209449000 * self.pd_ratio**1 * self.area_ratio**0 * self.blades**1 +
            - 0.0018349100 * self.pd_ratio**1 * self.area_ratio**0 * self.blades**2 +
            - 0.1080090000 * self.pd_ratio**1 * self.area_ratio**1 * self.blades**0 +
              0.0043838800 * self.pd_ratio**1 * self.area_ratio**1 * self.blades**1 +
              0.0031809860 * self.pd_ratio**3 * self.area_ratio**1 * self.blades**0 +
              0.0000554194 * self.pd_ratio**6 * self.area_ratio**2 * self.blades**2,
              0.0088652300 * self.pd_ratio**0 * self.area_ratio**0 * self.blades**0 +
            - 0.0072340800 * self.pd_ratio**0 * self.area_ratio**1 * self.blades**1 +
              0.0008326500 * self.pd_ratio**0 * self.area_ratio**1 * self.blades**2 +
              0.0047431900 * self.pd_ratio**1 * self.area_ratio**0 * self.blades**1 +
            - 0.0885381000 * self.pd_ratio**1 * self.area_ratio**1 * self.blades**0 +
              0.0417122000 * self.pd_ratio**2 * self.area_ratio**2 * self.blades**0 +
            - 0.0031827800 * self.pd_ratio**3 * self.area_ratio**2 * self.blades**1,
            - 0.0106854000 * self.pd_ratio**0 * self.area_ratio**0 * self.blades**1 +
              0.0558082000 * self.pd_ratio**0 * self.area_ratio**1 * self.blades**0 +
              0.0035985000 * self.pd_ratio**0 * self.area_ratio**1 * self.blades**1 +
              0.0196283000 * self.pd_ratio**0 * self.area_ratio**2 * self.blades**0 +
            - 0.0300550000 * self.pd_ratio**1 * self.area_ratio**2 * self.blades**0 +
              0.0001124510 * self.pd_ratio**2 * self.area_ratio**0 * self.blades**2 +
              0.0011090300 * self.pd_ratio**3 * self.area_ratio**0 * self.blades**1 +
              0.0000869243 * self.pd_ratio**3 * self.area_ratio**2 * self.blades**2 +
            - 0.0000297228 * self.pd_ratio**6 * self.area_ratio**0 * self.blades**2
        ])


    def _calc_kt_pol(self):
        return Polynomial(symbol='J', coef=[
              0.008804960 * self.pd_ratio**0 * self.area_ratio**0 * self.blades**0 +
              0.014404300 * self.pd_ratio**0 * self.area_ratio**0 * self.blades**1 +
            - 0.000606848 * self.pd_ratio**0 * self.area_ratio**0 * self.blades**2 +
            - 0.012589400 * self.pd_ratio**0 * self.area_ratio**1 * self.blades**1 +
              0.000690904 * self.pd_ratio**0 * self.area_ratio**1 * self.blades**2 +
            - 0.050721400 * self.pd_ratio**0 * self.area_ratio**2 * self.blades**0 +
              0.166351000 * self.pd_ratio**1 * self.area_ratio**0 * self.blades**0 +
              0.014348100 * self.pd_ratio**1 * self.area_ratio**0 * self.blades**1 +
              0.158114000 * self.pd_ratio**2 * self.area_ratio**0 * self.blades**0 +
              0.415437000 * self.pd_ratio**2 * self.area_ratio**1 * self.blades**0 +
            - 0.004107980 * self.pd_ratio**2 * self.area_ratio**2 * self.blades**1 +
            - 0.133698000 * self.pd_ratio**3 * self.area_ratio**0 * self.blades**0 +
            - 0.008417280 * self.pd_ratio**3 * self.area_ratio**0 * self.blades**1 +
            - 0.031779100 * self.pd_ratio**3 * self.area_ratio**1 * self.blades**1 +
              0.004217490 * self.pd_ratio**3 * self.area_ratio**1 * self.blades**2 +
            - 0.001465640 * self.pd_ratio**3 * self.area_ratio**2 * self.blades**2 +
              0.006384070 * self.pd_ratio**6 * self.area_ratio**0 * self.blades**0,
            - 0.204554000 * self.pd_ratio**0 * self.area_ratio**0 * self.blades**0 +
            - 0.004981900 * self.pd_ratio**0 * self.area_ratio**0 * self.blades**2 +
              0.010968900 * self.pd_ratio**0 * self.area_ratio**1 * self.blades**1 +
              0.018604000 * self.pd_ratio**0 * self.area_ratio**2 * self.blades**1 +
              0.060682600 * self.pd_ratio**1 * self.area_ratio**0 * self.blades**1 +
            - 0.481497000 * self.pd_ratio**1 * self.area_ratio**1 * self.blades**0 +
            - 0.001636520 * self.pd_ratio**2 * self.area_ratio**0 * self.blades**2 +
              0.016842400 * self.pd_ratio**3 * self.area_ratio**0 * self.blades**1 +
            - 0.000328787 * self.pd_ratio**6 * self.area_ratio**0 * self.blades**2 +
              0.010465000 * self.pd_ratio**6 * self.area_ratio**2 * self.blades**0,
            - 0.053005400 * self.pd_ratio**0 * self.area_ratio**0 * self.blades**1 +
              0.002598300 * self.pd_ratio**0 * self.area_ratio**0 * self.blades**2 +
            - 0.147581000 * self.pd_ratio**0 * self.area_ratio**1 * self.blades**0 +
              0.085455900 * self.pd_ratio**0 * self.area_ratio**2 * self.blades**0 +
            - 0.001327180 * self.pd_ratio**6 * self.area_ratio**0 * self.blades**0 +
              0.000116502 * self.pd_ratio**6 * self.area_ratio**0 * self.blades**2 +
            - 0.006482720 * self.pd_ratio**6 * self.area_ratio**2 * self.blades**0,
            - 0.000560528 * self.pd_ratio**0 * self.area_ratio**0 * self.blades**2 +
              0.168496000 * self.pd_ratio**0 * self.area_ratio**1 * self.blades**0 +
            - 0.050447500 * self.pd_ratio**0 * self.area_ratio**2 * self.blades**0 +
            - 0.001022960 * self.pd_ratio**3 * self.area_ratio**0 * self.blades**1 +
              0.0000565229* self.pd_ratio**6 * self.area_ratio**1 * self.blades**2
        ])

    @property
    def blades_min(self):
        return 2

    @property
    def blades_max(self):
        return 7

    @property
    def area_ratio_min(self):
        return 0.3

    @property
    def area_ratio_max(self):
        return 1.05

    @property
    def pd_ratio_min(self):
        return 0.5

    @property
    def pd_ratio_max(self):
        return 1.4

    @property
    def j_max(self):
        return self._j_max

    def kt(self, j):
        pass

    def kq(self, j):
        pass

    def _find_j_for_ktj2(self, ktj2):
        # Define a new polynomial: kt(j) - kt/j^2 * j^2
        p = self.kt.coef.copy()
        p[2] -= ktj2

        # Find the root of this polynomial between 0 < j < j_max
        r = roots(p[::-1])
        r = r[(0 < r) & (r < self.j_max)]

        # At this point, there should be exactly 1 real root
        assert len(r) == 1
        assert isreal(r[0])
        return r[0]
