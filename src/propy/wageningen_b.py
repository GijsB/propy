from propy.propeller import Propeller

from dataclasses import dataclass
from numpy.polynomial.polynomial import Polynomial



@dataclass
class WageningenBPropeller(Propeller):
    area_ratio: float
    pd_ratio: float

    def __post_init__(self):
        super().__post_init__()

        if not (self.blades >= 2):
            raise ValueError(f'Amount of blades (= {self.blades}) must be >= 2')

        if not (self.blades <= 7):
            raise ValueError(f'Amount of blades (= {self.blades}) must be <= 7')

        if not (self.area_ratio >= 0.3):
            raise ValueError(f'Area ratio (= {self.area_ratio}) must be >= 0.3')

        if not (self.area_ratio <= 1.05):
            raise ValueError(f'Area ratio (= {self.area_ratio}) must be <= 1.05')

        if not (self.pd_ratio >= 0.5):
            raise ValueError(f'Pitch/Diameter ratio (= {self.pd_ratio}) must be >= 0.5')

        if not (self.pd_ratio <= 1.4):
            raise ValueError(f'Pitch/Diameter ratio (= {self.pd_ratio}) must be <= 1.4')

        self.kt = self._calc_kt_pol()
        self.kq = self._calc_kq_pol()


    def _calc_kq_pol(self):
        return Polynomial(coef=[
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
        return Polynomial(coef=[
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
        ], symbol='J')

    def kt(self, j):
        pass

    def kq(self, j):
        pass


