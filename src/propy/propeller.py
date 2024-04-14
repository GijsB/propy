from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from numpy import pi, ndarray, array
from scipy.optimize import root_scalar, minimize, minimize_scalar


@dataclass
class Propeller(ABC):
    blades:     int = 4
    diameter:   float = 1.0
    area_ratio: float = 0.5
    pd_ratio:   float = 0.8

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

    def eta(self, j):
        kt = self.kt(j)
        kq = self.kq(j)
        res = j * float('NaN')

        eta_valid = j < self.j_max
        if isinstance(j, ndarray):
            res[eta_valid] = kt[eta_valid] * j[eta_valid] / 2 / pi / kq[eta_valid]
        elif eta_valid:
            res = kt * j / 2 / pi / kq

        return res

    def find_j_for_ktj2(self, ktj2):
        return root_scalar(
            f=lambda j: self.kt(j)/j**2 - ktj2,
            bracket=[1e-9, self.j_max],
            x0=0.8*self.j_max
        ).root

    def find_j_for_ktj4(self, ktj4):
        return root_scalar(
            f=lambda j: self.kt(j)/j**4 - ktj4,
            bracket=[1e-9, self.j_max],
            x0=0.8*self.j_max
        ).root

    def minimum_area_ratio(self, thrust, immersion, single_screw=False, rho=1025):
        min_area_ratio = (1.3 + 0.3 * self.blades) * thrust / self.diameter**2 / (1e5 + rho * 9.81 * immersion - 1700)
        if single_screw:
            min_area_ratio += 0.2
        return min_area_ratio

    def optimize_for_diameter(self, thrust, immersion, velocity, single_screw=False, rho=1025):
        min_area_ratio = self.minimum_area_ratio(thrust, immersion, single_screw=single_screw, rho=rho)
        min_area_ratio = max(min_area_ratio, 0.3)
        ktj2 = thrust / rho / velocity ** 2 / self.diameter ** 2

        def losses(x):
            p = replace(self, area_ratio=x[0], pd_ratio=x[1])
            return 1 - p.eta(p.find_j_for_ktj2(ktj2))

        opt_res = minimize(
            fun=losses,
            x0=array([self.area_ratio, self.pd_ratio]),
            bounds=[(min_area_ratio, 1.05), (0.5, 1.4)]
        )

        prop = replace(self, area_ratio=opt_res.x[0], pd_ratio=opt_res.x[1])

        return prop

    def optimize_for_rotation_rate(self, thrust, n, velocity, immersion, single_screw=False, rho=1025):
        ktj4 = thrust * n**2 / rho / velocity**4

        def losses(x):
            p = replace(self, area_ratio=x[0], pd_ratio=x[1])
            return 1 - p.eta(p.find_j_for_ktj4(ktj4))

        def cavitation_margin(x):
            p = replace(self, area_ratio=x[0], pd_ratio=x[1])
            d = velocity / n / p.find_j_for_ktj4(ktj4)
            p = replace(p, diameter=d)
            min_ear = p.minimum_area_ratio(thrust, immersion, single_screw=single_screw, rho=rho)
            return x[0] - min_ear

        def immersion_margin(x):
            p = replace(self, area_ratio=x[0], pd_ratio=x[1])
            d = velocity / n / p.find_j_for_ktj4(ktj4)
            return immersion - d/2

        # noinspection PyTypeChecker
        opt_res = minimize(
            fun=losses,
            x0=array([self.area_ratio, self.pd_ratio]),
            bounds=[(self.area_ratio_min, self.area_ratio_max),
                    (self.pd_ratio_min, self.pd_ratio_max)],
            constraints=[
                {'type': 'ineq', 'fun': cavitation_margin},
                {'type': 'ineq', 'fun': immersion_margin}
            ]
        )

        prop = replace(self, area_ratio=opt_res.x[0], pd_ratio=opt_res.x[1])
        diameter = velocity / n / prop.find_j_for_ktj4(ktj4)
        prop = replace(prop, diameter=diameter)

        return prop

    @property
    @abstractmethod
    def j_max(self):
        pass

    @property
    @abstractmethod
    def blades_min(self):
        pass

    @property
    @abstractmethod
    def blades_max(self):
        pass

    @property
    @abstractmethod
    def area_ratio_min(self):
        pass

    @property
    @abstractmethod
    def area_ratio_max(self):
        pass

    @property
    @abstractmethod
    def pd_ratio_min(self):
        pass

    @property
    @abstractmethod
    def pd_ratio_max(self):
        pass

    @abstractmethod
    def kt(self, j):
        pass

    @abstractmethod
    def kq(self, j):
        pass