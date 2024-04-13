from abc import ABC, abstractmethod
from dataclasses import dataclass
from numpy import pi, ndarray

@dataclass
class Propeller(ABC):
    diameter: float
    blades: int

    def __post_init__(self):
        if not (self.diameter > 0):
            raise ValueError(f'Diameter (= {self.diameter}) must be > 0')

        if not (isinstance(self.blades, int)):
            raise TypeError(f'The amount of blades (= {self.blades}) must be an integer')


    def eta(self, j):
        kt = self.kt(j)
        kq = self.kq(j)
        res = j * float('NaN')

        eta_valid = kt > 0
        if isinstance(j, ndarray):
            res[eta_valid] = kt[eta_valid] * j[eta_valid] / 2 / pi / kq[eta_valid]
        elif eta_valid:
            res = kt * j / 2 / pi / kq

        return res


    @abstractmethod
    def kt(self, j):
        pass

    @abstractmethod
    def kq(self, j):
        pass