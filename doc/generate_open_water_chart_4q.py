import numpy as np
import matplotlib.pyplot as plt

from propy import WageningenBPropeller


if __name__ == "__main__":
    prop = WageningenBPropeller(
        diameter=0.3,
        blades=2,
        area_ratio=0.9,
        pd_ratio=0.5
    )

    j = np.linspace(0, prop.j_max)
    ct_1q = 8 * prop.kt(j) / np.pi / (j**2 + 0.7**2 * np.pi**2)
    cq_1q = 8 * prop.kq(j) / np.pi / (j ** 2 + 0.7 ** 2 * np.pi ** 2)

    beta = np.linspace(0, 2 * np.pi)
    ct = prop.ct(beta)
    cq = prop.cq(beta)

    plt.figure()
    plt.polar(beta, ct, label='Ct (4q)')
    plt.polar(np.atan(j / 0.7 / np.pi), ct_1q, label="Ct (1q)")
    plt.polar(beta, cq * 10, label='10 * Cq (4q)')
    plt.polar(np.atan(j / 0.7 / np.pi), cq_1q * 10, label="10 * Cq (1q)")

    plt.gca().set_rorigin(-1)
    plt.gca().set_rmin(-0.5)
    plt.gca().set_rmax(0.5)
    plt.xlabel('Advance angle')
    plt.legend()
    plt.title('Open water chart, 4-quadrants')

    plt.savefig('open_water_chart_4q.png', dpi=300)
