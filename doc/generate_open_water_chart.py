import numpy as np
import matplotlib.pyplot as plt

from propy import WageningenBPropeller

prop = WageningenBPropeller(
    diameter=0.3,
    blades=2,
    area_ratio=0.9,
    pd_ratio=0.5
)

j = np.linspace(0, prop.j_max)

plt.figure()
plt.plot(j, prop.kt(j), label='kt')
plt.plot(j, prop.kq(j), label='kq')
plt.plot(j, prop.eta(j), label='eta')

plt.xlabel('Advance ratio J')
plt.title(f'Open-water chart')
plt.grid()
plt.legend()

plt.savefig('open_water_chart.png', dpi=300)