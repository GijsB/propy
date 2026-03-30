from propy.wageningen_b import WageningenBPropeller

from scipy.optimize import minimize, Bounds, NonlinearConstraint
from numpy import inf


def test_optimization_max_diameter() -> None:
    thrust = 1393000
    speed = 8.65
    immersion = 3.51

    def f_obj(args):
        print(f'f_obj({args})')
        return WageningenBPropeller(4, *args).losses(speed, thrust)

    def f_con(args):
        print(f'f_con({args})')
        return WageningenBPropeller(4, *args).cavitation_margin(thrust, immersion)

    minimize(
        method='COBYQA',
        fun=f_obj,
        x0=(1.0, 0.5, 0.8),
        bounds=Bounds(
            lb=(0.03, 0.3, 0.5),
            ub=(7.0, 1.05, 1.4),
            keep_feasible=(True, True, True)
        ),
        constraints=[
            NonlinearConstraint(f_con, 0, inf)
        ],
        options={
            'disp': True
        }
    )

    assert True