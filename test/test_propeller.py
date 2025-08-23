from math import atan2

import numpy as np

from propy.propeller import Propeller, WorkingPoint, WorkingPoint4Q
from propy.wageningen_b import WageningenBPropeller

from pytest import raises, approx
from numpy import pi, linspace
from numpy.testing import assert_allclose


def test_instantiation() -> None:
    """Check whether instantiation of an abstract Propeller raises a TypeError"""
    with raises(TypeError):
        # noinspection PyAbstractClass
        Propeller()  # type: ignore


def test_new() -> None:
    """Check whether calling new (on ABC) raises a TypeError"""
    with raises(TypeError):
        Propeller.new()


def test_optimization_max_diameter() -> None:
    """
    This test compares the result of a propeller optimization with the results from [1] chapter 9.3.

        [1] G. Kuiper, The Wageningen propeller series, MARIN Publication 92-001, 1992
    """
    wp = WorkingPoint(
        thrust=1393000,
        speed=8.65,
        immersion=3.51,
    )

    prop = WageningenBPropeller(
        blades=4,
    ).optimize(
        objective=lambda p: p.losses(wp.speed, wp.thrust, rho=wp.rho),
        constraints=[
            lambda p: p.cavitation_margin(wp.thrust, wp.immersion, rho=wp.rho)
        ],
        diameter_max=7,
    )

    # Immersion is modified to achieve a safety factor for the minimum area_ratio, this is also done in the book by
    # simply "choosing" a higher area_ratio manually (0.55)
    assert prop.cavitation_margin(wp.thrust, wp.immersion, rho=wp.rho) > -1e-15

    assert prop.diameter <= 7
    assert_allclose(prop.pd_ratio, 1.0, rtol=6e-3)

    pp = prop.find_performance(wp.speed, wp.thrust, wp.rho)

    assert pp.torque[0] == approx(1667435, rel=1e-2)
    assert pp.rotation_speed[0] == approx(1.767, rel=1e-2)
    assert pp.j[0] == approx(0.699, rel=1e-2)
    assert pp.kt[0] == approx(0.181, rel=1e-2)
    assert pp.kq[0] == approx(0.0310, rel=1e-2)
    assert pp.eta[0] == approx(0.651, rel=1e-2)

    # The results are a bit different compared to [1], this is because [1] just provides an example of a manual
    # optimization. We expect our optimizer to perform a bit better.
    assert pp.eta > 0.651


def test_optimization_min_rotation_speed() -> None:
    """
    This test compares the result of a propeller optimization with the results from [1] chapter 9.4.

        [1] G. Kuiper, The Wageningen propeller series, MARIN Publication 92-001, 1992
    """
    wp = WorkingPoint(
        thrust=1393000,
        speed=8.65
    )

    prop = WageningenBPropeller(
        blades=4
    ).optimize(
        objective=lambda p: p.losses(wp.speed, wp.thrust, rho=wp.rho),
        constraints=[
            lambda p: p.torque_margin(wp, 1667435)
        ]
    )

    pp = prop.find_performance(wp.speed, wp.thrust, wp.rho)

    assert_allclose(prop.diameter, 7.36, rtol=2e-3)
    assert_allclose(pp.j, 0.665, rtol=11e-3)
    assert_allclose(pp.kt, 0.148, rtol=21e-3)
    assert_allclose(pp.kq, 0.0239, rtol=27e-3)

    # The results are a bit different compared to [1], this is because [1] just provides an example of a manual
    # optimization. We expect our optimizer to perform a bit better.
    assert pp.rotation_speed < 1.767 * (1 + 1e-10)
    assert pp.torque < 1667435 * (1 + 1e-10)
    assert pp.eta > 0.656


def test_torque_limit() -> None:
    wp = WorkingPoint(
        thrust=1000,
        speed=10,
    )

    prop = WageningenBPropeller(
        blades=3,
    ).optimize(
        objective=lambda p: p.losses(wp.speed, wp.thrust, rho=wp.rho),
        constraints=[
            lambda p: p.torque_margin(wp, 60)
        ]
    )

    pp = prop.find_performance(wp.speed, wp.thrust, wp.rho)

    assert prop.torque_margin(wp, 60) > -5e-8
    assert pp.torque < 60 * (1 + 5e-8)


def test_rpm_limit() -> None:
    wp = WorkingPoint(
        thrust=1000,
        speed=10,
    )

    prop = WageningenBPropeller(
        blades=3
    ).optimize(
        objective=lambda p: p.losses(wp.speed, wp.thrust, rho=wp.rho),
        constraints=[
            lambda p: p.rotation_speed_margin(wp, 20)
        ]
    )

    pp = prop.find_performance(wp.speed, wp.thrust, wp.rho)

    assert prop.rotation_speed_margin(wp, 20) > -1-15
    assert pp.rotation_speed < 20 * (1 + 1e-15)


def test_diameter_limit() -> None:
    wp = WorkingPoint(
        thrust=1000,
        speed=10,
    )

    prop = WageningenBPropeller(
        blades=3,
    ).optimize(
        objective=lambda p: p.losses(wp.speed, wp.thrust, rho=wp.rho),
        diameter_max=0.2
    )

    assert prop.diameter < 0.2*(1+1e-15)


def test_area_ratio_limit() -> None:
    wp = WorkingPoint(
        thrust=1000,
        speed=20,
        immersion=1
    )

    prop = WageningenBPropeller(
        blades=3
    ).optimize(
        objective=lambda p: p.losses(wp.speed, wp.thrust, rho=wp.rho),
        constraints=[
            lambda p: p.cavitation_margin(wp.thrust, wp.immersion, rho=wp.rho)
        ]
    )

    # That's not really close, weird
    assert prop.cavitation_margin(wp.thrust, wp.immersion, rho=wp.rho) > -1e-6


def test_tip_speed_limit() -> None:
    wp = WorkingPoint(
        thrust=1000,
        speed=10,
    )

    prop = WageningenBPropeller(
        blades=3
    ).optimize(
        objective=lambda p: p.losses(wp.speed, wp.thrust, rho=wp.rho),
        constraints=[
            lambda p: p.tip_speed_margin(wp, 24)
        ]
    )

    pp = prop.find_performance(wp.speed, wp.thrust, wp.rho)

    # That's not really close, weird
    assert prop.tip_speed_margin(wp, 24) > -1e-6
    assert pp.rotation_speed * pi * prop.diameter < 24 * (1 + 1e-6)


def test_4q_prop() -> None:
    prop = WageningenBPropeller()

    assert prop.ct(0) == approx(8 * prop.kt(0) / pi / (0.7**2 * pi**2))
    assert prop.cq(0) == approx(8 * prop.kq(0) / pi / (0.7**2 * pi**2))

    beta_max = atan2(prop.j_max, 0.7 * pi)

    assert prop.ct(beta_max) == approx(8 * prop.kt_min / pi / (prop.j_max**2 + 0.7**2 * pi**2))
    assert prop.cq(beta_max) == approx(8 * prop.kq_min / pi / (prop.j_max**2 + 0.7**2 * pi**2))


def test_4q_1q_compare_performance() -> None:
    prop = WageningenBPropeller(blades=4, area_ratio=0.7, pd_ratio=1.4)

    wp1q = WorkingPoint(
        thrust=1000,
        speed=linspace(1, 10, 1000),
    )
    pp1q = prop.find_performance(wp1q.speed, wp1q.thrust, wp1q.rho)

    wp4q = WorkingPoint4Q(
        rotation_speed=pp1q.rotation_speed,
        speed=wp1q.speed,
        rho=wp1q.rho,
    )
    pp4q = prop.find_performance_4q(wp4q)

    assert np.allclose(pp1q.torque, pp4q.torque)
    assert np.allclose(wp1q.thrust, pp4q.thrust)
    assert np.allclose(pp1q.rotation_speed, wp4q.rotation_speed)


def test_4q_performance_robustness() -> None:
    prop = WageningenBPropeller()

    wp = WorkingPoint4Q(
        rotation_speed=[0, 0, 0, 1, 1, 1, -1, -1, -1],
        speed=[0, 1, -1, 0, 1, -1, 0, 1, -1],
    )
    pp = prop.find_performance_4q(wp)
    assert pp is not None

    rotation_speed = 1
    speed = 1

    wp = WorkingPoint4Q(
        rotation_speed=rotation_speed,
        speed=speed,
    )

    assert prop.find_performance_4q(wp) == prop.find_performance_4q(
        WorkingPoint4Q(
            rotation_speed=[rotation_speed],
            speed=speed,
        ))

    assert prop.find_performance_4q(wp) == prop.find_performance_4q(
        WorkingPoint4Q(
            rotation_speed=[rotation_speed],
            speed=[speed],
        ))

    assert prop.find_performance_4q(wp) == prop.find_performance_4q(
        WorkingPoint4Q(
            rotation_speed=rotation_speed,
            speed=[speed],
        ))
