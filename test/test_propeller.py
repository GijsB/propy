from propy.propeller import Propeller, WorkingPoint
from propy.wageningen_b import WageningenBPropeller

from pytest import raises
from numpy import pi
from numpy.testing import assert_allclose

def test_instantiation():
    """Check whether instantiation of an abstract Propeller raises a TypeError"""
    with raises(TypeError):
        Propeller()

def test_new():
    """Check whether calling new (on ABC) raises a TypeError"""
    with raises(TypeError):
        Propeller.new()


def test_optimization_max_diameter():
    """
    This test compares the result of a propeller optimization with the results from [1] champter 9.3.

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
        objective = lambda p: p.losses(wp),
        constraints = [
            lambda p: p.cavitation_margin(wp)
        ],
        diameter_max=7,
    )

    # Immersion is modified to achieve a safety factor for the minimum area_ratio, this is also done in the book by
    # simply "chosing" a higher area_ratio manually (0.55)
    assert prop.cavitation_margin(wp) > -1e-15

    assert prop.diameter <= 7
    assert_allclose(prop.pd_ratio, 1.0, rtol=6e-3)

    pp = prop.find_performance(wp)

    assert_allclose([pp.torque, pp.rotation_speed, pp.j, pp.kt, pp.kq, pp.eta],
                    [1667435, 1.767, 0.699, 0.181, 0.0310, 0.651], rtol=1e-2)

    # The results are a bit different compared to [1], this is because [1] just provides an example of a manual
    # optimization. We expect our optimizer to perform a bit better.
    assert pp.eta > 0.651


def test_optimization_min_rotation_speed():
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
        objective=lambda p: p.losses(wp),
        constraints=[
            lambda p: p.torque_margin(wp, 1667435)
        ]
    )

    pp = prop.find_performance(wp)

    assert_allclose(prop.diameter, 7.36, rtol=2e-3)
    assert_allclose(pp.j, 0.665, rtol=11e-3)
    assert_allclose(pp.kt, 0.148, rtol=21e-3)
    assert_allclose(pp.kq, 0.0239, rtol=27e-3)

    # The results are a bit different compared to [1], this is because [1] just provides an example of a manual
    # optimization. We expect our optimizer to perform a bit better.
    assert pp.rotation_speed < 1.767 * (1 + 1e-10)
    assert pp.torque < 1667435 * (1 + 1e-10)
    assert pp.eta > 0.656


def test_torque_limit():
    wp = WorkingPoint(
        thrust=1000,
        speed=10,
    )

    prop = WageningenBPropeller(
        blades=3,
    ).optimize(
        objective = lambda p: p.losses(wp),
        constraints = [
            lambda p: p.torque_margin(wp, 60)
        ]
    )

    pp  = prop.find_performance(wp)

    assert prop.torque_margin(wp, 60) > -5e-8
    assert pp.torque < 60 * (1 + 5e-8)


def test_rpm_limit():
    wp = WorkingPoint(
        thrust= 1000,
        speed= 10,
    )

    prop = WageningenBPropeller(
        blades = 3
    ).optimize(
        objective = lambda p: p.losses(wp),
        constraints = [
            lambda p: p.rotation_speed_margin(wp, 20)
        ]
    )

    pp = prop.find_performance(wp)

    assert prop.rotation_speed_margin(wp, 20) > -1-15
    assert pp.rotation_speed < 20 * (1 + 1e-15)


def test_diameter_limit():
    wp = WorkingPoint(
        thrust=1000,
        speed=10,
    )

    prop = WageningenBPropeller(
        blades=3,
    ).optimize(
        objective=lambda p: p.losses(wp),
        diameter_max=0.2
    )

    assert prop.diameter < 0.2*(1+1e-15)


def test_area_ratio_limit():
    wp = WorkingPoint(
        thrust=1000,
        speed=20,
        immersion=1
    )

    prop = WageningenBPropeller(
        blades=3
    ).optimize(
        objective=lambda p: p.losses(wp),
        constraints=[
            lambda p: p.cavitation_margin(wp)
        ]
    )

    # That's not really close.. weird
    assert prop.cavitation_margin(wp) > -1e-6


def test_tip_speed_limit():
    wp = WorkingPoint(
        thrust=1000,
        speed=10,
    )

    prop = WageningenBPropeller(
        blades=3
    ).optimize(
        objective=lambda p: p.losses(wp),
        constraints=[
            lambda p: p.tip_speed_margin(wp, 24)
        ]
    )

    pp = prop.find_performance(wp)

    # That's not really close.. weird
    assert prop.tip_speed_margin(wp, 24) > -1e-6
    assert pp.rotation_speed * pi * prop.diameter < 24 * (1 + 1e-6)