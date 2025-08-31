from math import atan2

from propy.propeller import Propeller
from propy.wageningen_b import WageningenBPropeller

from pytest import raises, approx
from numpy import pi, array, ndarray
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
    thrust = 1393000
    speed = 8.65
    immersion = 3.51

    prop = WageningenBPropeller(
        blades=4,
    ).optimize(
        objective=lambda p: p.losses(speed, thrust),
        constraints=[
            lambda p: p.cavitation_margin(thrust, immersion)
        ],
        diameter_max=7,
    )

    # Immersion is modified to achieve a safety factor for the minimum area_ratio, this is also done in the book by
    # simply "choosing" a higher area_ratio manually (0.55)
    assert prop.cavitation_margin(thrust, immersion) > -1e-15

    assert prop.diameter <= 7
    assert_allclose(prop.pd_ratio, 1.0, rtol=6e-3)

    j = prop.find_j_for_vt(speed, thrust)
    n, q = prop.find_nq_for_vt(speed, thrust)

    assert q == approx(1667435, rel=1e-2)
    assert n == approx(1.767, rel=1e-2)
    assert j == approx(0.699, rel=1e-2)
    assert prop.kt(j) == approx(0.181, rel=1e-2)
    assert prop.kq(j) == approx(0.0310, rel=1e-2)
    assert prop.eta(j) == approx(0.651, rel=1e-2)

    # The results are a bit different compared to [1], this is because [1] just provides an example of a manual
    # optimization. We expect our optimizer to perform a bit better.
    assert prop.eta(j) > 0.651


def test_optimization_min_rotation_speed() -> None:
    """
    This test compares the result of a propeller optimization with the results from [1] chapter 9.4.

        [1] G. Kuiper, The Wageningen propeller series, MARIN Publication 92-001, 1992
    """
    thrust = 1393000
    speed = 8.65

    prop = WageningenBPropeller(
        blades=4
    ).optimize(
        objective=lambda p: p.losses(speed, thrust),
        constraints=[
            lambda p: p.torque_margin(speed, thrust, 1667435)
        ]
    )

    j = prop.find_j_for_vt(speed, thrust)

    assert_allclose(prop.diameter, 7.36, rtol=2e-3)
    assert_allclose(j, 0.665, rtol=11e-3)
    assert_allclose(prop.kt(j), 0.148, rtol=21e-3)
    assert_allclose(prop.kq(j), 0.0239, rtol=27e-3)

    # The results are a bit different compared to [1], this is because [1] just provides an example of a manual
    # optimization. We expect our optimizer to perform a bit better.
    n, q = prop.find_nq_for_vt(speed, thrust)
    assert n < 1.767 * (1 + 1e-10)
    assert q < 1667435 * (1 + 1e-10)
    assert prop.eta(j) > 0.656


def test_torque_limit() -> None:
    thrust = 1000
    speed = 10

    prop = WageningenBPropeller(
        blades=3,
    ).optimize(
        objective=lambda p: p.losses(speed, thrust),
        constraints=[
            lambda p: p.torque_margin(speed, thrust, 60)
        ]
    )

    n, q = prop.find_nq_for_vt(speed, thrust)

    assert prop.torque_margin(speed, thrust, 60) > -5e-8
    assert q < 60 * (1 + 5e-8)


def test_rpm_limit() -> None:
    thrust = 1000
    speed = 10

    prop = WageningenBPropeller(
        blades=3
    ).optimize(
        objective=lambda p: p.losses(speed, thrust),
        constraints=[
            lambda p: p.rotation_speed_margin(speed, thrust, 20)
        ]
    )

    n, q = prop.find_nq_for_vt(speed, thrust)

    assert prop.rotation_speed_margin(speed, thrust, 20) > -1-15
    assert n < 20 * (1 + 1e-15)


def test_diameter_limit() -> None:
    thrust = 1000
    speed = 10

    prop = WageningenBPropeller(
        blades=3,
    ).optimize(
        objective=lambda p: p.losses(speed, thrust),
        diameter_max=0.2
    )

    assert prop.diameter < 0.2*(1+1e-15)


def test_area_ratio_limit() -> None:
    thrust = 1000
    speed = 20
    immersion = 1

    prop = WageningenBPropeller(
        blades=3
    ).optimize(
        objective=lambda p: p.losses(speed, thrust),
        constraints=[
            lambda p: p.cavitation_margin(thrust, immersion)
        ]
    )

    # That's not really close, weird
    assert prop.cavitation_margin(thrust, immersion) > -1e-6


def test_tip_speed_limit() -> None:
    thrust = 1000
    speed = 10

    prop = WageningenBPropeller(
        blades=3
    ).optimize(
        objective=lambda p: p.losses(speed, thrust),
        constraints=[
            lambda p: p.tip_speed_margin(speed, thrust, 24)
        ]
    )

    n, q = prop.find_nq_for_vt(speed, thrust)

    # That's not really close, weird
    assert prop.tip_speed_margin(speed, thrust, 24) > -1e-6
    assert n * pi * prop.diameter < 24 * (1 + 1e-6)


def test_4q_prop() -> None:
    prop = WageningenBPropeller()

    assert prop.ct(0) == approx(8 * prop.kt(0) / pi / (0.7**2 * pi**2))
    assert prop.cq(0) == approx(8 * prop.kq(0) / pi / (0.7**2 * pi**2))

    beta_max = atan2(prop.j_max, 0.7 * pi)

    assert prop.ct(beta_max) == approx(8 * prop.kt_min / pi / (prop.j_max**2 + 0.7**2 * pi**2))
    assert prop.cq(beta_max) == approx(8 * prop.kq_min / pi / (prop.j_max**2 + 0.7**2 * pi**2))


def test_finding_type_consistency() -> None:
    prop = WageningenBPropeller()

    assert isinstance(prop.find_j_for_vn(1, 1), float)
    assert isinstance(prop.find_j_for_vn_vec(array([1]), array([1])), ndarray)
    assert (prop.find_j_for_vn_vec(array([1.23456]), array([6.789]))[0] ==
            approx(prop.find_j_for_vn(1.23456, 6.789)))

    assert isinstance(prop.find_j_for_vt(1, 1), float)
    assert isinstance(prop.find_j_for_vt_vec(array([1]), array([1])), ndarray)
    assert (prop.find_j_for_vt_vec(array([1.23456]), array([6.789]))[0] ==
            approx(prop.find_j_for_vt(1.23456, 6.789)))

    assert isinstance(prop.find_beta_for_vn(1, 1), float)
    assert isinstance(prop.find_beta_for_vn_vec(array([1]), array([1])), ndarray)
    assert (prop.find_beta_for_vn_vec(array([1.23456]), array([6.789]))[0] ==
            approx(prop.find_beta_for_vn(1.23456, 6.789)))

    assert isinstance(prop.ct(1), float)
    assert isinstance(prop.ct(1.0), float)
    assert isinstance(prop.ct(array([1, 2])), ndarray)
    assert prop.ct(array([0.123456]))[0] == approx(prop.ct(0.123456))

    assert isinstance(prop.cq(1), float)
    assert isinstance(prop.cq(1.0), float)
    assert isinstance(prop.cq(array([1, 2])), ndarray)
    assert prop.cq(array([0.123456]))[0] == approx(prop.cq(0.123456))

    assert isinstance(prop.kt(0.1), float)
    assert isinstance(prop.kt(array([0.1, 0.2])), ndarray)
    assert prop.kt(array([0.123456]))[0] == approx(prop.kt(0.123456))

    assert isinstance(prop.kq(0.1), float)
    assert isinstance(prop.kq(array([0.1, 0.2])), ndarray)
    assert prop.kq(array([0.123456]))[0] == approx(prop.kq(0.123456))

    assert isinstance(prop.eta(0.1), float)
    assert isinstance(prop.eta(array([0.1, 0.2])), ndarray)
    assert prop.eta(array([0.123456]))[0] == approx(prop.eta(0.123456))
