from typing import Type

from pytest import approx, mark
from numpy.testing import assert_allclose
from numpy import pi

from propy.wageningen_b import WageningenBPropeller
from propy.gawn_burrill import GawnBurrillPropeller
from propy.propeller import Propeller
from propy.optimization import slsqp, trust_constrained, OptimizationMethod


optimization_methods = (slsqp, trust_constrained, )
propeller_types = (WageningenBPropeller, GawnBurrillPropeller)


@mark.parametrize('method', optimization_methods)
def test_optimization_max_diameter(method: OptimizationMethod) -> None:
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
        method=method,
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


@mark.parametrize('method', optimization_methods)
def test_optimization_min_rotation_speed(method: OptimizationMethod) -> None:
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
        method=method,
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


@mark.parametrize('method', optimization_methods)
@mark.parametrize('propeller_type', propeller_types)
def test_torque_limit(method: OptimizationMethod, propeller_type: Type[Propeller]) -> None:
    """
    This test checks wether the torque limit is honoured by the optimizers.
    """

    thrust = 1000
    speed = 10

    prop = propeller_type(
        blades=3,
    ).optimize(
        objective=lambda p: p.losses(speed, thrust),
        method=method,
        constraints=[
            lambda p: p.torque_margin(speed, thrust, 60)
        ]
    )

    n, q = prop.find_nq_for_vt(speed, thrust)

    assert prop.torque_margin(speed, thrust, 60) > -5e-8
    assert q < 60 * (1 + 5e-8)


@mark.parametrize('method', optimization_methods)
@mark.parametrize('propeller_type', propeller_types)
def test_rpm_limit(method: OptimizationMethod, propeller_type: Type[Propeller]) -> None:
    """
    This test checks wether the rpm limit is honoured by the optimizers.
    """

    thrust = 1000
    speed = 10

    prop = propeller_type(
        blades=3
    ).optimize(
        objective=lambda p: p.losses(speed, thrust),
        method=method,
        constraints=[
            lambda p: p.rotation_speed_margin(speed, thrust, 17)
        ]
    )

    n, q = prop.find_nq_for_vt(speed, thrust)

    assert prop.rotation_speed_margin(speed, thrust, 17) > -1-15
    assert n == approx(17, rel=1e-3, abs=1e-3)


@mark.parametrize('method', optimization_methods)
@mark.parametrize('propeller_type', propeller_types)
def test_diameter_limit(method: OptimizationMethod, propeller_type: Type[Propeller]) -> None:
    """
    This test checks wether the diameter upper-limit is honoured by the optimizers.
    """

    thrust = 1000
    speed = 10

    prop = propeller_type(
        blades=3,
        diameter=0.10
    ).optimize(
        objective=lambda p: p.losses(speed, thrust),
        diameter_max=0.2,
        method=method
    )

    assert prop.diameter < (0.2 + 1e-6)


@mark.parametrize('method', optimization_methods)
@mark.parametrize('propeller_type', propeller_types)
def test_area_ratio_limit(method: OptimizationMethod, propeller_type: Type[Propeller]) -> None:
    """
    This test checks wether the cavitation margin is honoured by the optimizers.
    """
    thrust = 1000
    speed = 20
    immersion = 1

    prop = propeller_type(
        blades=3
    ).optimize(
        objective=lambda p: p.losses(speed, thrust),
        method=method,
        constraints=[
            lambda p: p.cavitation_margin(thrust, immersion)
        ]
    )

    assert prop.cavitation_margin(thrust, immersion) > -1e-6


@mark.parametrize('method', (slsqp, ))  # trust_constrained doesn't work for the Gawn propeller
@mark.parametrize('propeller_type', propeller_types)
def test_tip_speed_limit(method: OptimizationMethod, propeller_type: Type[Propeller]) -> None:
    """
    This test checks wether the tip speed margin is honoured by the optimizers
    """

    thrust = 1000
    speed = 10
    
    prop = propeller_type(
        blades=3
    ).optimize(
        objective=lambda p: p.losses(speed, thrust),
        method=method,
        constraints=[
            lambda p: p.tip_speed_margin(speed, thrust, 24)
        ]
    )

    n, q = prop.find_nq_for_vt(speed, thrust)

    assert prop.tip_speed_margin(speed, thrust, 24) > -1e-6
    assert n * pi * prop.diameter < 24 * (1 + 1e-6)
