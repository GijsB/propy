from propy import GawnBurrillPropeller

from pytest import raises, mark
from numpy.testing import assert_allclose

p = GawnBurrillPropeller()


def test_valid_blades() -> None:
    # Test whether limits are set
    assert GawnBurrillPropeller.blades_min > 0
    assert GawnBurrillPropeller.blades_max >= GawnBurrillPropeller.blades_min

    # Test ability to instantiate at limits
    GawnBurrillPropeller(blades=GawnBurrillPropeller.blades_min)
    GawnBurrillPropeller(blades=GawnBurrillPropeller.blades_max)

    # Test ability to instantiate outside limits
    with raises(ValueError):
        GawnBurrillPropeller(blades=GawnBurrillPropeller.blades_min - 1)

    with raises(ValueError):
        GawnBurrillPropeller(blades=GawnBurrillPropeller.blades_max + 1)

    # Test input type
    with raises(TypeError):
        GawnBurrillPropeller(blades=float(GawnBurrillPropeller.blades_min))  # type: ignore


def test_valid_area_ratio() -> None:
    # Test whether limits are set
    assert GawnBurrillPropeller.area_ratio_min > 0
    assert GawnBurrillPropeller.area_ratio_max >= GawnBurrillPropeller.area_ratio_min

    # Test ability to instantiate at limits
    GawnBurrillPropeller(area_ratio=GawnBurrillPropeller.area_ratio_min)
    GawnBurrillPropeller(area_ratio=GawnBurrillPropeller.area_ratio_max)

    # Test ability to instantiate outside limits
    with raises(ValueError):
        GawnBurrillPropeller(area_ratio=GawnBurrillPropeller.area_ratio_min * 0.9)

    with raises(ValueError):
        GawnBurrillPropeller(area_ratio=GawnBurrillPropeller.area_ratio_max * 1.1)


def test_valid_pd_ratio() -> None:
    # Test whether limits are set
    assert GawnBurrillPropeller.pd_ratio_min > 0
    assert GawnBurrillPropeller.pd_ratio_max >= GawnBurrillPropeller.pd_ratio_min

    # Test ability to instantiate at limits
    GawnBurrillPropeller(pd_ratio=GawnBurrillPropeller.pd_ratio_min)
    GawnBurrillPropeller(pd_ratio=GawnBurrillPropeller.pd_ratio_max)

    # Test ability to instantiate outside limits
    with raises(ValueError):
        GawnBurrillPropeller(pd_ratio=GawnBurrillPropeller.pd_ratio_min * 0.9)

    with raises(ValueError):
        GawnBurrillPropeller(pd_ratio=GawnBurrillPropeller.pd_ratio_max * 1.1)


def test_valid_diameter() -> None:
    # Test ability to instantiate above limits
    GawnBurrillPropeller(diameter=1.0)

    # Test ability to instantiate outside limits
    with raises(ValueError):
        GawnBurrillPropeller(diameter=0.0)

    with raises(ValueError):
        GawnBurrillPropeller(diameter=-1.0)


def test_j_range() -> None:
    # j-max should be calculated such that kt(j_max) is close to 0
    assert_allclose(0, p.kt(p.j_max), rtol=1e-15, atol=1e-15)


def test_kq_range() -> None:
    # The kq-curve should stop before it's at 0, where kt=0
    assert_allclose(p.kq_min, p.kq(p.j_max), rtol=1e-15, atol=1e-15)
    assert_allclose(p.kq_max, p.kq(p.j_min), rtol=1e-15, atol=1e-15)

@mark.parametrize('area_ratio,pd_ratio,j,kt', [
    (0.50, 0.8, [0.4, 0.6, 0.8], [0.22, 0.13, 0.03]),
    (0.50, 1.2, [0.6, 0.8, 1.0, 1.2], [0.34, 0.23, 0.14, 0.04]),
    (0.50, 1.6, [0.8, 1.0, 1.2, 1.4, 1.6], [0.43, 0.31, 0.23, 0.13, 0.04]),
    (0.90, 0.8, [0.4, 0.6, 0.8], [0.22, 0.11, 0.01]),
    (0.90, 1.2, [0.6, 0.8, 1.0, 1.2], [0.36, 0.25, 0.12, 0.03]),
    (0.90, 1.6, [1.0, 1.2, 1.4, 1.6], [0.35, 0.24, 0.11, 0.03]),
])
def test_kt_radojcic(area_ratio: float, pd_ratio: float, j: float, kt: float) -> None:
    """
    Compare the calculated kt values with manual chart readings from [1].

        [1] D. Radojcic, Mathematical Model of Segmental Section Propeller Series for Open-Water and Cavitating
        Conditions Applicable in CAD.
    """
    prop = GawnBurrillPropeller(
        area_ratio=area_ratio,
        pd_ratio=pd_ratio
    )
    assert_allclose(prop.kt(j), kt, atol=0.02)


@mark.parametrize('area_ratio,pd_ratio,j,kq', [
    (0.50, 0.8, [0.4, 0.6, 0.8], [0.026, 0.018, 0.006]),
    (0.50, 1.2, [0.6, 0.8, 1.0, 1.2], [0.06, 0.043, 0.028, 0.014]),
    (0.50, 1.6, [0.8, 1.0, 1.2, 1.4, 1.6], [0.1, 0.078, 0.057, 0.037, 0.018]),
    (0.90, 0.8, [0.4, 0.6, 0.8], [0.028, 0.018, 0.006]),
    (0.90, 1.2, [0.6, 0.8, 1.0, 1.2], [0.068, 0.047, 0.028, 0.012]),
    (0.90, 1.6, [1.0, 1.2, 1.4, 1.6], [0.089, 0.062, 0.036, 0.014]),
])
def test_kq_radojcic(area_ratio: float, pd_ratio: float, j: float, kq: float) -> None:
    """
    Compare the calculated kq values with manual chart readings from [1].

        [1] D. Radojcic, Mathematical Model of Segmental Section Propeller Series for Open-Water and Cavitating
        Conditions Applicable in CAD.
    """
    prop = GawnBurrillPropeller(
        area_ratio=area_ratio,
        pd_ratio=pd_ratio
    )
    assert_allclose(prop.kq(j), kq, atol=0.001)