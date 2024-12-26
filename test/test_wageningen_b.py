from propy.wageningen_b import WageningenBPropeller

from pytest import raises
from numpy.testing import assert_allclose

def test_valid_blades():
    # Test whether limits are set
    assert WageningenBPropeller.blades_min > 0
    assert WageningenBPropeller.blades_max >= WageningenBPropeller.blades_min

    # Test ability to instantiate at limits
    WageningenBPropeller(blades=WageningenBPropeller.blades_min)
    WageningenBPropeller(blades=WageningenBPropeller.blades_max)

    # Test ability to instantiate outside limits
    with raises(ValueError):    WageningenBPropeller(blades=WageningenBPropeller.blades_min - 1)
    with raises(ValueError):    WageningenBPropeller(blades=WageningenBPropeller.blades_max + 1)

    # Test input type
    with raises(TypeError):     WageningenBPropeller(blades=float(WageningenBPropeller.blades_min))


def test_valid_area_ratio():
    # Test whether limits are set
    assert WageningenBPropeller.area_ratio_min > 0
    assert WageningenBPropeller.area_ratio_max >= WageningenBPropeller.area_ratio_min

    # Test ability to instantiate at limits
    WageningenBPropeller(area_ratio=WageningenBPropeller.area_ratio_min)
    WageningenBPropeller(area_ratio=WageningenBPropeller.area_ratio_max)

    # Test ability to instantiate outside limits
    with raises(ValueError):    WageningenBPropeller(area_ratio=WageningenBPropeller.area_ratio_min * 0.9)
    with raises(ValueError):    WageningenBPropeller(area_ratio=WageningenBPropeller.area_ratio_max * 1.1)


def test_valid_pd_ratio():
    # Test whether limits are set
    assert WageningenBPropeller.pd_ratio_min > 0
    assert WageningenBPropeller.pd_ratio_max >= WageningenBPropeller.pd_ratio_min

    # Test ability to instantiate at limits
    WageningenBPropeller(pd_ratio=WageningenBPropeller.pd_ratio_min)
    WageningenBPropeller(pd_ratio=WageningenBPropeller.pd_ratio_max)

    # Test ability to instantiate outside limits
    with raises(ValueError):    WageningenBPropeller(pd_ratio=WageningenBPropeller.pd_ratio_min * 0.9)
    with raises(ValueError):    WageningenBPropeller(pd_ratio=WageningenBPropeller.pd_ratio_max * 1.1)


def test_valid_diameter():
    # Test ability to instantiate above limits
    WageningenBPropeller(diameter=1.0)

    # Test ability to instantiate outside limits
    with raises(ValueError):    WageningenBPropeller(diameter=0.0)
    with raises(ValueError):    WageningenBPropeller(diameter=-1.0)


def test_kt_range():
    p = WageningenBPropeller()
    # The kt-curve should run all the way down to 0
    assert p.kt_min == 0

    # Kt should be max at j=0
    assert_allclose(p.kt_max, p.kt(0))


def test_j_range():
    p = WageningenBPropeller()
    # j-max should be calculated such that kt(j_max) is close to 0
    assert_allclose(0, p.kt(p.j_max))


def test_kq_range():
    p = WageningenBPropeller()
    # The kq-curve should stop before it's at 0, where kt=0
    assert_allclose(p.kq_min, p.kq(p.j_max))
    assert_allclose(p.kq_max, p.kq(0))