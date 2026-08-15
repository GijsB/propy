from math import atan2

from propy.propeller import Propeller
from propy.wageningen_b import WageningenBPropeller
from propy.gawn_burrill import GawnBurrillPropeller
from propy.mau import MAUPropeller


from pytest import raises, approx, mark
from pytest_benchmark.fixture import BenchmarkFixture
from numpy import pi, array, ndarray, linspace


@mark.benchmark(group='find')
def test_j_for_vn(benchmark: BenchmarkFixture) -> None:
    prop = WageningenBPropeller()
    benchmark(prop.find_j_for_vn, 10, 10)


@mark.benchmark(group='find')
def test_beta_for_vn(benchmark: BenchmarkFixture) -> None:
    prop = WageningenBPropeller()
    benchmark(prop.find_beta_for_vn, 10, 10)


def test_instantiation() -> None:
    """Check whether instantiation of an abstract Propeller raises a TypeError"""
    with raises(TypeError):
        # noinspection PyAbstractClass
        Propeller()  # type: ignore


def test_new() -> None:
    """Check whether calling new (on ABC) raises a TypeError"""
    with raises(TypeError):
        Propeller.new()


@mark.parametrize('blades', [2, 3, 4, 5, 6])
@mark.parametrize('area_ratio_rel', [0.1, 0.5, 0.9])
@mark.parametrize('pd_ratio_rel', [0.1, 0.5, 0.9])
@mark.parametrize('prop_type', [WageningenBPropeller, GawnBurrillPropeller, MAUPropeller])
def test_4q_prop(blades: int, area_ratio_rel: float, pd_ratio_rel: float, prop_type: type[Propeller]) -> None:
    if not (prop_type.blades_min <= blades <= prop_type.blades_max):
        return
        
    area_ratio_min = prop_type.area_ratio_min_for_blades(blades)
    area_ratio_max = prop_type.area_ratio_min_for_blades(blades)

    pd_ratio_min = prop_type.pd_ratio_min_for_blades(blades)
    pd_ratio_max = prop_type.pd_ratio_max_for_blades(blades)

    prop = prop_type(
        blades=blades,
        area_ratio=area_ratio_min + area_ratio_rel * (area_ratio_max - area_ratio_min),
        pd_ratio=pd_ratio_min + pd_ratio_rel * (pd_ratio_max - pd_ratio_min)
    )

    beta_min = atan2(prop.j_min, 0.7 * pi)

    assert prop.ct(beta_min) == approx(8 * prop.kt_max / pi / (prop.j_min**2 + 0.7**2 * pi**2))
    assert prop.cq(beta_min) == approx(8 * prop.kq_max / pi / (prop.j_min**2 + 0.7**2 * pi**2))

    beta_max = atan2(prop.j_max, 0.7 * pi)

    assert prop.ct(beta_max) == approx(8 * prop.kt_min / pi / (prop.j_max**2 + 0.7**2 * pi**2))
    assert prop.cq(beta_max) == approx(8 * prop.kq_min / pi / (prop.j_max**2 + 0.7**2 * pi**2))


@mark.parametrize('propeller_type', [WageningenBPropeller, GawnBurrillPropeller, MAUPropeller])
def test_finding_type_consistency(propeller_type: type[Propeller]) -> None:
    prop = propeller_type()

    assert isinstance(prop.find_j_for_vn(1, 1), float)
    assert isinstance(prop.find_j_for_vn(array([1]), array([1])), ndarray)
    assert (prop.find_j_for_vn(array([1.23456]), array([6.789]))[0] ==
            approx(prop.find_j_for_vn(1.23456, 6.789)))

    assert isinstance(prop.find_j_for_vt(1, 1), ndarray)
    assert isinstance(prop.find_j_for_vt(array([1]), array([1])), ndarray)
    assert (prop.find_j_for_vt(array([1.23456]), array([6.789]))[0] ==
            approx(prop.find_j_for_vt(1.23456, 6.789)))
    
    assert isinstance(prop.find_j_for_vq(1, 1), ndarray)
    assert isinstance(prop.find_j_for_vq(array([1]), array([1])), ndarray)
    assert (prop.find_j_for_vq(array([1.23456]), array([6.789]))[0] ==
            approx(prop.find_j_for_vq(1.23456, 6.789)))
    
    assert isinstance(prop.find_j_for_nq(1, 1), ndarray)
    assert isinstance(prop.find_j_for_nq(array([1]), array([1])), ndarray)
    assert (prop.find_j_for_nq(array([1.23456]), array([6.789]))[0] ==
            approx(prop.find_j_for_nq(1.23456, 6.789)))
    
    assert isinstance(prop.find_j_for_nt(1, 1), ndarray)
    assert isinstance(prop.find_j_for_nt(array([1]), array([1])), ndarray)
    assert (prop.find_j_for_nt(array([1.23456]), array([6.789]))[0] ==
            approx(prop.find_j_for_nt(1.23456, 6.789)))

    assert isinstance(prop.find_beta_for_vn(1, 1), float)
    assert isinstance(prop.find_beta_for_vn(array([1]), array([1])), ndarray)
    assert (prop.find_beta_for_vn(array([1.23456]), array([6.789]))[0] ==
            approx(prop.find_beta_for_vn(1.23456, 6.789)))

    t, q = prop.find_tq_for_vn(1.23456, 6.789)
    t_vec, q_vec = prop.find_tq_for_vn(array([1.23456]), array([6.789]))
    assert t_vec[0] == approx(t)
    assert q_vec[0] == approx(q)

    n, q = prop.find_nq_for_vt(1.23456, 6.789)
    n_vec, q_vec = prop.find_nq_for_vt(array([1.23456]), array([6.789]))
    assert n_vec[0] == approx(n)
    assert q_vec[0] == approx(q)

    assert isinstance(prop.ct(1), ndarray)
    assert isinstance(prop.ct(1.0), ndarray)
    assert isinstance(prop.ct(array([1, 2])), ndarray)
    assert prop.ct(array([0.123456]))[0] == approx(prop.ct(0.123456))

    assert isinstance(prop.cq(1), ndarray)
    assert isinstance(prop.cq(1.0), ndarray)
    assert isinstance(prop.cq(array([1, 2])), ndarray)
    assert prop.cq(array([0.123456]))[0] == approx(prop.cq(0.123456))

    assert isinstance(prop.kt(0.1), ndarray)
    assert isinstance(prop.kt(array([0.1, 0.2])), ndarray)
    assert prop.kt(array([0.123456]))[0] == approx(prop.kt(0.123456))

    assert isinstance(prop.kq(0.1), ndarray)
    assert isinstance(prop.kq(array([0.1, 0.2])), ndarray)
    assert prop.kq(array([0.123456]))[0] == approx(prop.kq(0.123456))

    assert isinstance(prop.eta(0.1), ndarray)
    assert isinstance(prop.eta(array([0.1, 0.2])), ndarray)
    assert prop.eta(array([0.123456]))[0] == approx(prop.eta(0.123456))


@mark.parametrize('blades', [2, 3, 4, 5, 6])
@mark.parametrize('area_ratio_rel', [0.1, 0.5, 0.9])
@mark.parametrize('pd_ratio_rel', [0.1, 0.5, 0.9])
@mark.parametrize('speed', [1, 2, 5, 10, 20, 50])
@mark.parametrize('thrust', [10, 20, 50, 100, 200, 500])
@mark.parametrize('prop_type', [WageningenBPropeller, GawnBurrillPropeller, MAUPropeller])
def test_roundtrip_consistencies(
    blades: int,
    area_ratio_rel: float,
    pd_ratio_rel: float,
    speed: float,
    thrust: float,
    prop_type: type[Propeller]
) -> None:
    if not (prop_type.blades_min <= blades <= prop_type.blades_max):
        return
        
    area_ratio_min = prop_type.area_ratio_min_for_blades(blades)
    area_ratio_max = prop_type.area_ratio_min_for_blades(blades)

    pd_ratio_min = prop_type.pd_ratio_min_for_blades(blades)
    pd_ratio_max = prop_type.pd_ratio_max_for_blades(blades)

    prop = prop_type(
        blades=blades,
        area_ratio=area_ratio_min + area_ratio_rel * (area_ratio_max - area_ratio_min),
        pd_ratio=pd_ratio_min + pd_ratio_rel * (pd_ratio_max - pd_ratio_min)
    )

    n, q = prop.find_nq_for_vt(speed, thrust)
    v, t = prop.find_vt_for_nq(n, q)

    assert v == approx(speed)
    assert t == approx(thrust)

    v, q2 = prop.find_vq_for_nt(float(n), thrust)

    assert v == approx(speed)
    assert q2 == approx(q)

    t, q3 = prop.find_tq_for_vn(speed, float(n))

    assert t == approx(thrust)
    assert q3 == approx(q)

    n2, t2 = prop.find_nt_for_vq(speed, float(q))

    assert n2 == approx(n)
    assert t2 == approx(thrust)


@mark.parametrize('blades', [2, 3, 4, 5, 6])
@mark.parametrize('area_ratio_rel', [0.1, 0.5, 0.9])
@mark.parametrize('pd_ratio_rel', [0.1, 0.5, 0.9])
@mark.parametrize('speed', [1, 2, 5, 10, 20, 50])
@mark.parametrize('thrust', [10, 20, 50, 100, 200, 500])
@mark.parametrize('prop_type', [WageningenBPropeller, GawnBurrillPropeller, MAUPropeller])
def test_j_consistency_for_vt(
    blades: int,
    area_ratio_rel: float,
    pd_ratio_rel: float,
    speed: float,
    thrust: float,
    prop_type: type[Propeller]
) -> None:
    if not (prop_type.blades_min <= blades <= prop_type.blades_max):
        return
    
    area_ratio_min = prop_type.area_ratio_min_for_blades(blades)
    area_ratio_max = prop_type.area_ratio_min_for_blades(blades)

    pd_ratio_min = prop_type.pd_ratio_min_for_blades(blades)
    pd_ratio_max = prop_type.pd_ratio_max_for_blades(blades)

    prop = prop_type(
        blades=blades,
        area_ratio=area_ratio_min + area_ratio_rel * (area_ratio_max - area_ratio_min),
        pd_ratio=pd_ratio_min + pd_ratio_rel * (pd_ratio_max - pd_ratio_min)
    )
    
    j = prop.find_j_for_vt(speed, thrust)
    n, q = prop.find_nq_for_vt(speed, thrust)

    assert prop.find_j_for_nq(n, q) == approx(j)
    assert prop.find_j_for_nt(float(n), thrust) == approx(j)
    assert prop.find_j_for_vn(speed, float(n)) == approx(j)
    assert prop.find_j_for_vq(speed, float(q)) == approx(j)


@mark.parametrize('blades', [2, 3, 4, 5, 6])
@mark.parametrize('area_ratio_rel', [0.1, 0.5, 0.9])
@mark.parametrize('pd_ratio_rel', [0.1, 0.5, 0.9])
@mark.parametrize('prop_type', [WageningenBPropeller, GawnBurrillPropeller, MAUPropeller])
def test_kt_inv_roundtrip(blades: int, area_ratio_rel: float, pd_ratio_rel: float, prop_type: type[Propeller]) -> None:
    if not (prop_type.blades_min <= blades <= prop_type.blades_max):
        return
    
    area_ratio_min = prop_type.area_ratio_min_for_blades(blades)
    area_ratio_max = prop_type.area_ratio_min_for_blades(blades)

    pd_ratio_min = prop_type.pd_ratio_min_for_blades(blades)
    pd_ratio_max = prop_type.pd_ratio_max_for_blades(blades)

    prop = prop_type(
        blades=blades,
        area_ratio=area_ratio_min + area_ratio_rel * (area_ratio_max - area_ratio_min),
        pd_ratio=pd_ratio_min + pd_ratio_rel * (pd_ratio_max - pd_ratio_min)
    )
    js = linspace(prop.j_min, prop.j_min)
    assert prop.kt_inv(prop.kt(js)) == approx(js)


@mark.parametrize('blades', [2, 3, 4, 5, 6])
@mark.parametrize('area_ratio_rel', [0.1, 0.5, 0.9])
@mark.parametrize('pd_ratio_rel', [0.1, 0.5, 0.9])
@mark.parametrize('prop_type', [WageningenBPropeller, GawnBurrillPropeller, MAUPropeller])
def test_kq_inv_roundtrip(blades: int, area_ratio_rel: float, pd_ratio_rel: float, prop_type: type[Propeller]) -> None:
    if not (prop_type.blades_min <= blades <= prop_type.blades_max):
        return
    
    area_ratio_min = prop_type.area_ratio_min_for_blades(blades)
    area_ratio_max = prop_type.area_ratio_min_for_blades(blades)

    pd_ratio_min = prop_type.pd_ratio_min_for_blades(blades)
    pd_ratio_max = prop_type.pd_ratio_max_for_blades(blades)

    prop = prop_type(
        blades=blades,
        area_ratio=area_ratio_min + area_ratio_rel * (area_ratio_max - area_ratio_min),
        pd_ratio=pd_ratio_min + pd_ratio_rel * (pd_ratio_max - pd_ratio_min)
    )
    js = linspace(prop.j_min, prop.j_min)
    assert prop.kq_inv(prop.kq(js)) == approx(js)
