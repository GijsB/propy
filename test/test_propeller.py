from math import atan2

from propy.propeller import Propeller
from propy.wageningen_b import WageningenBPropeller

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


@mark.parametrize('blades', [2, 4, 6])
@mark.parametrize('area_ratio', [0.3, 0.6, 0.9])
@mark.parametrize('pd_ratio', [0.5, 0.8, 1.1])
def test_4q_prop(blades: int, area_ratio: float, pd_ratio: float) -> None:
    prop = WageningenBPropeller(
        blades=blades,
        area_ratio=area_ratio,
        pd_ratio=pd_ratio
    )

    assert prop.ct(0) == approx(8 * prop.kt(0) / pi / (0.7**2 * pi**2))
    assert prop.cq(0) == approx(8 * prop.kq(0) / pi / (0.7**2 * pi**2))

    beta_max = atan2(prop.j_max, 0.7 * pi)

    assert prop.ct(beta_max) == approx(8 * prop.kt_min / pi / (prop.j_max**2 + 0.7**2 * pi**2))
    assert prop.cq(beta_max) == approx(8 * prop.kq_min / pi / (prop.j_max**2 + 0.7**2 * pi**2))


def test_finding_type_consistency() -> None:
    prop = WageningenBPropeller()

    assert isinstance(prop.find_j_for_vn(1, 1), ndarray)
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

    assert isinstance(prop.find_beta_for_vn(1, 1), ndarray)
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


@mark.parametrize('blades', [2, 4, 6])
@mark.parametrize('area_ratio', [0.3, 0.6, 0.9])
@mark.parametrize('pd_ratio', [0.5, 0.8, 1.1])
@mark.parametrize('speed', [1, 2, 5, 10, 20, 50])
@mark.parametrize('thrust', [10, 20, 50, 100, 200, 500])
def test_roundtrip_consistencies(blades: int, area_ratio: float, pd_ratio: float, speed: float, thrust: float) -> None:
    prop = WageningenBPropeller(
        blades=blades,
        area_ratio=area_ratio,
        pd_ratio=pd_ratio
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


@mark.parametrize('blades', [2, 4, 6])
@mark.parametrize('area_ratio', [0.3, 0.6, 0.9])
@mark.parametrize('pd_ratio', [0.5, 0.8, 1.1])
@mark.parametrize('speed', [1, 2, 5, 10, 20, 50])
@mark.parametrize('thrust', [10, 20, 50, 100, 200, 500])
def test_j_consistency_for_vt(blades: int, area_ratio: float, pd_ratio: float, speed: float, thrust: float) -> None:
    prop = WageningenBPropeller(
        blades=blades,
        area_ratio=area_ratio,
        pd_ratio=pd_ratio
    )
    
    j = prop.find_j_for_vt(speed, thrust)
    n, q = prop.find_nq_for_vt(speed, thrust)

    assert prop.find_j_for_nq(n, q) == approx(j)
    assert prop.find_j_for_nt(float(n), thrust) == approx(j)
    assert prop.find_j_for_vn(speed, float(n)) == approx(j)
    assert prop.find_j_for_vq(speed, float(q)) == approx(j)


@mark.parametrize('blades', [2, 4, 6])
@mark.parametrize('area_ratio', [0.3, 0.6, 0.9])
@mark.parametrize('pd_ratio', [0.5, 0.8, 1.1])
def test_kt_inv_roundtrip(blades: int, area_ratio: float, pd_ratio: float) -> None:
    prop = WageningenBPropeller(
        blades=blades,
        area_ratio=area_ratio,
        pd_ratio=pd_ratio
    )
    js = linspace(prop.j_min, prop.j_min)
    assert prop.kt_inv(prop.kt(js)) == approx(js)


@mark.parametrize('blades', [2, 4, 6])
@mark.parametrize('area_ratio', [0.3, 0.6, 0.9])
@mark.parametrize('pd_ratio', [0.5, 0.8, 1.1])
def test_kq_inv_roundtrip(blades: int, area_ratio: float, pd_ratio: float) -> None:
    prop = WageningenBPropeller(
        blades=blades,
        area_ratio=area_ratio,
        pd_ratio=pd_ratio
    )
    js = linspace(prop.j_min, prop.j_min)
    assert prop.kq_inv(prop.kq(js)) == approx(js)
