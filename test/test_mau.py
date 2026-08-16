from propy import MAUPropeller

from pytest import raises, mark, approx
from pytest_benchmark.fixture import BenchmarkFixture
from numpy.testing import assert_allclose
from numpy import linspace


@mark.benchmark(group='Instantiation')
def test_instantiation_mau(benchmark: BenchmarkFixture) -> None:
    prop = benchmark(MAUPropeller)
    assert prop is not None


@mark.benchmark(group='open_water')
def test_kt_vec100_mau(benchmark: BenchmarkFixture) -> None:
    prop = MAUPropeller()
    j = linspace(prop.j_min, prop.j_max, 100)
    benchmark(prop.kt, j)


@mark.benchmark(group='open_water')
def test_kt_mau(benchmark: BenchmarkFixture) -> None:
    prop = MAUPropeller()
    j = 0.2
    benchmark(prop.kt, j)


@mark.benchmark(group='open_water')
def test_kq_vec100_mau(benchmark: BenchmarkFixture) -> None:
    prop = MAUPropeller()
    j = linspace(prop.j_min, prop.j_max, 100)
    benchmark(prop.kq, j)


@mark.benchmark(group='open_water')
def test_kq_mau(benchmark: BenchmarkFixture) -> None:
    prop = MAUPropeller()
    j = 0.2
    benchmark(prop.kq, j)


@mark.benchmark(group='find')
def test_j_for_vt_mau(benchmark: BenchmarkFixture) -> None:
    prop = MAUPropeller()
    benchmark(prop.find_j_for_vt, 10, 10000)


@mark.benchmark(group='find')
def test_j_for_vq_mau(benchmark: BenchmarkFixture) -> None:
    prop = MAUPropeller()
    benchmark(prop.find_j_for_vq, 10, 1000)


@mark.benchmark(group='find')
def test_j_for_nq_mau(benchmark: BenchmarkFixture) -> None:
    prop = MAUPropeller()
    benchmark(prop.find_j_for_nq, 10, 1000)


@mark.benchmark(group='find')
def test_j_for_nt_mau(benchmark: BenchmarkFixture) -> None:
    prop = MAUPropeller()
    benchmark(prop.find_j_for_nt, 10, 10000)


def test_valid_blades() -> None:
    # Test whether limits are set
    assert MAUPropeller.blades_min > 0
    assert MAUPropeller.blades_max >= MAUPropeller.blades_min

    # Test ability to instantiate at limits
    MAUPropeller(blades=MAUPropeller.blades_min)
    MAUPropeller(blades=MAUPropeller.blades_max, area_ratio=0.6)

    # Test ability to instantiate outside limits
    with raises(ValueError):
        MAUPropeller(blades=MAUPropeller.blades_min - 1)

    with raises(ValueError):
        MAUPropeller(blades=MAUPropeller.blades_max + 1)

    # Test input type
    with raises(TypeError):
        MAUPropeller(blades=float(MAUPropeller.blades_min))  # type: ignore


@mark.parametrize('blades', [3, 4, 5, 6])
def test_valid_area_ratio(blades: int) -> None:
    area_ratio_min = MAUPropeller.area_ratio_min_for_blades(blades)
    area_ratio_max = MAUPropeller.area_ratio_max_for_blades(blades)

    # Test whether limits are set
    assert area_ratio_min > 0
    assert area_ratio_max >= area_ratio_min

    # Test ability to instantiate at limits
    MAUPropeller(blades=blades, area_ratio=area_ratio_min)
    MAUPropeller(blades=blades, area_ratio=area_ratio_max)

    # Test ability to instantiate outside limits
    with raises(ValueError):
        MAUPropeller(blades=blades, area_ratio=area_ratio_min * 0.9)

    with raises(ValueError):
        MAUPropeller(blades=blades, area_ratio=area_ratio_max * 1.1)


@mark.parametrize('blades', [3, 4, 5, 6])
def test_valid_pd_ratio(blades: int) -> None:
    pd_ratio_min = MAUPropeller.pd_ratio_min_for_blades(blades)
    pd_ratio_max = MAUPropeller.pd_ratio_max_for_blades(blades)

    area_ratio_min = MAUPropeller.area_ratio_min_for_blades(blades)

    # Test whether limits are set
    assert pd_ratio_min > 0
    assert pd_ratio_max >= pd_ratio_min

    # Test ability to instantiate at limits
    MAUPropeller(blades=blades, pd_ratio=pd_ratio_min, area_ratio=area_ratio_min)
    MAUPropeller(blades=blades, pd_ratio=pd_ratio_max, area_ratio=area_ratio_min)

    # Test ability to instantiate outside limits
    with raises(ValueError):
        MAUPropeller(blades=blades, pd_ratio=pd_ratio_min * 0.9)

    with raises(ValueError):
        MAUPropeller(blades=blades, pd_ratio=pd_ratio_max * 1.1)


def test_valid_diameter() -> None:
    # Test ability to instantiate above limits
    MAUPropeller(diameter=1.0)

    # Test ability to instantiate outside limits
    with raises(ValueError):
        MAUPropeller(diameter=0.0)

    with raises(ValueError):
        MAUPropeller(diameter=-1.0)


@mark.parametrize('blades', [3, 4, 5, 6])
def test_j_range(blades: int) -> None:
    area_ratio_min = MAUPropeller.area_ratio_min_for_blades(blades)
    area_ratio_max = MAUPropeller.area_ratio_min_for_blades(blades)

    pd_ratio_min = MAUPropeller.pd_ratio_min_for_blades(blades)
    pd_ratio_max = MAUPropeller.pd_ratio_max_for_blades(blades)

    for area_ratio in linspace(area_ratio_min, area_ratio_max, 10):
        for pd_ratio in linspace(pd_ratio_min, pd_ratio_max, 10):
            p = MAUPropeller(
                blades=blades,
                area_ratio=area_ratio,
                pd_ratio=pd_ratio
            )

            # j-max should be calculated such that kt(j_max) is close to 0
            assert_allclose(0, p.kt(p.j_max), rtol=1e-15, atol=1e-15)

    
@mark.parametrize('blades', [3, 4, 5, 6])
def test_kq_range(blades: int) -> None:
    area_ratio_min = MAUPropeller.area_ratio_min_for_blades(blades)
    area_ratio_max = MAUPropeller.area_ratio_min_for_blades(blades)

    pd_ratio_min = MAUPropeller.pd_ratio_min_for_blades(blades)
    pd_ratio_max = MAUPropeller.pd_ratio_max_for_blades(blades)

    for area_ratio in linspace(area_ratio_min, area_ratio_max, 10):
        for pd_ratio in linspace(pd_ratio_min, pd_ratio_max, 10):
            p = MAUPropeller(
                blades=blades,
                area_ratio=area_ratio,
                pd_ratio=pd_ratio
            )

            # The kq-curve should stop before it's at 0, where kt=0
            assert_allclose(p.kq_min, p.kq(p.j_max), rtol=1e-15, atol=1e-15)
            assert_allclose(p.kq_max, p.kq(p.j_min), rtol=1e-15, atol=1e-15)


@mark.parametrize('blades,area_ratio', [
    (3, 0.35),
    (3, 0.50),
    (4, 0.40),
    (4, 0.55),
    (4, 0.70),
])
def test_kt_kq(blades: int, area_ratio: float) -> None:
    """
    
    """
    with open(f'test/data/MAU{blades}-{int(area_ratio*100)}.csv') as file:
        for line in file:
            print(line)
            if line.startswith('x'):
                _, pd_ratio = line.split(',')
                k_type, pd_ratio = pd_ratio.split('_')
                func = MAUPropeller(
                    blades=blades,
                    area_ratio=area_ratio,
                    pd_ratio=float(pd_ratio.strip()[-2:])/10
                ).__getattribute__(k_type)
            elif len(line.strip()) > 0:
                j: float
                k: float
                j, k = (float(x) for x in line.strip().split(','))
                if k_type == 'kq':
                    k /= 10
                assert func(j) == approx(k, rel=1e-2, abs=5e-3)
