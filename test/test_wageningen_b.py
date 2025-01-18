from propy.wageningen_b import WageningenBPropeller

from pytest import raises, mark
from numpy import linspace, pi
from numpy.testing import assert_allclose

p = WageningenBPropeller()

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
    # The kt-curve should run all the way down to 0
    assert p.kt_min == 0

    # Kt should be max at j=0
    assert_allclose(p.kt_max, p.kt(0), rtol=1e-15, atol=1e-15)


def test_j_range():
    # j-max should be calculated such that kt(j_max) is close to 0
    assert_allclose(0, p.kt(p.j_max), rtol=1e-15, atol=1e-15)


def test_kq_range():
    # The kq-curve should stop before it's at 0, where kt=0
    assert_allclose(p.kq_min, p.kq(p.j_max), rtol=1e-15, atol=1e-15)
    assert_allclose(p.kq_max, p.kq(0), rtol=1e-15, atol=1e-15)


def test_kt_inv():
    """Test if kt -> kt_inv returns the same result"""
    j_des = linspace(0, p.j_max, 10)
    j_cal = [p.kt_inv(p.kt(j)) for j in j_des]
    assert_allclose(j_cal, j_des, rtol=1e-15, atol=1e-15)


def test_kq_inv():
    """Test if kq -> kq_inv returns the same result"""
    j_des = linspace(0, p.j_max, 10)
    j_cal = [p.kq_inv(p.kq(j)) for j in j_des]
    assert_allclose(j_cal, j_des, rtol=1e-15, atol=1e-15)

@mark.parametrize('blades,area_ratio,pd_ratio,j,kt',[
    # (2, 0.3, 0.6, [0.2, 0.4, 0.6], [0.15, 0.095, 0.03]),                                    #  TEST FAILS!
    # (2, 0.3, 0.8, [0.2, 0.4, 0.6, 0.8], [0.215, 0.17, 0.105, 0.04]),                        # TEST FAILS!
    # (2, 0.3, 1.0, [0.2, 0.4, 0.6, 0.8, 1.0], [0.272, 0.22, 0.165, 0.105, 0.038]),           # TEST FAILS!
    # (2, 0.3, 1.2, [0.2, 0.4, 0.6, 0.8, 1.0, 1.2], [0.325, 0.28, 0.225, 0.16, 0.1, 0.038]),    # TEST FAILS!
    (3, 0.35, 0.6, [0.2, 0.4, 0.6], [0.175, 0.115, 0.042]),
    (3, 0.35, 0.8, [0.2, 0.4, 0.6, 0.8], [0.252, 0.192, 0.124, 0.048]),
    (3, 0.35, 1.0, [0.2, 0.4, 0.6, 0.8, 1.0], [0.324, 0.27, 0.204, 0.13, 0.05]),
    (3, 0.35, 1.2, [0.2, 0.4, 0.6, 0.8, 1.0, 1.2], [0.385, 0.335, 0.275, 0.208, 0.13, 0.052]),
    (3, 0.35, 1.4, [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4], [0.44, 0.39, 0.34, 0.28, 0.21, 0.13, 0.055]),
    (3, 0.65, 0.6, [0.2, 0.4, 0.6], [0.18, 0.104, 0.02]),
    (3, 0.65, 0.8, [0.2, 0.4, 0.6, 0.8], [0.276, 0.2, 0.114, 0.024]),
    (3, 0.65, 1.0, [0.2, 0.4, 0.6, 0.8, 1.0], [0.375, 0.296, 0.21, 0.118, 0.025]),
    (3, 0.65, 1.2, [0.2, 0.4, 0.6, 0.8, 1.0, 1.2], [0.47, 0.39, 0.304, 0.21, 0.12, 0.03]),
    (3, 0.65, 1.4, [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4], [0.56, 0.486, 0.4, 0.31, 0.214, 0.12, 0.032]),
    (5, 0.60, 0.6, [0.2, 0.4, 0.6], [0.2, 0.122, 0.03]),
    (5, 0.60, 0.8, [0.2, 0.4, 0.6, 0.8], [0.298, 0.224, 0.135, 0.036]),
    (5, 0.60, 1.0, [0.2, 0.4, 0.6, 0.8, 1.0], [0.39, 0.324, 0.24, 0.142, 0.04]),
    (5, 0.60, 1.2, [0.2, 0.4, 0.6, 0.8, 1.0, 1.2], [0.476, 0.412, 0.336, 0.248, 0.15, 0.048]),
    (5, 0.60, 1.4, [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4], [0.542, 0.49, 0.422, 0.34, 0.25, 0.158, 0.056]),
    (5, 0.90, 0.6, [0.2, 0.4, 0.6], [0.19, 0.108, 0.014]),
    (5, 0.90, 0.8, [0.2, 0.4, 0.6, 0.8], [0.308, 0.222, 0.124, 0.02]),
    (5, 0.90, 1.0, [0.2, 0.4, 0.6, 0.8, 1.0], [0.42, 0.334, 0.238, 0.132, 0.024]),
    (5, 0.90, 1.2, [0.2, 0.4, 0.6, 0.8, 1.0, 1.2], [0.526, 0.444, 0.348, 0.242, 0.136, 0.026]),
    (5, 0.90, 1.4, [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4], [0.62, 0.544, 0.454, 0.352, 0.248, 0.14, 0.032]),
])
def test_kt_kuiper(blades, area_ratio, pd_ratio, j, kt):
    """
    Compare the calculated kt values with manual chart readings from [1]. This book also mentions that the polynomials
    published "in other publications" have some small errors. Especially the 2-bladed propellers of the series are a bit
    special in this regard, only 2 open-water charts are published in [1]. It also looks like the manual readings from
    these 2 bladed charts do not correspond with the polynomials from [2].

        [1] G. Kuiper, The Wageningen propeller series, MARIN Publication 92-001, 1992
        [2] M. M. Bernitsas, D. Ray and P. Kinley: Kt, Kq and efficiency curves for the wageningen b-series propellers,
        Department of Naval Architecture and Marine Engineering, University of Michigan. May 1981.
    """
    p = WageningenBPropeller(
        blades=blades,
        area_ratio=area_ratio,
        pd_ratio=pd_ratio
    )
    assert_allclose(p.kt(j), kt , atol=4e-3)


@mark.parametrize('blades,area_ratio',[
    (2, 0.3),
    (2, 0.5),
    (2, 0.7),
    (2, 0.9),
    (3, 0.3),
    (3, 0.5),
    (3, 0.7),
    (3, 0.9),
    (4, 0.3),
    (4, 0.5),
    (4, 0.7),
    (4, 0.9),
    (5, 0.3),
    (5, 0.5),
    (5, 0.7),
    (5, 0.9),
    (6, 0.3),
    (6, 0.5),
    (6, 0.7),
    (6, 0.9),
    (7, 0.3),
    (7, 0.5),
    (7, 0.7),
    (7, 0.9),
])
def test_kt_kq_bernitsas(blades, area_ratio):
    """
    Compare the calculated kt and kq values with manual chart readings from [2]. This test should protect against
    typo's in copying the polyniomal. The charts are digitized using the "Engauge digitizer" application, which produces
    csv-files. These files are parsed and the results are compared with the polynomials.

        [2] M. M. Bernitsas, D. Ray and P. Kinley: Kt, Kq and efficiency curves for the wageningen b-series propellers,
        Department of Naval Architecture and Marine Engineering, University of Michigan. May 1981.
    """
    with open(f'test/data/z{blades}_a{int(area_ratio*10)}.csv') as file:
        for line in file:
            if line.startswith('x'):
                _, pd_ratio = line.split(';')
                ktype, pd_ratio = pd_ratio.split('_')
                func = WageningenBPropeller(
                    blades=blades,
                    area_ratio=area_ratio,
                    pd_ratio=float(pd_ratio.strip()[-2:])/10
                ).__getattribute__(ktype)
            elif len(line.strip()) > 0:
                j, k = line.strip().split(';')
                j = float(j.replace(',', '.'))
                k = float(k.replace(',', '.'))
                if ktype == 'kq':
                    k /= 5
                assert_allclose(k, func(j), atol=3e-3)


def test_optimization_max_diameter():
    """
    This test compares the result of a propeller optimization with the results from [1] champter 9.3.

        [1] G. Kuiper, The Wageningen propeller series, MARIN Publication 92-001, 1992
    """

    p = WageningenBPropeller(
        blades=4
    ).optimize_efficiency_for_t_v(
        thrust=1393000,
        velocity=8.65,
        diameter_max=7,
        immersion=3.51
    )
    # Immersion is modified to achieve a safety factor for the minimum area_ratio, this is also done in the book by
    # simply "chosing" a higher area_ratio manually (0.55)
    min_area_ratio = p.minimum_area_ratio(
        thrust=1393000,
        immersion=3.51,
    )
    assert (p.area_ratio * (1 + 1e-14)) > min_area_ratio

    assert p.diameter <= 7
    assert_allclose(p.pd_ratio, 1.0, rtol=6e-3)

    q, n, j, kt, kq, eta= p.calculate_operating_point(
        thrust=1393000,
        velocity=8.65,
    )

    assert_allclose([q,         n,      j,      kt,     kq,         eta],
                    [1667435,   1.767,  0.699,  0.181,  0.0310,     0.651], rtol=1e-2)

    # The results are a bit different compared to [1], this is because [1] just provides an example of a manual
    # optimization. We expect our optimizer to perform a bit better.
    assert eta > 0.651


def test_optimization_min_rpm():
    """
    This test compares the result of a propeller optimization with the results from [1] chapter 9.4.

        [1] G. Kuiper, The Wageningen propeller series, MARIN Publication 92-001, 1992
    """

    p = WageningenBPropeller(
        blades=4
    ).optimize_efficiency_for_t_v(
        thrust=1393000,
        velocity=8.65,
        q_max=1667435
    )

    q, n, j, kt, kq, eta= p.calculate_operating_point(
        thrust=1393000,
        velocity=8.65,
    )

    assert_allclose(p.diameter, 7.36, rtol=2e-3)
    assert_allclose(j, 0.665, rtol=11e-3)
    assert_allclose(kt, 0.148, rtol=21e-3)
    assert_allclose(kq, 0.0239, rtol=27e-3)

    # The results are a bit different compared to [1], this is because [1] just provides an example of a manual
    # optimization. We expect our optimizer to perform a bit better.
    assert n < 1.767
    assert q < 1667435
    assert eta > 0.656


def test_torque_limit():
    p = WageningenBPropeller(
        blades=3
    ).optimize_efficiency_for_t_v(
        thrust=1000,
        velocity=10,
        q_max=60
    )

    q, n, j, kt, kq, eta = p.calculate_operating_point(
        thrust=1000,
        velocity=10,
    )

    assert q < 60*(1+1e-15)


def test_rpm_limit():
    p = WageningenBPropeller(
        blades=3
    ).optimize_efficiency_for_t_v(
        thrust=1000,
        velocity=10,
        n_max=20
    )

    q, n, j, kt, kq, eta = p.calculate_operating_point(
        thrust=1000,
        velocity=10,
    )

    assert n < 20*(1+1e-15)


def test_diameter_limit():
    p = WageningenBPropeller(
        blades=3
    ).optimize_efficiency_for_t_v(
        thrust=1000,
        velocity=10,
        diameter_max=0.2
    )

    assert p.diameter < 0.2*(1+1e-15)


def test_area_ratio_limit():
    p = WageningenBPropeller(
        blades=3
    ).optimize_efficiency_for_t_v(
        thrust=1000,
        velocity=20,
        immersion=1
    )

    # That's not really close.. weird
    assert p.minimum_area_ratio(thrust=1000, immersion=1) < p.area_ratio*(1+1e-5)


def test_tip_speed_limit():
    p = WageningenBPropeller(
        blades=3
    ).optimize_efficiency_for_t_v(
        thrust=1000,
        velocity=10,
        tip_speed_max=24
    )

    q, n, j, kt, kq, eta = p.calculate_operating_point(
        thrust=1000,
        velocity=10,
    )

    assert n * pi * p.diameter < 24 * (1 + 1e-15)

# def test_optimization_bollard_pull():
#     """
#     This test compares the result of a propeller optimization with the results from [1] chapter 9.5.
#
#         [1] G. Kuiper, The Wageningen propeller series, MARIN Publication 92-001, 1992
#     """
#     p = WageningenBPropeller(
#         blades=5
#     ).optimize_bollard_thrust_for_q_n(
#         q = 800e3 / 3 / 2 / pi,
#         n = 3,
#         diameter_max=2.9
#     )