from pendulum import Pendulum
import pytest
import numpy as np


def test_Pendulum__call__():
    L, omega, theta = 2.7, 0.15, np.pi/6
    obj = Pendulum(L)
    y = (theta, omega)
    dy = obj(None, y)
    dy_expected = (0.15, (-109/60))
    eps = 1e-6
    assert abs(dy_expected[0] - dy[0]) < eps and abs(dy_expected[1] - dy[1])  < eps, \
                     f'dy_expected:{dy_expected}, dy:{dy}, params: L:{L}, omega:{omega}, theta:{theta}'

def test_Pendulum_valuesnotcalculated():
    # dt, T = 0.1, 100
    # y0 = (0,0)
    # t_expected = np.arange(dt, T, dt)
    obj = Pendulum()
    # obj.solve(y0, T, dt)

    with pytest.raises(Exception):
        obj.t
        obj.omega
        obj.theta

def test_Pendulum_init_zeros():
    dt, T = 0.11, 100
    y0 = (0,0)
    # t_expected = np.arange(dt, T, dt)
    obj = Pendulum()
    obj.solve(y0, T, dt)
    eps = 1e-14
    assert obj.theta.all() < eps and obj.omega.all() < eps and len(obj.t) == T//dt, f'test_Pendulum_init_zeros: theta: {obj.theta}, omega: {obj.omega}, should be zero (or less than {eps})'

def test_Pendulum_cartesian():
    L, omega, theta = 2.7, 0.15, np.pi/6
    dt, T = 0.11, 100
    obj = Pendulum(L=L)
    y0 = (omega,theta)
    obj.solve(y0,T, dt)
    x = obj.x
    y = obj.y
    eps = 1e-12
    for i,j in zip(x,y):
        assert abs((i**2 + j**2) - L**2) < eps, 'test_Pendulum_cartesian'


if __name__ == "__main__":
    test_Pendulum__call__()
    test_Pendulum_valuesnotcalculated()
    test_Pendulum_init_zeros()
    test_Pendulum_cartesian()

    
    