import numpy as np
import pytest

from double_pendulum import DoublePendulum


def test_DoublePendulum_zeroes_as_y0():
    y0 = (0 ,0 ,0 , 0)
    obj = DoublePendulum()
    obj.solve(y0, 5, 0.0001)
    y = obj.y

    for i in y:
        assert all(i == 0), "all vals in y is not zero when they should be"

def test_DoublePendulum_properties_not_implemented():
    obj = DoublePendulum()
    # y0 = (0 ,0 ,0 , 0)
    # obj.solve(y0, 5, 0.0001)
    # properties = [a for a in dir(obj) if not a.startswith(('__','_')) and not callable(getattr(obj,a))]

    with pytest.raises(ValueError):
        obj.t
    with pytest.raises(ValueError):   
        obj.omega1
    with pytest.raises(ValueError):   
        obj.theta1
    with pytest.raises(ValueError):   
        obj.theta2
    with pytest.raises(ValueError):   
        obj.omega2
    with pytest.raises(ValueError):   
        obj.y
    with pytest.raises(ValueError):   
        obj.x1
    with pytest.raises(ValueError):   
        obj.y1
    with pytest.raises(ValueError):   
        obj.x2
    with pytest.raises(ValueError):   
        obj.y2
    with pytest.raises(ValueError):   
        obj.vx1
    with pytest.raises(ValueError):   
        obj.vy1
    with pytest.raises(ValueError):   
        obj.vx2
    with pytest.raises(ValueError):   
        obj.vy2
        
def test_DoublePendulum_null_mass():
    obj = DoublePendulum(M1=1e-15, M2=1e-15)
    y0 = (1 ,1 ,1 , 1)
    obj.solve(y0, 5, 1e-4)
    K = obj.kinetic
    P = obj.potential
    eps = 1e-13
    print(max(K), max(P))
    assert all(abs(K) < eps) and all(abs(P) < eps)



if __name__ == "__main__":
    test_DoublePendulum_zeroes_as_y0()
    test_DoublePendulum_properties_not_implemented()
    test_DoublePendulum_null_mass()



























































































































































































