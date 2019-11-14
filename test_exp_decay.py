#1a
from exp_decay import ExponentialDecay
def test_ExponentialDecay():
    dudt_expected = -1.28 #expected output
    a, u, t = 0.4, 3.2, None #parameters
    eps = 1e-14 #error margin
    dudt = ExponentialDecay(a)
    dudt_calculated = dudt(t, u)
    assert abs(dudt_expected - dudt_calculated) < eps, f'test_ExponentialDecay_unit: dudt_calculated:{dudt_calculated}, dudt_expected:{dudt_expected}, params: a:{a}, u:{u}'

def test_ExponentialDecay_false_positive():
    '''
    dudt_expected = 1.28, instead of -1.28
    '''
    dudt_expected = 1.28 #expected output
    a, u, t = 0.4, 3.2, None #parameters
    eps = 1e-14 #error margin
    dudt = ExponentialDecay(a)
    dudt_calculated = dudt(t, u)
    assert not (dudt_expected - dudt_calculated) < eps, f'test_ExponentialDecay_unit: dudt_calculated:{dudt_calculated}, dudt_expected:{dudt_expected}, params: a:{a}, u:{u}'

# def test_ExponentialDecay_solve():
#     decay_model = ExponentialDecay(a)
#     t, u = decay_model.solve(u0, T, dt)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    test_ExponentialDecay()
    test_ExponentialDecay_false_positive()

    # try
    # 1b)
    u0 = 3.2
    T = 10
    dt = 0.1
    a = 0.4

    decay_model = ExponentialDecay(a)
    t, u = decay_model.solve(u0, T, dt)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = plt.plot(t, u)
    fig.savefig('fig/Exercise1b_Solving_the_ODE_example.png')
    plt.show()
    # todo: prettify 


