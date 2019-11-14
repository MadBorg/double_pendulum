import numpy as np
from scipy.integrate import solve_ivp



class ExponentialDecay:
    def __init__(self, a):
        self.a = a

    def __call__(self, t, u):
        '''
        retunrs the RHS (right hand side) of the ODE.
        
        Params:
        ---------
        t: (None, int, flat)
            Only form compatibility atm.
        u: (int, float)
            parameter to be calculated for getting dudt
        '''
        return -self.a*u
    
    def solve(self, u0, T, dt):
        ''' solves the ODE with solve_ivp from cipy.integrate

        Params:
        ----------
        u0: (int, float)
            starting posistion for the ODE solver
        T: (int, float)
            ending posistion for the  ODE solver
        dt: (int, float)
            increments for the ODE solver

        Retuns:
        ----------
        t: (array)
            time points array
        u: (array)
            solutuoion points u(t_i)
        '''
        sol = solve_ivp(self, [dt, T], (u0,), t_eval=np.arange(dt,T,dt))
        # print(sol.t, sol.y[0])
        return sol.t, sol.y[0]

    
    
