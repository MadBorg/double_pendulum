
import numpy as np
from scipy.integrate import solve_ivp


class Pendulum:
    def __init__(self,  L=1, M=1, g=9.81):
        '''
        Params:
        ------------
        L: int, float
            Lenght of the rod
            physical unit: meter: m
        M: int, float
            the mass of the pendulum
            physical unit: kilo grams: kg
        g: int, float
            gravitational acceleration, 9.81 standard on earth
            physical unit: meters/second**2: m/s**2
        '''
        self.L, self.M, self.g = L, M, g

    def __call__(self, t, y):
        '''
            Params:
            ------------
            t: int, float
                time point
            y: list, array, tuple
                ODE's

            Return:
            ------------
            RHS: tuple
        '''
        theta, omega = y
        dtheta = omega
        domega = -(self.g/self.L)*np.sin(theta)
        RHS = (dtheta, domega)
        return RHS

    @property
    def t(self):
        '''
        t: An array of the time mesh points  𝑡𝑖=𝑖Δ𝑡 .
        '''
        if hasattr(self, '_t'):
            return self._t
        else:
            raise Exception('t not calculated, run solve')

    @t.setter
    def t(self, t):
        self._t = t

    @property
    def theta(self):
        '''
        theta: An array of the pendulums position  𝜃𝑖=𝜃(𝑡𝑖) .
        '''
        if hasattr(self, '_theta'):
            return self._theta
        else:
            raise Exception('theta not calculated, run solve')

    @theta.setter
    def theta(self, theta):
        self._theta = theta

    @property
    def omega(self):
        '''
        omega: An array of the pendulums velocity  𝜔𝑖=𝜔(𝑡𝑖) .
        '''
        if hasattr(self, '_omega'):
            return self._omega
        else:
            raise Exception('omega not calculated, run solve')

    @omega.setter
    def omega(self, omega):
        self._omega = omega

    @property
    def x(self):
        return self.L * np.sin(self.theta)

    @property
    def y(self):
        return - self.L*np.cos(self.theta)

    @property
    def potential(self):
        return self.M*self.g*(self.y+self.L)

    @property
    def vx(self):
        return np.gradient(self.x, self.t)
        # raise NotImplementedError

    @property
    def vy(self):
        return np.gradient(self.y, self.t)
        # raise NotImplementedError

    @property
    def kinetic(self):
        M, norm = self.M, self._norm
        return (1/2)*M*(self.vx**2 + self.vy**2)
        # raise NotImplementedError

    @property
    def _norm(self):
        return np.linalg.norm((self.vx, self.vy))

    def solve(self, y0, T, dt, angles='rad'):
        '''
            method that uses solve_ivp to solve the equations of motions on the range  𝑡∈(0,𝑇]

            Params:
            ----------
            y0: tuple, list: int, float
                start values
            T: int, float
                stop time point
            dt: int, float
                delta time value
            angles: str
            choose coordinates between radians(deg) and radians(rad)

        '''
        if angles == 'deg':
            y0 = self._deg_to_rad(y0)

        # self.t = np.arange(dt, T, dt)
        t = np.arange(dt, T, dt)

        sol = solve_ivp(self, [dt, T], y0, t_eval=t)
        self.theta = sol.y[0]
        self.omega = sol.y[1]
        self.t = sol.t
        # print(self.theta, self.omega, self.t)
        # print(y)

    def _deg_to_rad(self, deg):
        return np.radians(deg)


class DampenedPendulum(Pendulum):
    def __init__(self,B, L=1, M=1, g=9.81):
        super().__init__(L=L, M=M, g=g)
        self.B = B

    def __call__(self, t, y):
        '''
            Params:
            ------------
            t: int, float
                time point
            y: list, array, tuple
                ODE's
            Return:
            ------------
            RHS: tuple
        '''
        theta, omega = y
        dtheta = omega
        domega = -(self.g/self.L)*np.sin(theta) - (self.B/self.M)*omega
        RHS = (dtheta, domega)
        return RHS


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fig, axs =plt.subplots(2,1)

    # Exercise 2e) Example use¶
    L = 2.7
    omega0 = 0.15
    theta0 = np.pi/6
    pendulum = Pendulum(L=L)
    y0 = (theta0, omega0)
    dt, T = 0.11, 10
    pendulum.solve(y0, T, dt)
    theta, omega, t = pendulum.theta, pendulum.omega, pendulum.t
    # K = pendulum.kinetic
    K = [pendulum.kinetic for i in t]


    axs[0].set_title('Pendulum')
    axs[0].plot(t, theta, label= 'Theta')
    axs[0].plot(t, omega, label='Omega')
    # axs[0].plot(t, K, label="K")
    axs[0].legend()

    # Exercise 2f) A Dampened Pendulum
    B = 0.5
    L = 2.7
    omega0 = 0.15
    theta0 = np.pi/6
    dampened_pendulum = DampenedPendulum(B=B ,L=L)
    y0 = (theta0, omega0)
    dt, T = 0.11, 10
    dampened_pendulum.solve(y0, T, dt)
    theta, omega, t = dampened_pendulum.theta, dampened_pendulum.omega, dampened_pendulum.t
    # K = pendulum.kinetic
    K = [dampened_pendulum.kinetic for i in t]


    axs[1].set_title("DampenedPendulum")
    axs[1].plot(dampened_pendulum.t, dampened_pendulum.theta, label= 'Theta')
    axs[1].plot(dampened_pendulum.t, dampened_pendulum.omega, label='Omega')
    # axs[1].plot(dampened_pendulum.t, K, label="K")
    axs[1].legend()

    
    plt.show()
    
