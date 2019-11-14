import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy.integrate import solve_ivp


class DoublePendulum:

    def __init__(self, 𝑀1=1, 𝐿1=1, 𝑀2=1, 𝐿2=1, g=9.81):
        self.M1 = M1
        self.M2 = M2
        self.L1 = L1
        self.L2 = L2
        self.g = g

    def __call__(self, t, y):
        '''
            Params:
            ----------
            y: tuple, list
                𝑦=(𝜃1,𝜔1,𝜃2,𝜔2)
            t: int
                time point

            Returns:
            -----------
            RHS: tuple
                the right hand side of the ODE
        '''
        theta1, omega1, theta2, omega2 = y

        M1, M2 = self.M1, self.M2
        L1, L2 = self.L1, self.L2
        g = self.g

        dtheta1dt = omega1
        dtheta2dt = omega2

        del_theta = theta2 - theta1
        domega1dt = ((M2*L1*omega1**2*np.sin(del_theta)*np.cos(del_theta) +
                      M2*g*np.sin(theta2)*np.cos(del_theta) +
                      M2*L2*omega2**2*np.sin(del_theta) -
                      (M1 + M2)*g*np.sin(theta1)) /
                     ((M1 + M2)*L2 - M2*L2*np.cos(del_theta)**2))
        domega2dt = ((-M2*L2*omega2**2*np.sin(del_theta)*np.cos(del_theta) +
                      (M1 + M2)*g*np.sin(theta1)*np.cos(del_theta) -
                      (M1 + M2)*L1*omega1**2*np.sin(del_theta) -
                      (M1 + M2)*g*np.sin(theta2)) /
                     ((M1+M2)*L2 - M2*L2*np.cos(del_theta)**2))

        RHS = (dtheta1dt, domega1dt, dtheta2dt, domega2dt)
        return RHS

    @property
    def t(self):
        if hasattr(self, '_t'):
            return self._t
        else:
            raise ValueError('t is not calculated, run solve')

    @t.setter
    def t(self, t):
        self._t = t

    @property
    def theta1(self):
        if hasattr(self, "_theta1"):
            return self._theta1
        else:
            raise ValueError('theta1 is not calculated, run solve')

    @theta1.setter
    def theta1(self, theta1):
        self._theta1 = theta1

    @property
    def omega1(self):
        if hasattr(self, "_omega1"):
            return self._omega1
        else:
            raise ValueError('omega1 is not calculated, run solve')

    @omega1.setter
    def omega1(self, omega1):
        self._omega1 = omega1

    @property
    def theta2(self):
        if hasattr(self, "_theta2"):
            return self._theta2
        else:
            raise ValueError('theta2 is not calculated, run solve')

    @theta2.setter
    def theta2(self, theta2):
        self._theta2 = theta2

    @property
    def omega2(self):
        if hasattr(self, "_omega2"):
            return self._omega2
        else:
            raise ValueError('omega2 is not calculated, run solve')

    @omega2.setter
    def omega2(self, omega2):
        self._omega2 = omega2

    @property
    def y(self):
        return self.theta1, self.omega2, self.theta2, self.omega2

    @property
    def x1(self):
        return self.L1*np.sin(self.theta1)

    @property
    def y1(self):
        return -self.L1*np.cos(self.theta1)

    @property
    def x2(self):
        return self.x1 + self.L2*np.sin(self.theta2)

    @property
    def y2(self):
        return self.y1 - self.L2*np.cos(self.theta2)

    @property
    def potential(self):
        P1 = self.M1*self.g*(self.y1 + self.L1)
        P2 = self.M2*self.g*(self.y2 + self.L1 + self.L2)
        return P1+P2

    @property
    def vx1(self):
        return np.gradient(self.x1, self.t)
        # raise NotImplementedError

    @property
    def vy1(self):
        return np.gradient(self.y1, self.t)
        # raise NotImplementedError

    @property
    def vx2(self):
        return np.gradient(self.x2, self.t)
        # raise NotImplementedError

    @property
    def vy2(self):
        return np.gradient(self.y2, self.t)
        # raise NotImplementedError

    @property
    def kinetic(self):
        K1 = 1/2 * self.M1 * (self.vx1**2 + self.vy1**2)
        K2 = 1/2 * self.M2 * (self.vx2**2 + self.vy2**2)
        return K1 + K2

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

        t = np.arange(dt, T, dt)

        sol = solve_ivp(self, [dt, T], y0, t_eval=t, method="Radau")
        self.theta1 = sol.y[0]
        self.omega1 = sol.y[1]
        self.theta2 = sol.y[2]
        self.omega2 = sol.y[3]
        self.t = sol.t
        self.dt = dt

    def _deg_to_rad(self, deg):
        return np.radians(deg)

    def create_animation(self):
        # Create empty figure
        fig = plt.figure(figsize=(6.4*3, 4.8*3))

        # Configure figure
        plt.axis('equal')
        # plt.axis('off')
        plt.axis((-3, 3, -3, 3))

        # Make an "empty" plot object to be updated throughout the animation
        self.pendulums, = plt.plot([], [], 'o-', lw=2)

        # Call FuncAnimation
        self.animation = animation.FuncAnimation(fig,
                                                 self._next_frame,
                                                 frames=range(len(self.x1)),
                                                 repeat=None,
                                                 interval=1000*self.dt,
                                                 blit=True)

    def _next_frame(self, i):
        self.pendulums.set_data((0, self.x1[i], self.x2[i]),
                                (0, self.y1[i], self.y2[i]))
        return self.pendulums,

    def show_animation(self):
        plt.show()

    def save_animation(self, fname, fps=60):
        self.animation.save(f"fig/{fname}", fps=fps)


if __name__ == "__main__":

    y0 = (np.pi/6, 0.15, np.pi/6, 0.15)
    obj = DoublePendulum()
    obj.solve(y0, 5, 0.0001)
    t = obj.t
    theta1, omega1, theta2, omega2 = obj.theta1, obj.omega2, obj.theta2, obj.omega2

    # plott:
    fig, axs = plt.subplots(3, 1, figsize=(6.4*5, 4.8*5))
    # kinetic
    K = obj.kinetic
    axs[0].plot(t, K, label="kinetic energy")
    axs[0].legend()
    # potential
    P = obj.potential
    axs[1].plot(t, P, label="potential energy")
    axs[1].legend()
    # total
    total = P + K
    axs[2].plot(t, total, label="total energy")
    axs[2].legend()

    fig.savefig("fig/double_pendulum.jpg", quality=95, dpi=100)
    fig.savefig("fig/double_pendulum.png",
                quality=95, dpi=100, transparent=True)
    plt.show()

    # Animation:
    y0 = (np.pi, 0.15, np.pi/6, 0.15)  # y = (theta1, omega1, theta2, omega2)
    obj = DoublePendulum(M1=10, L2=0.7/1.5, M2=1, L1=1/1.5)
    T, dt = 30, 1/60
    obj.solve(y0, T, dt)
    obj.create_animation()
    obj.save_animation('example_simulation.mp4')
    obj.show_animation()
