import numpy as np
from numba import int32, float64
from numba.experimental import jitclass

spec = [
    ('tau', float64),
    ('int_steps', int32),
    ('h', float64),
    ('sigma', float64),
    ('beta', float64),
    ('rho', float64),
    ('state', float64[:]),
]


@jitclass(spec)
class LorenzModel(object):
    def __init__(self, tau=0.1, int_steps=10, sigma=10.,
                 beta=8 / 3, rho=28., ic=np.array([]), ic_seed=0):
        self.tau = tau
        self.int_steps = int_steps
        self.h = tau / int_steps
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        if ic.size == 0:
            np.random.seed(ic_seed)
            self.state = (np.random.rand(3) * 2 - 1) * np.array([1., 1., 30.])
        elif ic.size == 3:
            self.state = ic.flatten()
        else:
            raise ValueError

    def run(self, T, discard_len=0):
        model_output = np.zeros((T + discard_len + 1, 3))
        model_output[0] = self.state
        for i in range(T + discard_len):
            model_output[i + 1] = self.forward()

        return model_output[discard_len:]

    def forward(self):
        for i in range(self.int_steps):
            self.state = self.rk4()
        return self.state

    def rk4(self):
        # Fourth order Runge-Kutta integrator
        x = self.state
        k1 = self.dxdt(x)
        k2 = self.dxdt(x + k1 / 2 * self.h)
        k3 = self.dxdt(x + k2 / 2 * self.h)
        k4 = self.dxdt(x + self.h * k3)

        xnext = x + 1 / 6 * self.h * (k1 + 2 * k2 + 2 * k3 + k4)
        return xnext

    def dxdt(self, x):
        return np.array([self.sigma * (- x[0] + x[1]),
                         self.rho * x[0] - x[1] - x[0] * x[2],
                         x[0] * x[1] - self.beta * x[2]])


spec.extend([('time', float64), ('period', float64)])
@jitclass(spec)
class LorenzModelPeriodicRho(object):
    def __init__(self, tau=0.1, int_steps=10, time=0., period=100., sigma=10.,
                 beta=8 / 3, rho=28., ic=np.array([]), ic_seed=0):
        self.tau = tau
        self.int_steps = int_steps
        self.h = tau / int_steps
        self.time = time
        self.period = period
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        if ic.size == 0:
            np.random.seed(ic_seed)
            self.state = (np.random.rand(3) * 2 - 1) * np.array([1., 1., 30.])
        elif ic.size == 3:
            self.state = ic.flatten()
        else:
            raise ValueError

    def run(self, T, discard_len=0):
        model_output = np.zeros((T + discard_len + 1, 3))
        model_output[0] = self.state
        for i in range(T + discard_len):
            model_output[i + 1] = self.forward()

        return model_output[discard_len:]

    def forward(self):
        for i in range(self.int_steps):
            self.state, self.time = self.rk4()
        return self.state

    def rk4(self):
        # Fourth order Runge-Kutta integrator
        x = self.state
        k1 = self.dxdt(x, self.time)
        k2 = self.dxdt(x + k1 / 2 * self.h, self.time + self.h / 2)
        k3 = self.dxdt(x + k2 / 2 * self.h, self.time + self.h / 2)
        k4 = self.dxdt(x + self.h * k3, self.time + self.h)

        xnext = x + 1 / 6 * self.h * (k1 + 2 * k2 + 2 * k3 + k4)
        return xnext, self.time + self.h

    def dxdt(self, x, t):
        self.rhofun(t)
        return np.array([self.sigma * (- x[0] + x[1]),
                         self.rho * x[0] - x[1] - x[0] * x[2],
                         x[0] * x[1] - self.beta * x[2]])

    def rhofun(self, t):
        self.rho = 10 * np.sin(2 * np.pi * t / self.period + 1.5 * np.pi) + 38