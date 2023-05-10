import numpy as np

class FDTD_Maxwell_1D():
    def __init__(self, L=10, CFL=1.0, Nx=101):
        self.x = np.linspace(0, L, num=Nx)
        self.d = np.abs(self.x[-1] - self.x[0])
        self.xDual = (self.x[1:] + self.x[:-1]) / 2

        self.eps_0 = 1.0
        self.eps_inf = 1.1
        self.mu_0 = 1.0
        self.eta_0 = np.sqrt(self.mu_0 / self.eps_0)
        self.c_0 = 1 / np.sqrt(self.eps_0 * self.mu_0)

        self.a_p = 0.2 + 0j
        self.c_p = 1 + 1j

        self.dx = self.x[1] - self.x[0]
        self.dt = CFL * self.dx / self.c_0

        self.e = np.zeros(self.x.shape)
        self.J = np.zeros(self.x.shape)
        self.h = np.zeros(self.xDual.shape)
        self.Epsilon = np.zeros(self.x.shape)

    def get_a_p(self):
        return self.get_a_p()

    def get_c_p(self):
        return self.get_c_p()

    def step(self):
        e = self.e
        h = self.h

        cE = -self.dt / self.dx / self.eps_0
        cH = -self.dt / self.dx / self.mu_0

        # eMur = e[1]
        e[1:-1] = cE * (h[1:] - h[:-1]) + e[1:-1]

        # Lado izquierdo
        e[0] = 0.0  # PEC
        # e[0] = e[0] - 2* dt/dx/eps*h[0]                  # PMC
        # e[0] =  (-dt / dx / eps) * (h[0] - h[-1]) + e[0] # Periodica
        # e[0] = eMur + (c0*self.dt-self.dx)/(c0*self.dt+self.dx)*(e[1]-e[0]) # Mur

        # Lado derecho
        e[-1] = 0.0
        # e[-1] = e[0]

        h[:] = cH * (e[1:] - e[:-1]) + h[:]

    def theoretical_test(self, w):
        eps_c = self.eps_0 * self.eps_inf + self.eps_0 * self.c_0 / (w * 1j - self.a_p) + np.conj(self.c_0) / (
                    w * 1j - np.conj(self.a_p))
        eps_c = np.abs(eps_c)
        return eps_c
