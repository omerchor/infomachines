import matplotlib

import scipy.integrate
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets

matplotlib.use('TKAgg')



def ideal_gas_expansion(t, z, b, c):
    """
    Differential equation for expansion of ideal gas with Newton's second law

    Parameters
    ----------
    t: time
    z: tuple of [dx/dt, x]
    b: value of (N k_B T)/m (right hand side of equation of state divided by barrier mass)
    c: friction coefficient gamma, divided by mass (gamma/m)

    Returns
    -------
    Tuple with value of dz/dt, i.e. (x'', x')
    """
    return b/z[1] - c*z[0], z[0]


@widgets.interact(m=widgets.FloatSlider(2, min=0.1, max=5),
                  nkt=widgets.FloatSlider(1, min=0.1, max=5),
                  gamma=widgets.FloatSlider(1, min=0.1, max=5),
                  y0=widgets.FloatSlider(0.5, min=0.1, max=5))
def int_plot(m=1, nkt=1, gamma=1, y0=0.5):
    time_range = [0, 5]
    args = (nkt/m, gamma/m)
    t = np.linspace(*time_range, 300)
    sol = scipy.integrate.solve_ivp(ideal_gas_expansion, time_range, [0, y0], args=args, dense_output=True)
    z = sol.sol(t)
    plt.plot(t, z.T)
    plt.plot(t, args[0]/z.T[:,1], label="Pressure")
    plt.xlabel("t [sec]")
    plt.legend(['dx/dt', 'x', "Pressure"])
    plt.title("Expansion of ideal gas against friction")
    return plt.figure()


if __name__ == "__main__":
    int_plot()
