# region ###########################  Modules #####################
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

# endregion ###########################  Modules #####################

# region ###########################  Constants #####################
q = -1.602e-19  # C - The charge of an elektron
n = 2.0  # +-0.1 - The idealisation factor, not quite sure what this is as i can't find any outside references for it
T = 292  # +- 2 K - The temperature is assumed to be room temperature in kelvin
KB = 1.380649e-23 # J*K^-1 - The boltzmann constant per Meet the Constants | NIST. (2026, September 10). NIST. https://www.nist.gov/si-redefinition/meet-constants - we will be using this to measure the variance of our calculated Boltzmann with the actuall constant.

# endregion ###########################  Constants #####################

# region ###########################  Data #####################
r_1 = 10000  # +- ohm
r_2 = 1000  # ohm
r_3 = 330  # ohm

v_source = np.array(
    [
        0.05,
        0.10,
        0.14,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.49,
        2.99,
        3.49,
        3.99,
        4.49,
        4.99,
        5.49,
        5.98,
    ]
)  # V
v_res = (
    np.array(
        [
            0.007,
            0.017,
            0.026,
            0.070,
            0.315,
            1.16,
            3.86,
            11.57,
            28.92,
            57.5,
            95.93,
            184.2,
            286,
            393,
            502,
            614,
            727,
            842,
            957,
        ]
    )
    * 1e-3
)  # V
v_after = r_3 / (r_3 + r_2) * v_source

vol_dio = v_after - v_res
cur_dio = v_res / r_1
# endregion ###########################  Data #####################


# region ###########################  Model #####################
def model(Voltage, c1, c2):
    """
    ---Model for the current leaked from a diode---

    Additional notes: Use this only when the voltage is greater than 0.1V, otherwise you get weird values that dont fit
    This is also using a log scale; in algebra, the model is ln(I_D) = ln(I_0)+ -q/n*K_b*T*V where the current and the boltzman constant are parameters that must be fitted, q is the charge of an electron (it's already negative so really it becomes + |q|), T is the temperature of the diode, n is the idealisation factor and V_d is the voltage on the resistor attached to the diode (This becomes your argument in this model).
    """
    return c1 + (-q / (n * c2 * T)) * Voltage


popt, pcov = curve_fit(
    model,
    vol_dio[vol_dio > 0.1],
    np.log(cur_dio[vol_dio > 0.1]),
    p0=(-20, 1.3806503e-23),
)
print(f"Boltzmann according to our fit ({popt[1]:.7} ± {np.sqrt(pcov.diagonal()[1]):.7})")
# Here we are using a statistical method called the t-value to compare the constant from our data and the constant's definition
# As a rule of thumb: 0<t<1 is perfect since our difference falls within our uncertainty and thus we can't tell if the values are actually different since our tools are not accurate enough to tell, this is practically the best we can hope for barring zero uncertainty which is impossible; 1<t<2 is more tricky, could be a really weird spread of data or the values are actually different, executing the experiment again would be the best course of action; 2<t<3: we can be reasonably sure that the values are actually different (though not 100%, you can never be 100% sure of anything).
t = np.abs((popt[1]-KB)/(np.sqrt(pcov.diagonal()[1])))
print(f"Our t-value: {t:.2f}")
# endregion ###########################  Model #####################

# region ###########################  Plotting #####################
plt.figure()
plt.grid()
plt.scatter(vol_dio, np.log(cur_dio))
plt.plot(vol_dio[vol_dio > 0.1], model(vol_dio[vol_dio > 0.1], popt[0], popt[1]))
plt.ylabel("Ln of the current from the diode $(A)$")
plt.xlabel("Voltage on the diode $(V)$")
plt.show()
# endregion ###########################  Plotting #####################
