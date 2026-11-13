#Modules
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

#Data

#Resistor data
r_1 = 10000 #ohm
r_2 = 1000 #ohm
r_3 =330 #ohm

v_source = np.array([0.05,0.10,0.14,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.49,2.99,3.49,3.99,4.49,4.99,5.49,5.98]) #V
v_res = np.array([0.007,0.017,0.026,0.070,0.315,1.16,3.86,11.57,28.92,57.5,95.93,184.2,286,393,502,614,727,842,957])*1e-3 #V
v_after = r_3/(r_3+r_2)*v_source

#Diode data
vol_dio = v_after-v_res
cur_dio = v_res / r_1

#Plotting
plt.figure()
plt.grid()
plt.yscale("log")
plt.scatter(vol_dio,cur_dio)
plt.ylabel("Log current from the diode $(A)$")
plt.xlabel("Voltage on the diode $(V)$")
plt.show()