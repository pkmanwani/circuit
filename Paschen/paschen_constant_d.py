import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Specify the path to your ASCII file
file_path = 'electric_field/data_solid_8.txt'
# Use numpy.genfromtxt to read the file and create NumPy arrays
# In this case, we specify a fixed delimiter of whitespace and skip the header lines.
data = np.genfromtxt(file_path, delimiter=None, skip_header=2)

# Now 'data' is a NumPy array containing the two arrays
# Split the data into two separate NumPy arrays
z = data[:, 0]  # Assuming the first column is the first array
Ez = data[:, 1]  # Assuming the second column is the second array
V = 8e3 #volt
d = 8 #cm
r = 1.5e-1 #cm
d_r = d/r
# Now 'array1' and 'array2' contain the two separate arrays
print("z:")
print(z)
print("Electric field:")
print(Ez)
print("Normalized Electric field")
norm_Ez = Ez/(V*100/d)
print(norm_Ez)

# Choose the degree of the polynomial fit (e.g., 2 for a quadratic fit)
degree = 10



# Generate x values for the fit curve
z_fit = np.linspace(z.min(), z.max(), 100)
x = z/(z.max()-z.min())
x_fit = z_fit/(z.max()-z.min())

# Perform the polynomial fit
coefficients = np.polyfit(x, norm_Ez, degree)

# Create a polynomial using the coefficients
poly = np.poly1d(coefficients)

# Compute corresponding y values for the fit curve
norm_Ez_fit = poly(x_fit)

# Plot the original data points and the polynomial fit
#plt.figure(figsize=(8, 6))
plt.scatter(x, norm_Ez, label='Data Points', color='b')
plt.plot(x_fit, norm_Ez_fit, label=f'Polynomial Fit (Degree {degree})', color='r')
#plt.xlabel('z')
#plt.ylabel('Electric_field ')
#plt.legend()
#plt.grid(True)
plt.show()

print(f'd/r = {d_r}')
# Constants for argon at 20°C and 1 atm
pressure_range = np.logspace(-4,3, 10000)  # Pressure in torr
constant_distance = 8.0  # Constant electrode separation distance in centimeters
discharge_voltage = np.zeros(len(pressure_range))

# Paschen curve constants for argon
A_argon = 12  # (cm.Torr)^-1
B_argon = 180  # V*(cm.Torr)^-1
gamma_se = 0.06

def smooth_log(x, small_value=1e-10):
    return np.log(x + small_value)

for i, p in enumerate(pressure_range):
    # Modified Paschen curve equation for argon with constant distance and gamma_se dependence
    discharge_voltage[i] = B_argon * p*constant_distance / (smooth_log(A_argon * p*constant_distance) - smooth_log(smooth_log(1 + 1/gamma_se)))

# Filter out negative values and shorten the arrays
positive_indices = discharge_voltage > 0
pressure_range = pressure_range[positive_indices]
discharge_voltage = discharge_voltage[positive_indices]
print(discharge_voltage)
plt.figure(figsize=(8, 6))
plt.loglog(pressure_range, discharge_voltage, label='Voltage vs Pressure', color='b')
plt.xlabel('p (torr)')
plt.ylabel('Discharge Voltage (V)')
plt.title(f'Paschen Curve')
plt.grid(True)
plt.legend()
plt.savefig('paschen_curve.png')