import numpy as np
from scipy.integrate import quad
import scipy.optimize as opt
import matplotlib.pyplot as plt

# Set the font size for all text elements in the plot
plt.rc('font', size=12)  # Adjust the font size as needed
# Given values
points = 1000
V_b_solution_mod_solid = []
V_b_solution_mod_ring = []

# Constants for argon at 20°C and 1 atm
A_argon = 12  # (cm.Torr)^-1
B_argon = 180  # V*(cm.Torr)^-1
gamma_se = 0.06
constant_distance = 8.0  # Constant electrode separation distance in meters
pressure_range_mod = np.logspace(-1.52, 4, points)  # Specify the pressure range for modified Paschen curve (in torr)

# Specify the path to your ASCII file
file_path = 'electric_field/data_solid_8.txt'
file_path_2 = 'electric_field/data_ring_inner_1.0e-3_8.txt'
# Use numpy.genfromtxt to read the file and create NumPy arrays
# In this case, we specify a fixed delimiter of whitespace and skip the header lines.
data = np.genfromtxt(file_path, delimiter=None, skip_header=2)
data2 = np.genfromtxt(file_path_2, delimiter=None, skip_header=2)

# Now 'data' is a NumPy array containing the two arrays
# Split the data into two separate NumPy arrays
z = data[:, 0]  # Assuming the first column is the first array
Ez = data[:, 1]  # Assuming the second column is the second array
Ez_ring = data2[:, 1]  # Assuming the second column is the second array
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

print("Normalized Electric field for ring")
norm_Ez_ring = Ez_ring/(V*100/d)
print(norm_Ez_ring)


# Choose the degree of the polynomial fit (e.g., 2 for a quadratic fit)
degree = 10


# Generate x values for the fit curve
z_fit = np.linspace(0, z.max()-z.min(), points)
x = (z-z.min())/(z.max()-z.min())
x_fit = z_fit/(z.max()-z.min())

# Perform the polynomial fit
coefficients = np.polyfit(x, norm_Ez, degree)

# Perform the polynomial fit for ring
coefficients_ring = np.polyfit(x, norm_Ez_ring, degree)

# Create a polynomial using the coefficients
poly = np.poly1d(coefficients)


# Create a polynomial using the coefficients
poly_ring = np.poly1d(coefficients_ring)

# Compute corresponding y values for the fit curve
norm_Ez_fit = poly(x_fit)
norm_Ez_ring_fit = poly(x_fit)

# Define the function to integrate for modified Paschen curve
def integrand_mod(x, V_b, p, d):
    return np.exp(-B_argon * p * d / (V_b*(1+np.cos(x))))

# Define the function to integrate for modified Paschen curve
def integrand_mod_solid(x, V_b, p, d):
    return np.exp(-B_argon * p * d / (V_b*poly(x)))

# Define the function to integrate for modified Paschen curve for ring
def integrand_mod_ring(x, V_b, p, d):
    return np.exp(-B_argon * p * d / (V_b*poly_ring(x)))

# Define the equation to solve for V_b
def equation_to_solve_mod(V_b, p, d,func):
    result, _ = quad(func, 0, 1, args=(V_b, p, d),points=1000)
    return np.log(result) - np.log(np.log(1 + 1/gamma_se)) + np.log(A_argon*p*d)

# Solve for V_b for each pressure in the range for modified Paschen curve
for i, pressure in enumerate(pressure_range_mod):
    # Perform the optimization for modified Paschen curve
    V_b_solution_mod_solid.append(opt.root_scalar(equation_to_solve_mod, args=(pressure, constant_distance,integrand_mod_solid), bracket=[1e0, 1e8]).root)
    V_b_solution_mod_ring.append(opt.root_scalar(equation_to_solve_mod, args=(pressure, constant_distance, integrand_mod_ring),bracket=[1e0, 1e8]).root)
    if (i%100==0):
        print(str(int(i*100/points)) + '%')
# Calculate the original Paschen curve
discharge_voltage_original = B_argon * pressure_range_mod*constant_distance / (np.log(A_argon * pressure_range_mod*constant_distance) - np.log(np.log(1 + 1/gamma_se)))

# Plot both curves on the same plot
plt.figure(figsize=(8, 6))
plt.loglog(pressure_range_mod, V_b_solution_mod_ring, label='Modified Paschen Curve for ring electrode (1-1.5 mm)')
plt.loglog(pressure_range_mod, V_b_solution_mod_solid, label='Modified Paschen Curve for solid electrode')
plt.loglog(pressure_range_mod, discharge_voltage_original, label='Paschen Curve (assumes constant E)', color='r')
plt.xlabel('Pressure (torr)')
plt.ylabel('Discharge Voltage (V)')
plt.title(f'Paschen Curves (d = 8 cm, r = 1.5 mm, d/r = {d/r:.2f})')
plt.grid(True)
plt.legend()
plt.savefig('paschen_curve_total.png')
