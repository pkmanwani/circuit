import numpy as np
from scipy.integrate import quad
import scipy.optimize as opt
import matplotlib.pyplot as plt
# Constants for argon at 20°C and 1 atm
A_argon = 12  # (cm.Torr)^-1
B_argon = 180  # V*(cm.Torr)^-1
gamma_se = 0.06

# Given values
constant_distance = 8.0  # Constant electrode separation distance in meters
pressure_range = np.logspace(0,4, 10000) # Specify the pressure range (in torr)
V_b_solution = []

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
z_fit = np.linspace(0, z.max()-z.min(), 100)
x = (z-z.min())/(z.max()-z.min())
x_fit = z_fit/(z.max()-z.min())

# Perform the polynomial fit
coefficients = np.polyfit(x, norm_Ez, degree)

# Create a polynomial using the coefficients
poly = np.poly1d(coefficients)

# Compute corresponding y values for the fit curve
norm_Ez_fit = poly(x_fit)

# Plot the original data points and the polynomial fit
#plt.figure(figsize=(8, 6))
#plt.scatter(x, norm_Ez, label='Data Points', color='b')
#plt.plot(x_fit, norm_Ez_fit, label=f'Polynomial Fit (Degree {degree})', color='r')
#plt.xlabel('z')
#plt.ylabel('Electric_field ')
#plt.legend()
#plt.grid(True)
#plt.show()
# Define the function to integrate
def integrand(x, V_b, p, d):
    return np.exp(-B_argon * p * d / (V_b*(1+np.cos(x))))

# Solve for V_b for each pressure in the range
for pressure in pressure_range:
    # Define the equation to solve for V_b
    def equation_to_solve(V_b, p, d):
        result, _ = quad(integrand, 0, d, args=(V_b, p, d))
        return result - np.log(1 + 1/gamma_se) / (A_argon*d)
    #print(pressure)
    # Perform the optimization
    V_b_solution.append(opt.root_scalar(equation_to_solve, args=(pressure, constant_distance), bracket=[1e-6, 1e8]).root)

plt.loglog(pressure_range,V_b_solution)
plt.show()