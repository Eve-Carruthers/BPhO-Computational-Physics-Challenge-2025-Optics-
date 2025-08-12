import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import Model, RealData, ODR

# data
u_values = np.array([20, 25, 30, 35, 40, 45, 50, 55])  # object distances
v_values = np.array([60.5, 40, 31, 27, 25, 23.1, 21.5, 20.5])  # image distances

# Uncertainties
u_unc = np.full_like(u_values, 1.0)
v_unc = np.full_like(v_values, 1.0)

# Calculate 1/u and 1/v
inv_u = 1 / u_values
inv_v = 1 / v_values

# Propagate uncertainties: if y = 1/x, then σ_y = σ_x / x^2
inv_u_unc = u_unc / (u_values ** 2)
inv_v_unc = v_unc / (v_values ** 2)

# Define linear model for ODR: y = m*x + c
def linear_model(B, x):
    return B[0] * x + B[1]

# Set up ODR
model = Model(linear_model)
data = RealData(inv_u, inv_v, sx=inv_u_unc, sy=inv_v_unc)
odr = ODR(data, model, beta0=[-1, 0.05])  # Initial guess: slope -1, intercept 0.05
out = odr.run()

# Extract results
slope, intercept = out.beta
slope_err, intercept_err = out.sd_beta
focal_length = 1 / intercept
focal_length_unc = intercept_err / (intercept ** 2)  # Propagate uncertainty

# Print results
print(f"Slope: {slope:.4f} ± {slope_err:.4f}")
print(f"Intercept (1/f): {intercept:.4f} ± {intercept_err:.4f}")
print(f"Focal length f: {focal_length:.2f} ± {focal_length_unc:.2f} cm")

# Plot with error bars
plt.errorbar(inv_u, inv_v, xerr=inv_u_unc, yerr=inv_v_unc,
             fmt='o', label='Data with error bars', ecolor='gray', capsize=3)

# Plot best fit line
x_fit = np.linspace(min(inv_u), max(inv_u), 100)
y_fit = linear_model([slope, intercept], x_fit)
plt.plot(x_fit, y_fit, color='red', label='Best Fit Line (ODR)')

plt.xlabel('1/u (1/cm)')
plt.ylabel('1/v (1/cm)')
plt.title('Thin Lens Equation with Uncertainties')
plt.legend()
plt.grid(True)
plt.show()
