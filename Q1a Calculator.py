
import numpy as np

# Sellmeier coefficients for BK7 glass
B1 = 1.03961212
B2 = 0.231792344
B3 = 1.01046945
C1 = 0.00600069867
C2 = 0.0200179144
C3 = 103.560653


def calculate_refractive_index(wavelength_nm):
    """
    Calculate the refractive index of BK7 glass for a given wavelength.

    Parameters:
    wavelength_nm (float): Wavelength in nanometers.

    Returns:
    float: Refractive index.
    """
    # Convert wavelength from nm to µm
    wavelength_um = wavelength_nm / 1000
    print(f"Wavelength: {wavelength_nm} nm = {wavelength_um:.6f} µm")

    # Compute λ²
    lambda_squared = wavelength_um ** 2
    print(f"λ² = {lambda_squared:.6f} µm²")

    # Compute each term of the Sellmeier equation
    term1 = (B1 * lambda_squared) / (lambda_squared - C1)
    term2 = (B2 * lambda_squared) / (lambda_squared - C2)
    term3 = (B3 * lambda_squared) / (lambda_squared - C3)

    print(f"Term 1: {term1:.6f}")
    print(f"Term 2: {term2:.6f}")
    print(f"Term 3: {term3:.6f}")

    # Sum the terms to get n²
    n_squared = 1 + term1 + term2 + term3
    print(f"n² = {n_squared:.6f}")

    # Calculate the refractive index
    refractive_index = np.sqrt(n_squared)
    print(f"Refractive Index (n) = {refractive_index:.6f}")

    return refractive_index


# Example usage:
wavelength_input = 375# Wavelength in nm
n = calculate_refractive_index(wavelength_input)
