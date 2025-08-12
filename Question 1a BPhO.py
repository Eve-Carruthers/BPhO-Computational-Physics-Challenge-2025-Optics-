#challenge 1a
import numpy as np
import plotly.graph_objects as go

# Sellmeier coefficients for BK7 glass
B1 = 1.03961212
B2 = 0.231792344
B3 = 1.01046945
C1 = 0.00600069867
C2 = 0.0200179144
C3 = 103.560653

def sellmeier(wavelength_um):
    lambda_sq = wavelength_um ** 2
    n_squared = 1 + (B1 * lambda_sq) / (lambda_sq - C1) + \
                   (B2 * lambda_sq) / (lambda_sq - C2) + \
                   (B3 * lambda_sq) / (lambda_sq - C3)
    return np.sqrt(n_squared)

# Generate wavelength data (400–800 nm)
wavelengths_nm = np.linspace(400, 800, 1000)
wavelengths_um = wavelengths_nm / 1000
refractive_indices = sellmeier(wavelengths_um)

# Color mapping
def wavelength_to_rgb(wl):
    # 380–780 nm
    wl = int(wl)
    if 380 <= wl <= 440:
        R = -(wl - 440) / (440 - 380)
        G = 0.0
        B = 1.0
    elif 440 <= wl <= 490:
        R = 0.0
        G = (wl - 440) / (490 - 440)
        B = 1.0
    elif 490 <= wl <= 510:
        R = 0.0
        G = 1.0
        B = -(wl - 510) / (510 - 490)
    elif 510 <= wl <= 580:
        R = (wl - 510) / (580 - 510)
        G = 1.0
        B = 0.0
    elif 580 <= wl <= 645:
        R = 1.0
        G = -(wl - 645) / (645 - 580)
        B = 0.0
    elif 645 <= wl <= 780:
        R = 1.0
        G = 0.0
        B = 0.0
    else:
        R = G = B = 0.0

    # Adjust intensity
    factor = 1.0
    if wl < 420:
        factor = 0.3 + 0.7 * (wl - 380) / (420 - 380)
    elif wl > 700:
        factor = 0.3 + 0.7 * (780 - wl) / (780 - 700)

    R = int(round(R * factor * 255))
    G = int(round(G * factor * 255))
    B = int(round(B * factor * 255))

    return f'rgb({R},{G},{B})'

colors = [wavelength_to_rgb(wl) for wl in wavelengths_nm]

# scatter plot
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=wavelengths_nm,
    y=refractive_indices,
    mode='markers',
    marker=dict(color=colors, size=5),
    hovertemplate=
        'Wavelength: %{x:.1f} nm<br>' +
        'Refractive Index: %{y:.6f}<extra></extra>'
))

fig.update_layout(
    title='Refractive Index of BK7 Glass vs. Wavelength',
    xaxis_title='Wavelength (nm)',
    yaxis_title='Refractive Index (n)',
    template='plotly_white',
    hovermode='closest',
    width=900,
    height=500
)

fig.show()
#challenge 1b

# Sellmeier formula constants for water
def refractive_index_water(frequency_hz):
    f_scaled = frequency_hz / 1e15
    inv_sq = 1.731 - 0.261 * f_scaled**2
    n_squared = 1 + 1 / np.sqrt(inv_sq)
    return np.sqrt(n_squared)

# Frequency range in THz
frequencies_thz = np.linspace(375, 750, 1000)
frequencies_hz = frequencies_thz * 1e12
refractive_indices = refractive_index_water(frequencies_hz)

# Convert to wavelengths
c = 3e8
wavelengths_nm = (c / frequencies_hz) * 1e9

# Colour mapping
def wavelength_to_rgb(wl):
    wl = int(wl)
    if 380 <= wl <= 440:
        R = -(wl - 440) / (440 - 380)
        G = 0.0
        B = 1.0
    elif 440 <= wl <= 490:
        R = 0.0
        G = (wl - 440) / (490 - 440)
        B = 1.0
    elif 490 <= wl <= 510:
        R = 0.0
        G = 1.0
        B = -(wl - 510) / (510 - 490)
    elif 510 <= wl <= 580:
        R = (wl - 510) / (580 - 510)
        G = 1.0
        B = 0.0
    elif 580 <= wl <= 645:
        R = 1.0
        G = -(wl - 645) / (645 - 580)
        B = 0.0
    elif 645 <= wl <= 780:
        R = 1.0
        G = 0.0
        B = 0.0
    else:
        R = G = B = 0.0

    factor = 1.0
    if wl < 420:
        factor = 0.3 + 0.7 * (wl - 380) / (420 - 380)
    elif wl > 700:
        factor = 0.3 + 0.7 * (780 - wl) / (780 - 700)

    R = int(round(R * factor * 255))
    G = int(round(G * factor * 255))
    B = int(round(B * factor * 255))

    return f'rgb({R},{G},{B})'

colors = [wavelength_to_rgb(wl) for wl in wavelengths_nm]

# plot
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=frequencies_thz,
    y=refractive_indices,
    mode='markers',
    marker=dict(color=colors, size=5),
    hovertemplate=
        'Frequency: %{x:.1f} THz<br>' +
        'Refractive Index: %{y:.6f}<extra></extra>'
))

# range slider
fig.update_layout(
    title='Refractive Index of Water vs Frequency (Visible Spectrum)',
    xaxis=dict(
        title='Frequency (THz)',
        rangeslider=dict(visible=True),
        range=[375, 750]
    ),
    yaxis=dict(title='Refractive Index (n)'),
    template='plotly_white',
    hovermode='closest',
    width=900,
    height=500
)

fig.show()