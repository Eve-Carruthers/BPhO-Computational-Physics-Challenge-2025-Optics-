import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as Polygon

# Sellmeier equation for BK7 glass
def n_bk7(wavelength_nm):
    wavelength_um = wavelength_nm / 1000  # convert nm to micrometers
    B1, B2, B3 = 1.03961212, 0.231792344, 1.01046945
    C1, C2, C3 = 0.00600069867, 0.0200179144, 103.560653
    return np.sqrt(1 + (B1 * wavelength_um**2) / (wavelength_um**2 - C1) +
                   (B2 * wavelength_um**2) / (wavelength_um**2 - C2) +
                   (B3 * wavelength_um**2) / (wavelength_um**2 - C3))

# Snell's law
def snell(theta_incident, n1, n2):
    return np.arcsin(n1 * np.sin(theta_incident) / n2)

# Draw prism refraction for multiple wavelengths
def prism_refraction(ax, apex_angle, incident_angle):
    ax.clear()
#SHAPES AND COLOURS
    #Prism
    prism_height = 2
    prism_width = 2
    prism_vertices = np.array([[0, 0], [prism_width, 0], [prism_width/2, prism_height]])
    prism = Polygon.Polygon(prism_vertices, closed=True, facecolor='lightgrey', alpha=0.4, edgecolor='black')
    ax.add_patch(prism)

    #white incoming beam
    ax.plot([-2, 0], [1, 1], color='white', lw=3, solid_capstyle='round', marker='')

    # Wavelengths for dispersion (visible spectrum)
    wavelengths = np.linspace(400, 700, 7)  # nm
    colours = [plt.cm.jet((wl - 400) / 300) for wl in wavelengths]

    for wl, col in zip(wavelengths, colours):
        n_prism = n_bk7(wl)
        n_air = 1.0

        # Angles inside prism
        theta1 = np.radians(incident_angle)
        theta2 = snell(theta1, n_air, n_prism)
        theta3 = apex_angle - theta2
        theta4 = snell(theta3, n_prism, n_air)

        # Beam path points
        entry_x, entry_y = 0, 1
        inside_x = prism_width/2
        inside_y = entry_y + np.tan(theta2) * (inside_x - entry_x)
        exit_x, exit_y = prism_width, inside_y + np.tan(theta4) * (prism_width - inside_x)

        # Inside prism ray
        ax.plot([entry_x, inside_x], [entry_y, inside_y], color=col, lw=1.5)

        # Exit beam
        ax.plot([inside_x, exit_x + 1.5], [inside_y, exit_y + np.tan(theta4) * 1.5],
                color=col, lw=1.5)

    # Annotations
    ax.text(-1.5, 1.1, "Incident Beam", fontsize=10, color='black')
    ax.text(1.2, 1.8, "Dispersion\n(Snell's Law + Sellmeier Eq.)", fontsize=9, color='black')
    ax.text(2.3, 0.2, "Emerging spectrum", fontsize=10, color='black')

    ax.set_xlim(-2, 4)
    ax.set_ylim(-0.5, 3)
    ax.set_aspect('equal')
    ax.axis('off')

# Main plot
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
apex_angle_init = np.radians(60)
incident_angle_init = 30

prism_refraction(ax, apex_angle_init, incident_angle_init)

# Sliders
ax_apex = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgrey')
ax_incident = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor='lightgrey')

slider_apex = Slider(ax_apex, 'Apex Angle (°)', 30, 90, valinit=np.degrees(apex_angle_init))
slider_incident = Slider(ax_incident, 'Incident Angle (°)', 0, 80, valinit=incident_angle_init)

def update(val):
    prism_refraction(ax, np.radians(slider_apex.val), slider_incident.val)
    fig.canvas.draw_idle()

slider_apex.on_changed(update)
slider_incident.on_changed(update)

plt.show()
