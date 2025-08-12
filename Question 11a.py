import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.cm as cm

deg2rad = np.pi / 180
rad2deg = 180 / np.pi

A_deg = 60
A = A_deg * deg2rad

def refractive_index_BK7(wavelength_nm):
    wl_um = wavelength_nm / 1000
    A_c = 1.5046
    B_c = 0.00420
    C_c = 0.0
    n = A_c + B_c / wl_um**2 + C_c / wl_um**4
    return n

def snell(n1, n2, theta1):
    sin_theta2 = n1/n2 * np.sin(theta1)
    if np.abs(sin_theta2) > 1.0:
        return None  # total internal reflection
    return np.arcsin(sin_theta2)

def line_intersection(p, d, q, e):
    A = np.array([d, -e]).T
    b = q - p
    if np.linalg.matrix_rank(A) < 2:
        return None
    sol = np.linalg.solve(A, b)
    t, u = sol
    if t < 0 or u < 0 or u > 1:
        return None
    return p + t * d

def trace_ray(wavelength_nm, incidence_angle_deg):
    n_air = 1.0
    n_glass = refractive_index_BK7(wavelength_nm)
    theta1 = incidence_angle_deg * deg2rad
    theta2 = snell(n_air, n_glass, theta1)
    if theta2 is None:
        return None
    theta_inc2 = A - theta2
    theta3 = snell(n_glass, n_air, theta_inc2)
    if theta3 is None:
        return None

    O = np.array([0, 0])
    P1 = np.array([1, 0])
    P2 = np.array([np.cos(A), np.sin(A)])

    incidence_point = (O + P1) / 2

    dir_incident_angle = 90 - incidence_angle_deg
    dir_incident = np.array([np.cos(dir_incident_angle * deg2rad), np.sin(dir_incident_angle * deg2rad)])
    start_point = incidence_point - dir_incident * 2

    dir_inside_angle = 90 - theta2 * rad2deg
    dir_inside = np.array([np.cos(dir_inside_angle * deg2rad), np.sin(dir_inside_angle * deg2rad)])

    face2_vec = P2 - P1
    exit_point = line_intersection(incidence_point, dir_inside, P1, face2_vec)
    if exit_point is None:
        return None

    dir_exit_angle = A + theta3
    dir_exit = np.array([np.cos(dir_exit_angle), np.sin(dir_exit_angle)])
    exit_end_point = exit_point + dir_exit * 2

    return {
        "start_point": start_point,
        "incidence_point": incidence_point,
        "exit_point": exit_point,
        "exit_end_point": exit_end_point,
        "wavelength_nm": wavelength_nm,
        "n_glass": n_glass,
        "incidence_angle_deg": incidence_angle_deg,
        "theta2_deg": theta2 * rad2deg,
        "theta_inc2_deg": theta_inc2 * rad2deg,
        "theta3_deg": theta3 * rad2deg
    }

def plot_prism(ax):
    O = np.array([0, 0])
    P1 = np.array([1, 0])
    P2 = np.array([np.cos(A), np.sin(A)])
    prism_x = [O[0], P1[0], P2[0], O[0]]
    prism_y = [O[1], P1[1], P2[1], O[1]]
    ax.fill(prism_x, prism_y, color='lightgray', alpha=0.5)
    ax.plot(prism_x, prism_y, color='black')

def interactive_dispersion():
    fig, ax = plt.subplots(figsize=(8,6))
    plt.subplots_adjust(left=0.15, bottom=0.25)
    incidence_angle_init = 45

    wavelengths = np.linspace(400, 700, 50)

    def draw_rays(inc_angle_deg):
        ax.cla()
        plot_prism(ax)
        colors = cm.rainbow((wavelengths - 400) / 300)

        # Incident beam (white)
        O = np.array([0, 0])
        P1 = np.array([1, 0])
        incidence_point = (O + P1) / 2
        dir_incident_angle = 90 - inc_angle_deg
        dir_incident = np.array([np.cos(dir_incident_angle * deg2rad), np.sin(dir_incident_angle * deg2rad)])
        start_point = incidence_point - dir_incident * 2
        ax.plot([start_point[0], incidence_point[0]], [start_point[1], incidence_point[1]], color='white', linewidth=2, label='Incident beam')

        # Dispersed rays inside and out
        for wl, col in zip(wavelengths, colors):
            ray = trace_ray(wl, inc_angle_deg)
            if ray is None:
                continue
            sp = ray["start_point"]
            ip = ray["incidence_point"]
            ep = ray["exit_point"]
            eep = ray["exit_end_point"]
            # Inside prism ray
            ax.plot([ip[0], ep[0]], [ip[1], ep[1]], color=col, alpha=0.8)
            # Exit ray
            ax.plot([ep[0], eep[0]], [ep[1], eep[1]], color=col, alpha=0.8)

        ax.set_title(f"Dispersion through prism - Incidence angle = {inc_angle_deg:.1f}°", color='black')
        ax.set_xlabel("x (arbitrary units)", color='black')
        ax.set_ylabel("y (arbitrary units)", color='black')
        ax.axis('equal')
        ax.set_xlim(-1, 2)
        ax.set_ylim(-0.5, 1.5)
        ax.grid(True)
        ax.patch.set_facecolor('k')  # black background for better contrast with white incident beam
        ax.legend(loc='upper left', facecolor='white')

    draw_rays(incidence_angle_init)

    ax_angle = plt.axes([0.15, 0.1, 0.65, 0.03])
    slider_angle = Slider(ax_angle, "Incidence angle (deg)", 10, 80, valinit=incidence_angle_init, color='lightblue')

    def update(val):
        angle = slider_angle.val
        draw_rays(angle)
        fig.canvas.draw_idle()

    slider_angle.on_changed(update)
    plt.show()

if __name__ == "__main__":
    interactive_dispersion()
