import math
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import minimize_scalar
from PIL import Image, ImageTk

#task 1b again-
def refractive_index_water(frequency_hz):
    f_scaled = np.asarray(frequency_hz) / 1e15
    inv_sq = 1.731 - 0.261 * f_scaled**2
    n_squared = 1 + 1 / np.sqrt(inv_sq)
    return np.sqrt(n_squared)

c_m_s = 299792458.0

def wavelength_nm_to_frequency_thz(lambda_nm):
    return (c_m_s / (lambda_nm * 1e-9)) / 1e12

def frequency_thz_to_wavelength_nm(freq_thz):
    return (c_m_s / (freq_thz * 1e12)) * 1e9

#angular deviation
def phi_from_theta(theta_rad, n):
    """phi = arcsin(sin theta / n) (clipped for numerical safety)"""
    return np.arcsin(np.clip(np.sin(theta_rad)/n, -1.0, 1.0))

def deviation_primary(theta_rad, n):
    """D1 = pi + 2 theta - 4 phi"""
    phi = phi_from_theta(theta_rad, n)
    return np.pi + 2.0*theta_rad - 4.0*phi

def deviation_secondary(theta_rad, n):
    """D2 = 2*pi + 2 theta - 6 phi"""
    phi = phi_from_theta(theta_rad, n)
    return 2.0*np.pi + 2.0*theta_rad - 6.0*phi

def find_theta_min(fun, n, tol=1e-10):
    """Find theta that minimizes fun(theta,n). Use bounded minimiser with fallback."""
    try:
        res = minimize_scalar(lambda th: fun(th, n),
                              bounds=(1e-9, np.pi/2 - 1e-9),
                              method='bounded',
                              options={'xatol': tol})
        if res.success:
            return res.x, res.fun, True
    except Exception:
        pass
    # fallback dense sampling
    thetas = np.linspace(1e-6, np.pi/2 - 1e-6, 5000)
    vals = fun(thetas, n)
    idx = np.argmin(vals)
    return thetas[idx], vals[idx], False

# Fresnel reflectances (unpolarised average)
# For incidence from medium n1 into n2 at angle theta1 (rad).
def fresnel_unpolarised_reflectance(n1, n2, theta1):
    """
    Return power reflectance (unpolarised) for incidence angle theta1 (in rad).
    Handles TIR (returns 1.0).
    """
    # handle normal incidence
    if theta1 <= 0.0:
        rs = (n1 - n2) / (n1 + n2)
        rp = rs
        return 0.5*(rs*rs + rp*rp)
    # Snell's law: n1 sin theta1 = n2 sin theta2
    sin_theta2 = (n1 / n2) * math.sin(theta1) if n2 != 0 else 2.0
    if abs(sin_theta2) > 1.0:
        # Total internal reflection
        return 1.0
    theta2 = math.asin(sin_theta2)
    # amplitude reflection coefficients (from medium n1)
    # rs = (n1 cos θ1 - n2 cos θ2) / (n1 cos θ1 + n2 cos θ2)
    # rp = (n2 cos θ1 - n1 cos θ2) / (n2 cos θ1 + n1 cos θ2)
    c1 = math.cos(theta1); c2 = math.cos(theta2)
    rs = (n1*c1 - n2*c2) / (n1*c1 + n2*c2)
    rp = (n2*c1 - n1*c2) / (n2*c1 + n1*c2)
    Rs = rs*rs; Rp = rp*rp
    return 0.5*(Rs + Rp)

# colour mapping
def wavelength_to_rgb(lambda_nm):
    lam = float(lambda_nm)
    if lam < 380: lam = 380.0
    if lam > 780: lam = 780.0
    if 380 <= lam <= 440:
        r = -(lam - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif 440 < lam <= 490:
        r = 0.0
        g = (lam - 440) / (490 - 440)
        b = 1.0
    elif 490 < lam <= 510:
        r = 0.0
        g = 1.0
        b = -(lam - 510) / (510 - 490)
    elif 510 < lam <= 580:
        r = (lam - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif 580 < lam <= 645:
        r = 1.0
        g = -(lam - 645) / (645 - 580)
        b = 0.0
    else:
        r = 1.0; g = 0.0; b = 0.0
    if lam < 420:
        factor = 0.3 + 0.7*(lam - 380)/(420 - 380)
    elif lam > 700:
        factor = 0.3 + 0.7*(780 - lam)/(780 - 700)
    else:
        factor = 1.0
    return (r*factor, g*factor, b*factor)

def compute_spectrum(lam_min_nm=405.0, lam_max_nm=790.0, ns=61):
    lambdas = np.linspace(lam_min_nm, lam_max_nm, ns)
    freqs_thz = wavelength_nm_to_frequency_thz(lambdas)
    freqs_hz = freqs_thz * 1e12
    # use Task1b Sellmeier
    n_vals = refractive_index_water(freqs_hz)
    # arrays
    theta1_min = np.zeros_like(lambdas)
    D1_min = np.zeros_like(lambdas)
    alpha1_deg = np.zeros_like(lambdas)
    phi1_deg = np.zeros_like(lambdas)
    R_internal_1 = np.zeros_like(lambdas)
    T_entry_1 = np.zeros_like(lambdas)
    I_approx_1 = np.zeros_like(lambdas)

    theta2_min = np.zeros_like(lambdas)
    D2_min = np.zeros_like(lambdas)
    alpha2_deg = np.zeros_like(lambdas)
    phi2_deg = np.zeros_like(lambdas)
    R_internal_2 = np.zeros_like(lambdas)
    T_entry_2 = np.zeros_like(lambdas)
    I_approx_2 = np.zeros_like(lambdas)

    for i, (lam, n) in enumerate(zip(lambdas, n_vals)):
        # primary
        t1, D1, ok1 = find_theta_min(deviation_primary, n)
        theta1_min[i] = t1; D1_min[i] = D1
        alpha1_deg[i] = math.degrees(math.pi - D1)  # pi - D1
        phi1 = phi_from_theta(t1, n); phi1_deg[i] = math.degrees(phi1)
        # reflection/transmission approx
        # entry: air->water at theta=t1, use reflectance from air (n1=1) into water (n2=n)
        R_entry = fresnel_unpolarised_reflectance(1.0, n, t1)
        # internal reflectance at incidence phi on water->air
        R_int = fresnel_unpolarised_reflectance(n, 1.0, phi1)
        # approximate intensity: T_entry * R_int * T_exit  with T≈1-R
        T_entry = max(0.0, 1.0 - R_entry)
        T_exit = T_entry  # approximately symmetrical
        I_approx_1[i] = T_entry * R_int * T_exit
        R_internal_1[i] = R_int; T_entry_1[i] = T_entry

        # secondary
        t2, D2, ok2 = find_theta_min(deviation_secondary, n)
        theta2_min[i] = t2; D2_min[i] = D2
        alpha2_deg[i] = math.degrees(D2 - math.pi)  # D2 - pi
        phi2 = phi_from_theta(t2, n); phi2_deg[i] = math.degrees(phi2)
        R_entry2 = fresnel_unpolarised_reflectance(1.0, n, t2)
        R_int2 = fresnel_unpolarised_reflectance(n, 1.0, phi2)
        T_entry2 = max(0.0, 1.0 - R_entry2)
        I_approx_2[i] = T_entry2 * (R_int2**2) * T_entry2  # two internal reflections
        R_internal_2[i] = R_int2; T_entry_2[i] = T_entry2

    # critical angles phi_c = arcsin(1/n)
    phi_crit_deg = np.degrees(np.arcsin(np.clip(1.0 / n_vals, -1.0, 1.0)))
    return {
        'lambdas_nm': lambdas,
        'freqs_thz': freqs_thz,
        'n_vals': n_vals,
        'theta1_min_rad': theta1_min,
        'D1_min_rad': D1_min,
        'alpha1_deg': alpha1_deg,
        'phi1_deg': phi1_deg,
        'R_internal_1': R_internal_1,
        'T_entry_1': T_entry_1,
        'I_approx_1': I_approx_1,
        'theta2_min_rad': theta2_min,
        'D2_min_rad': D2_min,
        'alpha2_deg': alpha2_deg,
        'phi2_deg': phi2_deg,
        'R_internal_2': R_internal_2,
        'T_entry_2': T_entry_2,
        'I_approx_2': I_approx_2,
        'phi_crit_deg': phi_crit_deg
    }

def plot_n_vs_frequency(ax, lambdas, freqs_thz, n_vals):
    ax.clear()
    colors = [wavelength_to_rgb(lam) for lam in lambdas]
    ax.scatter(freqs_thz, n_vals, c=colors, s=12)
    # smooth line via interpolation for display
    idx = np.argsort(freqs_thz)
    ax.plot(freqs_thz[idx], n_vals[idx], color='gray', alpha=0.4)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Refractive index n(λ)")
    ax.set_title("Refractive index of water vs frequency (Task 1b model)")
    ax.grid(True)

def plot_deviation_curves(ax, lambdas, n_vals):
    ax.clear()
    thetas = np.linspace(0.0, np.pi/2 - 1e-6, 2000)
    idxs = np.linspace(0, len(lambdas)-1, min(9, len(lambdas))).astype(int)
    for i in idxs:
        lam = lambdas[i]; n = n_vals[i]; color = wavelength_to_rgb(lam)
        eps = (2.0*thetas - 4.0*np.arcsin(np.clip(np.sin(thetas)/n, -1.0, 1.0)))  # the simplified eps(θ) some texts use
        # But we'll show full D1 as well:
        D1 = deviation_primary(thetas, n)
        ax.plot(np.degrees(thetas), np.degrees(D1), color=color, label=f"{int(lam)} nm")
    ax.set_xlabel("θ (deg)")
    ax.set_ylabel("D1(θ) (deg)")
    ax.set_title("Deviation D1(θ) (primary) for selected wavelengths")
    ax.legend(fontsize='small')
    ax.grid(True)

def plot_elevation_vs_freq(ax, freqs_thz, elev1, elev2, lambdas):
    ax.clear()
    ax.plot(freqs_thz, elev1, '-k', label='Primary (α1)')
    ax.plot(freqs_thz, elev2, '--', label='Secondary (α2)')
    for lam, f, a1 in zip(lambdas, freqs_thz, elev1):
        ax.plot(f, a1, 'o', color=wavelength_to_rgb(lam), markersize=4)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Elevation / angular radius (deg)")
    ax.set_title("Rainbow elevation vs frequency")
    ax.legend(fontsize='small')
    ax.grid(True)

def plot_phi_vs_freq(ax, freqs_thz, phi_deg, phi_crit_deg, label):
    ax.clear()
    ax.plot(freqs_thz, phi_deg, '-k', label=label)
    ax.plot(freqs_thz, phi_crit_deg, ':r', label='Critical angle φ_c')
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("φ (deg)")
    ax.set_title("Internal refraction angle φ vs frequency")
    ax.legend(fontsize='small')
    ax.grid(True)

# sky rainbow arcs (sea level)
def draw_rainbow_sky(ax, alpha_deg_array, lambdas, sun_elevation_deg=0.0, primary=True):
    """
    Draw semi-sky with arcs for each wavelength.
    alpha_deg_array: angular radius α for each λ (deg)
    sun_elevation_deg: sun elevation β (deg) (0=on horizon)
    primary True->primary, False->secondary
    We use simple mapping: top elevation of arc = α - sun_elevation deg
    To place arc: we map elevation to a radius in drawing coords.
    """
    ax.clear()
    # sky rectangle
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.1, 1.2)
    ax.axis('off')
    # Map α values to radial coordinates r
    elev = np.array(alpha_deg_array)
    # Compute top-of-arc elevation relative to horizon:
    top_elev = elev - sun_elevation_deg
    # Only draw those with positive top elevation (visible)
    visible = top_elev > 0.1  # small threshold
    if not np.any(visible):
        ax.text(0.5, 0.6, "Rainbow below horizon for this sun elevation", ha='center')
        return
    # map elevation in [min,max] to radius [0.25, 1.0]
    tmin = np.min(top_elev[visible]); tmax = np.max(top_elev[visible])
    if tmax - tmin < 1e-6:
        tmax = tmin + 1.0
    rmin = 0.25; rmax = 1.0
    rs = rmin + (top_elev - tmin) * (rmax - rmin) / (tmax - tmin)
    theta_arc = np.linspace(-np.pi/3, np.pi/3, 300)
    if primary:
        # draw innermost first (violet) -> outer (red)
        order = np.argsort(lambdas)  # ascending lam (violet->red)
        for idx in order:
            if not visible[idx]: continue
            color = wavelength_to_rgb(lambdas[idx])
            r = rs[idx]
            x = r * np.cos(theta_arc)
            y = r * np.sin(theta_arc) + 0.2
            ax.plot(x, y, color=color, linewidth=2.2, alpha=0.95)
    else:
        # secondary: draw outermost first, reversed color ordering
        order = np.argsort(lambdas)[::-1]
        for idx in order:
            if not visible[idx]: continue
            color = wavelength_to_rgb(lambdas[idx])
            r = rs[idx] + 0.08  # offset for outer
            x = r * np.cos(theta_arc)
            y = r * np.sin(theta_arc) + 0.2
            ax.plot(x, y, color=color, linewidth=2.0, alpha=0.95, linestyle='--')
#GUIIII
class DescartesRainbowApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Descartes Rainbow Explorer — Full model with schematics")
        self.geometry("1250x850")

        # Left controls
        left = ttk.Frame(self, width=340)
        left.pack(side='left', fill='y', padx=6, pady=6)
        right = ttk.Frame(self)
        right.pack(side='right', fill='both', expand=True, padx=6, pady=6)

        ttk.Label(left, text="Spectral range & sampling", font=('Helvetica', 11, 'bold')).pack(anchor='w')
        self.lam_min_var = tk.DoubleVar(value=405.0)
        self.lam_max_var = tk.DoubleVar(value=790.0)
        self.ns_var = tk.IntVar(value=61)
        ttk.Label(left, text="λ min (nm)").pack(anchor='w'); ttk.Entry(left, textvariable=self.lam_min_var, width=12).pack(anchor='w')
        ttk.Label(left, text="λ max (nm)").pack(anchor='w'); ttk.Entry(left, textvariable=self.lam_max_var, width=12).pack(anchor='w')
        ttk.Label(left, text="N samples").pack(anchor='w'); ttk.Entry(left, textvariable=self.ns_var, width=8).pack(anchor='w')

        ttk.Separator(left, orient='horizontal').pack(fill='x', pady=8)
        ttk.Label(left, text="Sea-level visualization", font=('Helvetica', 11, 'bold')).pack(anchor='w')
        self.sun_elev_var = tk.DoubleVar(value=5.0)
        ttk.Label(left, text="Sun elevation (deg)").pack(anchor='w')
        ttk.Scale(left, from_=0.0, to=45.0, variable=self.sun_elev_var, orient='horizontal', command=lambda e: self.update_sky()).pack(fill='x')
        ttk.Label(left, text="(sun elevation 0° means sun on horizon)").pack(anchor='w')

        ttk.Button(left, text="Compute spectrum & update plots", command=self.compute_and_update).pack(fill='x', pady=8)


        # Explanation pane with schematic image and text
        ttk.Separator(left, orient='horizontal').pack(fill='x', pady=8)
        ttk.Label(left, text="Explanation"
                             "", font=('Helvetica', 11, 'bold')).pack(anchor='w')

        # right: plotting panels
        self.fig, axs = plt.subplots(3, 2, figsize=(10,12))
        self.ax_n = axs[0,0]; self.ax_eps = axs[0,1]
        self.ax_elev = axs[1,0]; self.ax_phi = axs[1,1]
        self.ax_loss = axs[2,0]; self.ax_sky = axs[2,1]
        plt.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # initial compute & draw
        self.data = None
        self.compute_and_update()

    def compute_and_update(self):
        lam_min = float(self.lam_min_var.get()); lam_max = float(self.lam_max_var.get()); ns = int(self.ns_var.get())
        if lam_min <= 0 or lam_max <= lam_min:
            messagebox.showerror("Input error", "Set a valid wavelength range.")
            return
        self.data = compute_spectrum(lam_min_nm=lam_min, lam_max_nm=lam_max, ns=ns)
        self.update_plots()

    def update_plots(self):
        if self.data is None: return
        d = self.data
        lambdas = d['lambdas_nm']; freqs = d['freqs_thz']; n_vals = d['n_vals']
        # n vs frequency
        plot_n_vs_frequency(self.ax_n, lambdas, freqs, n_vals)
        # deviation curves
        plot_deviation_curves(self.ax_eps, lambdas, n_vals)
        # elevation vs freq
        plot_elevation_vs_freq(self.ax_elev, freqs, d['alpha1_deg'], d['alpha2_deg'], lambdas)
        # phi vs freq with critical angle
        plot_phi_vs_freq(self.ax_phi, freqs, d['phi1_deg'], d['phi_crit_deg'], 'Primary φ (deg)')
        # reflectance/loss panel: show R_internal primary and secondary approx intensity
        self.ax_loss.clear()
        self.ax_loss.plot(freqs, d['R_internal_1'], label='R_internal primary', color='purple')
        self.ax_loss.plot(freqs, d['R_internal_2'], label='R_internal secondary', color='magenta')
        self.ax_loss.plot(freqs, d['I_approx_1'], label='Approx intensity primary', color='blue')
        self.ax_loss.plot(freqs, d['I_approx_2'], label='Approx intensity secondary', color='green')
        self.ax_loss.set_xlabel('Frequency (THz)'); self.ax_loss.set_ylabel('Reflectance / intensity (arb.)')
        self.ax_loss.set_title('Internal reflectance & approximate intensities')
        self.ax_loss.legend(fontsize='small')
        self.ax_loss.grid(True)
        # sky plot initial (primary)
        draw_rainbow_sky(self.ax_sky, d['alpha1_deg'], lambdas, sun_elevation_deg=self.sun_elev_var.get(), primary=True)
        self.ax_sky.set_title('Simulated sky (primary) — sun elevation = {:.1f}°'.format(self.sun_elev_var.get()))
        self.fig.tight_layout()
        self.canvas.draw()

    def update_sky(self):
        if self.data is None: return
        draw_rainbow_sky(self.ax_sky, self.data['alpha1_deg'], self.data['lambdas_nm'], sun_elevation_deg=self.sun_elev_var.get(), primary=True)
        self.ax_sky.set_title('Simulated sky (primary) — sun elevation = {:.1f}°'.format(self.sun_elev_var.get()))
        self.canvas.draw()

    def export_report(self):
        if self.data is None:
            messagebox.showerror("No data", "Compute data first.")
            return
        outdir = filedialog.askdirectory(title="Choose output folder for report")
        if not outdir:
            return
        out_dir = os.path.join(outdir, 'descartes_report')
        os.makedirs(out_dir, exist_ok=True)
        # save each subplot as PNG
        eps_plot = os.path.join(out_dir, 'deviation_curves.png')
        n_plot = os.path.join(out_dir, 'n_vs_freq.png')
        elev_plot = os.path.join(out_dir, 'elevation_vs_freq.png')
        phi_plot = os.path.join(out_dir, 'phi_vs_freq.png')
        loss_plot = os.path.join(out_dir, 'loss_plot.png')
        sky_plot = os.path.join(out_dir, 'sky_plot.png')
        # render and save
        plot_n_vs_frequency(plt.figure().add_subplot(111), self.data['lambdas_nm'], self.data['freqs_thz'], self.data['n_vals']); plt.savefig(n_plot); plt.close()
        plot_deviation_curves(plt.figure().add_subplot(111), self.data['lambdas_nm'], self.data['n_vals']); plt.savefig(eps_plot); plt.close()
        plot_elevation_vs_freq(plt.figure().add_subplot(111), self.data['freqs_thz'], self.data['alpha1_deg'], self.data['alpha2_deg'], self.data['lambdas_nm']); plt.savefig(elev_plot); plt.close()
        plot_phi_vs_freq(plt.figure().add_subplot(111), self.data['freqs_thz'], self.data['phi1_deg'], self.data['phi_crit_deg'], 'Primary φ'); plt.savefig(phi_plot); plt.close()
        figloss = plt.figure(); ax = figloss.add_subplot(111); ax.plot(self.data['freqs_thz'], self.data['R_internal_1'], label='R_internal_1'); ax.plot(self.data['freqs_thz'], self.data['I_approx_1'], label='I_approx_1'); ax.legend(); ax.set_title('Losses'); plt.savefig(loss_plot); plt.close()
        # sky plot using function
        figsky = plt.figure(figsize=(6,4)); axsky = figsky.add_subplot(111); draw_rainbow_sky(axsky, self.data['alpha1_deg'], self.data['lambdas_nm'], sun_elevation_deg=self.sun_elev_var.get(), primary=True); plt.savefig(sky_plot); plt.close()

#Run
def main():
    app = DescartesRainbowApp()
    app.mainloop()

if __name__ == "__main__":
    main()
