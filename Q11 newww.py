#!/usr/bin/env python3


import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d

# ----------------------------
# Physics helpers & dispersion
# ----------------------------
# All trigonometric functions use radians internally. Conversion to degrees only for display.

def deviation_epsilon(theta_rad, n):
    """
    Deviation function (first-order rainbow, approximate Descartes):
      epsilon(theta) = 2*theta - 4 * arcsin( sin(theta) / n )
    Inputs:
      theta_rad : scalar or numpy array of incident angle theta (radians)
      n : refractive index (scalar)
    Returns:
      epsilon in radians (same shape as theta_rad)
    Notes:
      - Domain: arcsin argument must be in [-1,1]: sin(theta)/n must satisfy that.
        For n>1 and theta in [0, pi/2], sin(theta) in [0,1] so sin(theta)/n <= 1 -> safe.
      - Use np.arcsin which returns principal value in radians.
    """
    # ensure arrays
    theta = np.asarray(theta_rad)
    arg = np.sin(theta) / float(n)
    # clip for numerical safety
    arg_clipped = np.clip(arg, -1.0, 1.0)
    return 2.0 * theta - 4.0 * np.arcsin(arg_clipped)

def find_minimum_theta(n, theta_bounds=(1e-6, np.pi/2 - 1e-6)):
    """
    Find theta that minimizes epsilon(theta) for given refractive index n.
    - Uses a robust scalar minimizer with bracket/ bounds.
    Returns (theta_min_rad, epsilon_min_rad, success_flag)
    """
    # objective: scalar epsilon
    def obj(theta):
        return deviation_epsilon(theta, n)

    # minimize objective on [small, pi/2)
    res = minimize_scalar(obj, bounds=theta_bounds, method='bounded', options={'xatol':1e-9})
    if res.success:
        theta_min = res.x
        eps_min = res.fun
        return theta_min, eps_min, True
    else:
        # fallback: coarse sampling + argmin
        thetas = np.linspace(theta_bounds[0], theta_bounds[1], 2000)
        eps = deviation_epsilon(thetas, n)
        idx = np.argmin(eps)
        return thetas[idx], eps[idx], False

# Dispersion model(s)
def water_dispersion_interpolated():
    """
    Return an interpolator n(lambda_nm) for water in the visible.
    This uses a small set of widely-known approx values (typical lab values),
    then interpolates. If you have Task 1b data, replace these points.
    Values are approximate:
      lambda (nm) : n
      650 nm (red) ~ 1.331
      550 nm (green) ~ 1.333
      450 nm (blue) ~ 1.337
    This simple model suffices to illustrate spectral separation (colour dependence).
    """
    lam_nm = np.array([400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0])
    n_vals = np.array([1.3430, 1.3370, 1.3350, 1.3330, 1.3320, 1.3310, 1.3305])
    # cubic interpolation in the visible
    return interp1d(lam_nm, n_vals, kind='cubic', bounds_error=False, fill_value='extrapolate')

def cauchy_n(lambda_nm, A=1.320, B=4000.0, C=0.0):
    """
    Simple Cauchy form: n(lambda) = A + B / lambda^2 + C / lambda^4
    lambda in nm here (so B units adjusted).
    Default A,B are rough; user can set coefficients in GUI if desired.
    """
    lam = lambda_nm.astype(float)
    return A + B / (lam**2) + C / (lam**4)

# Helper to convert wavelength <-> frequency (in THz or relative units)
c_m_s = 299792458.0
def wavelength_to_frequency_thz(lambda_nm):
    lam_m = lambda_nm * 1e-9
    freq = c_m_s / lam_m   # Hz
    return freq / 1e12     # THz

# Map wavelength to an approximate display color (RGB tuple 0..1)
def wavelength_to_rgb(lambda_nm):
    # approximate visible color mapping for plotting markers
    lam = float(lambda_nm)
    # simple piecewise mapping adapted from public approximations
    if lam < 380:
        lam = 380
    if lam > 780:
        lam = 780
    # use an approximate conversion (no gamma correction)
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
        r = 1.0
        g = 0.0
        b = 0.0
    # intensity falloff near spectral ends
    if lam > 700:
        factor = 0.3 + 0.7*(780 - lam) / (780 - 700)
        r *= factor; g *= factor; b *= factor
    if lam < 420:
        factor = 0.3 + 0.7*(lam - 380) / (420 - 380)
        r *= factor; g *= factor; b *= factor
    return (r, g, b)

# ----------------------------
# GUI (Tkinter) — plotting panel
# ----------------------------
class RainbowApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Descartes Rainbow — Deviation & Rainbow Angle")
        self.geometry("1000x700")

        # default dispersion: interpolated water
        self.n_interp = water_dispersion_interpolated()
        self.use_cauchy = False
        self.cauchy_params = {'A':1.320, 'B':4000.0, 'C':0.0}

        # GUI layout: controls on left, plot area on right
        left = ttk.Frame(self, width=300)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)
        right = ttk.Frame(self)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Controls
        ttk.Label(left, text="Dispersion model", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W, pady=(4,2))
        self.disp_var = tk.StringVar(value="water_interp")
        rb1 = ttk.Radiobutton(left, text="Water (interpolated sample n(λ))", variable=self.disp_var, value="water_interp", command=self.update_dispersion_choice)
        rb1.pack(anchor=tk.W)
        rb2 = ttk.Radiobutton(left, text="Cauchy model (n=A + B/λ² + C/λ⁴)", variable=self.disp_var, value="cauchy", command=self.update_dispersion_choice)
        rb2.pack(anchor=tk.W)

        # Cauchy parameter inputs
        self.cA = tk.DoubleVar(value=self.cauchy_params['A'])
        self.cB = tk.DoubleVar(value=self.cauchy_params['B'])
        self.cC = tk.DoubleVar(value=self.cauchy_params['C'])
        frm_cauchy = ttk.Frame(left)
        frm_cauchy.pack(fill=tk.X, pady=(4,8))
        ttk.Label(frm_cauchy, text="A").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(frm_cauchy, textvariable=self.cA, width=8).grid(row=0, column=1)
        ttk.Label(frm_cauchy, text="B").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(frm_cauchy, textvariable=self.cB, width=8).grid(row=1, column=1)
        ttk.Label(frm_cauchy, text="C").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(frm_cauchy, textvariable=self.cC, width=8).grid(row=2, column=1)

        # Wavelength range controls
        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        ttk.Label(left, text="Wavelength range (nm)", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W)
        self.lam_min = tk.DoubleVar(value=380.0)
        self.lam_max = tk.DoubleVar(value=700.0)
        ttk.Label(left, text="λ min (nm)").pack(anchor=tk.W, pady=(4,0))
        ttk.Entry(left, textvariable=self.lam_min, width=12).pack(anchor=tk.W)
        ttk.Label(left, text="λ max (nm)").pack(anchor=tk.W, pady=(4,0))
        ttk.Entry(left, textvariable=self.lam_max, width=12).pack(anchor=tk.W)

        ttk.Label(left, text="Number of samples").pack(anchor=tk.W, pady=(6,0))
        self.nsamples = tk.IntVar(value=31)
        ttk.Entry(left, textvariable=self.nsamples, width=8).pack(anchor=tk.W)

        ttk.Button(left, text="Compute & Plot", command=self.compute_and_plot).pack(fill=tk.X, pady=(12,6))

        # Results text
        ttk.Label(left, text="Results / Notes", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W, pady=(8,2))
        self.results_txt = tk.Text(left, wrap=tk.WORD, height=12, width=36)
        self.results_txt.pack(fill=tk.BOTH, expand=False)

        # Plot area: two subplots side-by-side
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(9,4))
        self.fig = fig; self.ax_eps = ax1; self.ax_summary = ax2
        self.canvas = FigureCanvasTkAgg(fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # initial compute
        self.compute_and_plot()

    def update_dispersion_choice(self):
        choice = self.disp_var.get()
        if choice == 'cauchy':
            self.use_cauchy = True
        else:
            self.use_cauchy = False

    def get_n_of_lambda(self, lambdas_nm):
        # lambdas_nm may be scalar or array
        if self.use_cauchy:
            A = float(self.cA.get()); B = float(self.cB.get()); C = float(self.cC.get())
            return cauchy_n(np.asarray(lambdas_nm), A=A, B=B, C=C)
        else:
            # interpolated water model
            return self.n_interp(np.asarray(lambdas_nm))

    def compute_and_plot(self):
        # read inputs
        lam_min = float(self.lam_min.get()); lam_max = float(self.lam_max.get())
        ns = int(self.nsamples.get())
        if lam_min <= 0 or lam_max <= 0 or lam_max <= lam_min:
            messagebox.showerror("Input error", "Please set a valid wavelength range (λmin < λmax, >0).")
            return
        lambdas = np.linspace(lam_min, lam_max, ns)  # nm
        # Compute refractive index for each wavelength
        n_vals = self.get_n_of_lambda(lambdas)

        # theta grid for plotting epsilon(theta) curves
        thetas = np.linspace(1e-6, np.pi/2 - 1e-6, 2000)  # radians

        # Prepare plotting axes
        self.ax_eps.clear(); self.ax_summary.clear()
        self.results_txt.delete('1.0', tk.END)

        # We'll compute minima for each wavelength, collect results
        theta_mins = np.zeros_like(lambdas)
        eps_mins = np.zeros_like(lambdas)
        elevation_angles_deg = np.zeros_like(lambdas)  # alpha = 180° - eps_min (deg)
        freq_thz = wavelength_to_frequency_thz(lambdas)

        # Plot epsilon(theta) curves for a subset / or all, but to avoid clutter we sample up to ~12 curves
        max_curves = 12
        idx_plot = np.linspace(0, len(lambdas)-1, min(len(lambdas), max_curves)).astype(int)

        for i, lam in enumerate(lambdas):
            n = n_vals[i]
            # Evaluate epsilon on the coarse theta grid for plotting
            eps_theta = deviation_epsilon(thetas, n)
            if i in idx_plot:
                color = wavelength_to_rgb(lam)
                self.ax_eps.plot(np.degrees(thetas), np.degrees(eps_theta), color=color, label=f"{int(lam)} nm")

            # Find refined minimum via minimiser (more accurate than pick-sample)
            # Use bounds slightly inside (0, pi/2)
            theta_min, eps_min, success = find_minimum_theta(n, theta_bounds=(1e-9, np.pi/2 - 1e-9))
            theta_mins[i] = theta_min
            eps_mins[i] = eps_min
            elevation_angles_deg[i] = 180.0 - np.degrees(eps_min)

        # Annotate epsilon plot
        self.ax_eps.set_xlabel("Incident angle θ (degrees)")
        self.ax_eps.set_ylabel("Deviation ε(θ) (degrees)")
        self.ax_eps.set_title("Deviation ε(θ) versus incident angle (selected wavelengths)")
        self.ax_eps.grid(True)
        if len(idx_plot) > 0:
            self.ax_eps.legend(fontsize='small', loc='upper left', ncol=1)

        # Mark minima for all wavelengths (as coloured dots plotted on epsilon plot)
        # We'll map theta_mins -> (theta_min in deg, epsilon_min in deg)
        for lam, tmin, emin in zip(lambdas, theta_mins, eps_mins):
            col = wavelength_to_rgb(lam)
            self.ax_eps.plot(np.degrees(tmin), np.degrees(emin), 'o', color=col, markersize=4, alpha=0.9)

        # SUMMARY PLOT: elevation angle vs frequency (and second x-axis for wavelength)
        self.ax_summary.plot(freq_thz, elevation_angles_deg, '-k', lw=1.2)
        # overlay coloured markers per wavelength
        for lam, f_hz, elev_deg in zip(lambdas, freq_thz, elevation_angles_deg):
            self.ax_summary.plot(f_hz, elev_deg, 'o', color=wavelength_to_rgb(lam), markersize=6)

        self.ax_summary.set_xlabel("Frequency (THz)")
        self.ax_summary.set_ylabel("Elevation angle (deg)  —  180° − ε_min")
        self.ax_summary.set_title("Rainbow elevation angle vs frequency")
        self.ax_summary.grid(True)

        # Add second x-axis: wavelength nm (reverse axis because higher freq -> lower lambda)
        ax_wl = self.ax_summary.twiny()
        # set same limits but map freq->wavelength
        f_min, f_max = freq_thz.min(), freq_thz.max()
        # create ticks in THz corresponding to some nice wavelength ticks
        wl_ticks_nm = np.linspace(lam_min, lam_max, 6)
        f_ticks = wavelength_to_frequency_thz(wl_ticks_nm)
        ax_wl.set_xlim(self.ax_summary.get_xlim())
        ax_wl.set_xticks(f_ticks)
        ax_wl.set_xticklabels([f"{int(x)}" for x in wl_ticks_nm], rotation=0)
        ax_wl.set_xlabel("Wavelength (nm)")

        # Print numeric summary in the text box (markscheme style)
        self.results_txt.insert(tk.END, "Rainbow minima (summary)\n")
        self.results_txt.insert(tk.END, "-"*36 + "\n")
        self.results_txt.insert(tk.END, f"{'λ (nm)':>8s} {'n(λ)':>8s} {'θ_min (deg)':>12s} {'ε_min (deg)':>12s} {'elev (deg)':>10s}\n")
        for lam, n, tmin, emin, elev in zip(lambdas, n_vals, theta_mins, eps_mins, elevation_angles_deg):
            self.results_txt.insert(tk.END, f"{lam:8.1f} {n:8.6f} {np.degrees(tmin):12.6f} {np.degrees(emin):12.6f} {elev:10.6f}\n")

        # Tight layout, redraw canvas
        self.fig.tight_layout()
        self.canvas.draw()

        # Additional note: check expected physical values (sanity check)
        # Typical primary rainbow around ~42° for water: check if our elevation angles lie ~40-42°
        avg_elev = np.mean(elevation_angles_deg)
        note = f"\nSanity check: mean elevation ≈ {avg_elev:.3f}°  (expect ~42° for water primary rainbow)\n"
        self.results_txt.insert(tk.END, note)

# ----------------------------
# Run app
# ----------------------------
def main():
    app = RainbowApp()
    app.mainloop()

if __name__ == "__main__":
    main()
