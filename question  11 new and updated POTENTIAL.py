
import os
import sys
import shutil
import subprocess
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
from PIL import Image

# 1b Sellmeier model
def refractive_index_water(frequency_hz):
    """
    Task 1b Sellmeier-style approximation as supplied by user.
    Input: frequency in Hz (scalar or numpy array)
    Output: refractive index n
    """
    f_scaled = np.asarray(frequency_hz) / 1e15
    inv_sq = 1.731 - 0.261 * f_scaled**2
    n_squared = 1 + 1 / np.sqrt(inv_sq)
    return np.sqrt(n_squared)


#  physics functions, minimization, colour mapping

c_m_s = 299792458.0

def wavelength_to_frequency_thz(lambda_nm):
    lam_m = lambda_nm * 1e-9
    freq = c_m_s / lam_m   # Hz
    return freq / 1e12     # THz

def frequency_thz_to_wavelength_nm(f_thz):
    f_hz = f_thz * 1e12
    lam_m = c_m_s / f_hz
    return lam_m * 1e9

def deviation_epsilon(theta_rad, n):
    # epsilon(theta) = 2*theta - 4*arcsin( sin(theta) / n )
    arg = np.sin(theta_rad) / float(n)
    arg_clipped = np.clip(arg, -1.0, 1.0)
    return 2.0 * theta_rad - 4.0 * np.arcsin(arg_clipped)

def find_minimum_theta_for_n(n, theta_bounds=(1e-9, np.pi/2 - 1e-9)):
    # minimize epsilon(theta) on bounds
    def obj(theta):
        return deviation_epsilon(theta, n)
    res = minimize_scalar(obj, bounds=theta_bounds, method='bounded', options={'xatol':1e-10})
    if res.success:
        return res.x, res.fun, True
    else:
        # fallback coarse sampling
        thetas = np.linspace(theta_bounds[0], theta_bounds[1], 5000)
        eps = deviation_epsilon(thetas, n)
        idx = np.argmin(eps)
        return thetas[idx], eps[idx], False

def wavelength_to_rgb(lambda_nm):
    # approximate mapping (same as before)
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
    # adjust intensity near ends
    if lam < 420:
        factor = 0.3 + 0.7*(lam - 380)/(420 - 380)
    elif lam > 700:
        factor = 0.3 + 0.7*(780 - lam)/(780 - 700)
    else:
        factor = 1.0
    return (r*factor, g*factor, b*factor)

# ---------------------------
# Main computation function (computes minima and data)
# ---------------------------
def compute_rainbow_data(lam_min=380.0, lam_max=700.0, ns=61, dispersion_model='sellmeier'):
    """
    Compute deviation minima for a spectrum of wavelengths.
    dispersion_model: 'sellmeier' uses the provided frequency->n function,
                      'interp' or 'cauchy' could be added.
    Returns dictionary with arrays: wavelengths (nm), frequencies (THz), n, theta_min (rad), eps_min (rad), elevation_deg
    """
    wavelengths = np.linspace(lam_min, lam_max, ns)
    freqs_thz = wavelength_to_frequency_thz(wavelengths)
    freqs_hz = freqs_thz * 1e12

    # Get refractive indices using Task1b Sellmeier function:
    if dispersion_model == 'sellmeier':
        n_vals = refractive_index_water(freqs_hz)
    else:
        # fallback: small interpolated water model if needed
        # (but in this script we use sellmeier by default)
        n_vals = refractive_index_water(freqs_hz)

    theta_mins = np.zeros_like(wavelengths)
    eps_mins = np.zeros_like(wavelengths)
    success_flags = np.zeros_like(wavelengths, dtype=bool)
    for i, n in enumerate(n_vals):
        tmin, emin, ok = find_minimum_theta_for_n(n)
        theta_mins[i] = tmin
        eps_mins[i] = emin
        success_flags[i] = ok

    elevation_deg = 180.0 - np.degrees(eps_mins)

    return {
        'wavelengths_nm': wavelengths,
        'freqs_thz': freqs_thz,
        'n_vals': n_vals,
        'theta_mins_rad': theta_mins,
        'eps_mins_rad': eps_mins,
        'elevation_deg': elevation_deg,
        'success': success_flags
    }

def plot_colored_rainbow(elevations_deg, wavelengths, figsize=(6,6), outpath=None):
    """
    Create a simple visualization: draw concentric circular arcs at angles = elevations_deg
    relative to an 'observer' at centre. Angles are in degrees; we map them to radii.
    Save figure to outpath if provided and return the figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-0.1, 1.5)
    ax.axis('off')
    # Map elevation in degrees into radius in [0.2..1.0] (simple linear mapping)
    elev = np.asarray(elevations_deg)
    r_min = 0.25; r_max = 1.0
    # Typical elevations ~ 40-42 -> map within r_min..r_max
    emin = np.min(elev); emax = np.max(elev)
    if emax - emin < 1e-6:
        emax = emin + 1.0
    rs = r_min + (elev - emin) * (r_max - r_min) / (emax - emin)
    # draw arcs spanning -60°..60° horizontally to depict rainbow arc
    theta_arc = np.linspace(-np.pi/3, np.pi/3, 200)
    for r, lam in zip(rs[::-1], wavelengths[::-1]):  # draw back-to-front so red outermost can be on top if ordered
        col = wavelength_to_rgb(lam)
        x = r * np.cos(theta_arc)
        y = r * np.sin(theta_arc) + 0.2  # lift up a bit
        ax.plot(x, y, color=col, linewidth=2.2, solid_capstyle='round', alpha=0.95)
    # annotate approximate antisolar point (center of circle)
    ax.plot(0, 0.2, marker='o', color='black')  # observer
    ax.text(0.02, 0.12, "Observer (antisolar origin)", fontsize=8)
    if outpath:
        fig.savefig(outpath, dpi=200, bbox_inches='tight')
    return fig

# ---------------------------
# Report generation: produce PNGs, a PDF (matplotlib PdfPages), and a LaTeX file + try to compile

LATEX_TEMPLATE = r"""
\documentclass[11pt]{article}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{geometry}
\geometry{margin=1in}
\begin{document}
\title{Descartes Rainbow — Report}
\author{Generated by descartes\_rainbow\_full\_tk.py}
\date{\today}
\maketitle

\section*{Introduction and physics}
We compute the deviation function for first-order rainbow (one internal reflection) using Descartes' approximate formula:
\[
\varepsilon(\theta) = 2\theta - 4\arcsin\!\left(\frac{\sin\theta}{n(\lambda)}\right).
\]
The rainbow angle corresponds to the minimum of \(\varepsilon(\theta)\) with respect to incident angle \(\theta\). We find this minimum numerically for each wavelength using a bounded scalar minimiser.

\section*{Dispersion model}
The refractive index used is the Task 1b Sellmeier-style approximation (frequency input), as provided.

\section*{Results}
\begin{figure}[ht]
\centering
\includegraphics[width=0.9\textwidth]{{{eps_theta_plot}}}
\caption{Deviation $\varepsilon(\theta)$ vs $\theta$ for selected wavelengths (degrees).}
\end{figure}

\begin{figure}[ht]
\centering
\includegraphics[width=0.7\textwidth]{{{elevation_plot}}}
\caption{Rainbow elevation angle (degrees) vs frequency (THz).}
\end{figure}

\begin{figure}[ht]
\centering
\includegraphics[width=0.6\textwidth]{{{rainbow_visual}}}
\caption{Simulated coloured rainbow arcs at computed elevation angles.}
\end{figure}

\section*{Numerical table}
\begin{tabular}{rrrrr}
\toprule
$\lambda$ (nm) & $n(\lambda)$ & $\theta_{\min}$ (deg) & $\varepsilon_{\min}$ (deg) & elevation (deg) \\
\midrule
{{table_rows}}
\bottomrule
\end{tabular}

\end{document}
"""

def produce_report_and_pdf(data, outdir="descartes_report", try_latex_compile=True):
    """
    data: dict returned by compute_rainbow_data
    Produces:
      - eps_theta_plot.png
      - elevation_plot.png
      - rainbow_visual.png
      - combined_report.pdf (matplotlib PdfPages)
      - report.tex and attempt to compile to report.pdf using pdflatex (optional)
    """
    os.makedirs(outdir, exist_ok=True)
    wavelengths = data['wavelengths_nm']; freqs = data['freqs_thz']
    n_vals = data['n_vals']; theta_mins = data['theta_mins_rad']; eps_mins = data['eps_mins_rad']; elev = data['elevation_deg']

    # 1) plot eps(theta) for a subsample of wavelengths (selected)
    thetas = np.linspace(1e-6, np.pi/2 - 1e-6, 2000)
    fig1, ax1 = plt.subplots(figsize=(6.5,4))
    idxs = np.linspace(0, len(wavelengths)-1, min(10, len(wavelengths))).astype(int)
    for i in idxs:
        n = n_vals[i]; lam = wavelengths[i]
        eps_theta = deviation_epsilon(thetas, n)
        ax1.plot(np.degrees(thetas), np.degrees(eps_theta), color=wavelength_to_rgb(lam), label=f"{int(lam)} nm")
        ax1.plot(np.degrees(theta_mins[i]), np.degrees(eps_mins[i]), 'o', color=wavelength_to_rgb(lam), markersize=4)
    ax1.set_xlabel("Incident angle θ (deg)"); ax1.set_ylabel("Deviation ε(θ) (deg)")
    ax1.set_title("Deviation ε(θ) vs θ (selected wavelengths)")
    ax1.grid(True); ax1.legend(fontsize='small')
    eps_plot_path = os.path.join(outdir, "eps_theta_plot.png")
    fig1.savefig(eps_plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig1)

    # 2) elevation vs frequency plot
    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.plot(freqs, elev, '-k', lw=1.2)
    for lam, f, e in zip(wavelengths, freqs, elev):
        ax2.plot(f, e, 'o', color=wavelength_to_rgb(lam))
    ax2.set_xlabel("Frequency (THz)"); ax2.set_ylabel("Elevation angle (deg)")
    ax2.set_title("Rainbow elevation vs frequency")
    ax2.grid(True)
    elev_plot_path = os.path.join(outdir, "elevation_plot.png")
    fig2.savefig(elev_plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig2)

    # 3) rainbow visual
    rainbow_path = os.path.join(outdir, "rainbow_visual.png")
    fig3 = plot_colored_rainbow(elev, wavelengths, figsize=(6,6), outpath=rainbow_path)
    plt.close(fig3)

    # 4) create a simple matplotlib PDF combining the images
    pdf_path = os.path.join(outdir, "combined_report.pdf")
    with PdfPages(pdf_path) as pdf:
        # page 1: eps plot
        img1 = Image.open(eps_plot_path)
        fig = plt.figure(figsize=(8.27, 11.69))  # A4
        plt.imshow(img1); plt.axis('off'); plt.title('Epsilon vs theta')
        pdf.savefig(); plt.close(fig)
        # page 2: elevation plot
        img2 = Image.open(elev_plot_path)
        fig = plt.figure(figsize=(8.27, 11.69))
        plt.imshow(img2); plt.axis('off'); plt.title('Elevation vs frequency')
        pdf.savefig(); plt.close(fig)
        # page 3: rainbow visual
        img3 = Image.open(rainbow_path)
        fig = plt.figure(figsize=(8.27, 11.69))
        plt.imshow(img3); plt.axis('off'); plt.title('Simulated Rainbow')
        pdf.savefig(); plt.close(fig)

    # 5) Build LaTeX file with table rows
    table_rows = []
    for lam, n, tmin, emin, e_deg in zip(wavelengths, n_vals, theta_mins, eps_mins, elev):
        table_rows.append(f"{lam:.1f} & {n:.6f} & {np.degrees(tmin):.6f} & {np.degrees(emin):.6f} & {e_deg:.6f} \\\\")
    table_rows_str = "\n".join(table_rows)
    tex_content = LATEX_TEMPLATE.replace("{{{eps_theta_plot}}}", os.path.basename(eps_plot_path)) \
                               .replace("{{{elevation_plot}}}", os.path.basename(elev_plot_path)) \
                               .replace("{{{rainbow_visual}}}", os.path.basename(rainbow_path)) \
                               .replace("{{{table_rows}}}", table_rows_str)
    tex_path = os.path.join(outdir, "report.tex")
    with open(tex_path, "w") as f:
        f.write(tex_content)

    # Try to compile with pdflatex if available
    compiled_pdf_path = os.path.join(outdir, "report_from_tex.pdf")
    if try_latex_compile:
        pdflatex = shutil.which("pdflatex")
        if pdflatex is not None:
            try:
                cwd = os.getcwd()
                os.chdir(outdir)
                # copy images into cwd already saved there
                cmd = [pdflatex, "-interaction=nonstopmode", "report.tex"]
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # second run for refs
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if os.path.exists("report.pdf"):
                    # move/rename
                    shutil.move(os.path.join(outdir, "report.pdf"), compiled_pdf_path)
                os.chdir(cwd)
            except Exception as e:
                os.chdir(cwd)
                print("LaTeX compilation failed or pdflatex error:", e)
        else:
            print("pdflatex not found on PATH; LaTeX file saved but not compiled.")

    return {
        'eps_plot': eps_plot_path,
        'elev_plot': elev_plot_path,
        'rainbow_plot': rainbow_path,
        'combined_pdf': pdf_path,
        'tex': tex_path,
        'tex_pdf': compiled_pdf_path if os.path.exists(compiled_pdf_path) else None
    }

# ---------------------------
# GUI (simple, re-using earlier patterns)
# ---------------------------
class DescartesGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Descartes Rainbow — Full App")
        self.geometry("1100x750")

        left = ttk.Frame(self, width=320)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)
        right = ttk.Frame(self)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Controls
        ttk.Label(left, text="Dispersion / Input", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W)
        self.disp_choice = tk.StringVar(value='sellmeier')
        ttk.Radiobutton(left, text="Task 1b Sellmeier (provided)", variable=self.disp_choice, value='sellmeier').pack(anchor=tk.W)
        ttk.Radiobutton(left, text="Interpolated example (fallback)", variable=self.disp_choice, value='interp').pack(anchor=tk.W)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        ttk.Label(left, text="Wavelength range (nm)").pack(anchor=tk.W)
        self.lam_min = tk.DoubleVar(value=380.0)
        self.lam_max = tk.DoubleVar(value=700.0)
        ttk.Entry(left, textvariable=self.lam_min, width=12).pack(anchor=tk.W, pady=2)
        ttk.Entry(left, textvariable=self.lam_max, width=12).pack(anchor=tk.W, pady=2)
        ttk.Label(left, text="Number of wavelengths (samples)").pack(anchor=tk.W, pady=(6,0))
        self.ns = tk.IntVar(value=61)
        ttk.Entry(left, textvariable=self.ns, width=8).pack(anchor=tk.W, pady=2)

        ttk.Button(left, text="Compute rainbow data", command=self.compute_and_show).pack(fill=tk.X, pady=(12,6))
        ttk.Button(left, text="Generate report (PNG+PDF+LaTeX)", command=self.generate_report).pack(fill=tk.X, pady=6)

        # Right: plotting area with two subplots and a small placeholder for rainbow visual
        fig, axes = plt.subplots(2,2, figsize=(10,6))
        self.fig = fig
        self.ax_eps = axes[0,0]
        self.ax_summary = axes[0,1]
        self.ax_table = axes[1,0]  # used to show text table (as image)
        self.ax_rainbow = axes[1,1]
        plt.tight_layout()
        self.canvas = FigureCanvasTkAgg(fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Results storage
        self.current_data = None
        self.report_files = None

        # initial compute
        self.compute_and_show()

    def compute_and_show(self):
        lam_min = float(self.lam_min.get()); lam_max = float(self.lam_max.get()); ns = int(self.ns.get())
        if lam_min <= 0 or lam_max <= 0 or lam_max <= lam_min:
            messagebox.showerror("Input error", "Please set a valid wavelength range (λmin < λmax, >0).")
            return
        model = 'sellmeier' if self.disp_choice.get() == 'sellmeier' else 'interp'
        data = compute_rainbow_data(lam_min=lam_min, lam_max=lam_max, ns=ns, dispersion_model=model)
        self.current_data = data
        self.update_plots(data)

    def update_plots(self, data):
        wavelengths = data['wavelengths_nm']; freqs = data['freqs_thz']
        n_vals = data['n_vals']; theta_mins = data['theta_mins_rad']; eps_mins = data['eps_mins_rad']; elev = data['elevation_deg']

        # Epsilon(theta) plot (plot a subset of curves)
        self.ax_eps.clear()
        thetas = np.linspace(1e-6, np.pi/2 - 1e-6, 2000)
        idxs = np.linspace(0, len(wavelengths)-1, min(10, len(wavelengths))).astype(int)
        for i in idxs:
            lam = wavelengths[i]
            eps_theta = deviation_epsilon(thetas, n_vals[i])
            self.ax_eps.plot(np.degrees(thetas), np.degrees(eps_theta), color=wavelength_to_rgb(lam), label=f"{int(lam)} nm")
            self.ax_eps.plot(np.degrees(theta_mins[i]), np.degrees(eps_mins[i]), 'o', color=wavelength_to_rgb(lam), markersize=4)
        self.ax_eps.set_xlabel("θ (deg)"); self.ax_eps.set_ylabel("ε (deg)"); self.ax_eps.set_title("ε(θ) for selected wavelengths"); self.ax_eps.grid(True)
        if len(idxs) > 0:
            self.ax_eps.legend(fontsize='small')

        # Summary plot: elevation vs frequency
        self.ax_summary.clear()
        self.ax_summary.plot(freqs, elev, '-k')
        for lam, f, e in zip(wavelengths, freqs, elev):
            self.ax_summary.plot(f, e, 'o', color=wavelength_to_rgb(lam))
        self.ax_summary.set_xlabel("Frequency (THz)"); self.ax_summary.set_ylabel("Elevation angle (deg)")
        self.ax_summary.set_title("Elevation angle vs Frequency"); self.ax_summary.grid(True)

        # Rainbow visual
        self.ax_rainbow.clear()
        # produce similar matplotlib content as plot_colored_rainbow but embed into axes
        self.ax_rainbow.set_aspect('equal'); self.ax_rainbow.axis('off')
        elev_arr = elev
        emin = np.min(elev_arr); emax = np.max(elev_arr)
        if emax - emin < 1e-6:
            emax = emin + 1.0
        r_min = 0.25; r_max = 1.0
        rs = r_min + (elev_arr - emin) * (r_max - r_min) / (emax - emin)
        theta_arc = np.linspace(-np.pi/3, np.pi/3, 200)
        for r, lam in zip(rs[::-1], wavelengths[::-1]):
            col = wavelength_to_rgb(lam)
            x = r * np.cos(theta_arc)
            y = r * np.sin(theta_arc) + 0.2
            self.ax_rainbow.plot(x, y, color=col, linewidth=2.2, solid_capstyle='round', alpha=0.95)
        self.ax_rainbow.plot(0, 0.2, marker='o', color='black')
        self.ax_rainbow.set_xlim(-1.1, 1.1); self.ax_rainbow.set_ylim(-0.1, 1.5)
        self.ax_rainbow.set_title("Simulated coloured rainbow")

        # Table: write textual summary in the lower-left axes
        self.ax_table.clear()
        self.ax_table.axis('off')
        lines = ["λ(nm)  n(λ)     θ_min(deg)  ε_min(deg)  elev(deg)"]
        for lam, n, tmin, emin, e_deg in zip(wavelengths, n_vals, theta_mins, eps_mins, elev):
            lines.append(f"{lam:6.1f} {n:8.6f} {np.degrees(tmin):9.4f} {np.degrees(emin):9.4f} {e_deg:9.4f}")
        txt = "\n".join(lines[:20])  # show first ~20 lines to avoid clutter
        self.ax_table.text(0, 1.0, txt, fontfamily='monospace', fontsize=8, va='top')

        self.fig.tight_layout()
        self.canvas.draw()

    def generate_report(self):
        if self.current_data is None:
            messagebox.showerror("No data", "Compute the rainbow data first.")
            return
        outdir = filedialog.askdirectory(title="Choose folder to save report (will create subfolder 'descartes_report')") or "."
        outdir = os.path.join(outdir, "descartes_report")
        files = produce_report_and_pdf(self.current_data, outdir=outdir, try_latex_compile=True)
        self.report_files = files
        msg = "Report files created:\n" + "\n".join([f"{k}: {v}" for k,v in files.items() if v])
        messagebox.showinfo("Report generated", msg)

#run
def main():
    app = DescartesGUI()
    app.mainloop()

if __name__ == "__main__":
    main()
