#!/usr/bin/env python3
"""
descartes_rainbow_app.py

Full Tkinter + Matplotlib app implementing:
 - Task 1b: refractive index n(f) (405-790 nm) with colour mapping
 - Task 11a: epsilon/D(theta) vs theta (primary & secondary) + animated ray schematic
 - Task 11b: elevation (rainbow radius) vs frequency/wavelength
 - Task 11c: internal refraction phi vs frequency with critical angle and reflectances
 - Task 11d: sea-level sky simulation showing primary & secondary arcs for adjustable sun elevation

Usage:
    pip install numpy scipy matplotlib pillow
    python descartes_rainbow_app.py
"""

import os, math, threading
import numpy as np
from scipy.optimize import minimize_scalar
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import animation
from PIL import Image

c = 299792458.0

def refractive_index_water(frequency_hz):
    # Task 1b
    f_scaled = np.asarray(frequency_hz) / 1e15
    inv_sq = 1.731 - 0.261 * f_scaled**2
    n_squared = 1.0 + 1.0 / np.sqrt(inv_sq)
    return np.sqrt(n_squared)

def wavelength_nm_to_frequency_thz(lambda_nm):
    lam_m = lambda_nm * 1e-9
    freq_hz = c / lam_m
    return freq_hz / 1e12

def frequency_thz_to_wavelength_nm(freq_thz):
    f_hz = freq_thz * 1e12
    lam_m = c / f_hz
    return lam_m * 1e9

def phi_from_theta(theta, n):
    # phi = arcsin(sin theta / n)
    return np.arcsin(np.clip(np.sin(theta)/n, -1.0, 1.0))

def deviation_primary(theta, n):
    # D1 = pi + 2 theta - 4 phi
    phi = phi_from_theta(theta, n)
    return math.pi + 2.0*theta - 4.0*phi

def deviation_secondary(theta, n):
    # D2 = 2pi + 2 theta - 6 phi
    phi = phi_from_theta(theta, n)
    return 2.0*math.pi + 2.0*theta - 6.0*phi

def deviation_primary_array(thetas, n):
    phi = np.arcsin(np.clip(np.sin(thetas)/n, -1.0, 1.0))
    return np.pi + 2.0*thetas - 4.0*phi

def deviation_secondary_array(thetas, n):
    phi = np.arcsin(np.clip(np.sin(thetas)/n, -1.0, 1.0))
    return 2.0*np.pi + 2.0*thetas - 6.0*phi

def find_min_theta(fun_array, n):
    # minimise fun(theta,n)
    thetas = np.linspace(1e-6, np.pi/2 - 1e-6, 5000)
    vals = fun_array(thetas, n)
    idx = np.argmin(vals)
    theta0 = thetas[idx]
    lo = max(1e-8, theta0 - 0.01)
    hi = min(np.pi/2 - 1e-8, theta0 + 0.01)
    res = minimize_scalar(lambda th: fun_array(np.array([th]), n)[0], bounds=(lo, hi), method='bounded', options={'xatol':1e-12})
    if res.success:
        return res.x, res.fun
    else:
        return theta0, vals[idx]

def wavelength_to_rgb(lambda_nm):
    lam = float(lambda_nm)
    if lam < 380: lam = 380.0
    if lam > 780: lam = 780.0
    if lam <= 440:
        r = -(lam - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif lam <= 490:
        r = 0.0
        g = (lam - 440) / (490 - 440)
        b = 1.0
    elif lam <= 510:
        r = 0.0
        g = 1.0
        b = -(lam - 510) / (510 - 490)
    elif lam <= 580:
        r = (lam - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif lam <= 645:
        r = 1.0
        g = -(lam - 645) / (645 - 580)
        b = 0.0
    else:
        r = 1.0; g = 0.0; b = 0.0
    # intensity taper
    if lam < 420:
        factor = 0.3 + 0.7*(lam - 380)/(420 - 380)
    elif lam > 700:
        factor = 0.3 + 0.7*(780 - lam)/(780 - 700)
    else:
        factor = 1.0
    return (r*factor, g*factor, b*factor)

# Fresnel unpolarised reflectance
def fresnel_unpolarized(n1, n2, theta1):
    # theta1 in radians; returns reflectance R
    sin_theta2 = (n1 / n2) * math.sin(theta1) if n2 != 0 else 2.0
    if abs(sin_theta2) > 1.0:
        return 1.0
    theta2 = math.asin(sin_theta2)
    c1 = math.cos(theta1); c2 = math.cos(theta2)
    rs = (n1*c1 - n2*c2) / (n1*c1 + n2*c2)
    rp = (n2*c1 - n1*c2) / (n2*c1 + n1*c2)
    return 0.5*(rs*rs + rp*rp)

def compute_all(lam_min=405.0, lam_max=790.0, ns=61):
    lambdas = np.linspace(lam_min, lam_max, ns)
    freqs_thz = wavelength_nm_to_frequency_thz(lambdas)
    freqs_hz = freqs_thz * 1e12
    n_vals = refractive_index_water(freqs_hz)

    theta1_min = np.zeros_like(lambdas)
    D1_min = np.zeros_like(lambdas)
    alpha1_deg = np.zeros_like(lambdas)
    phi1_deg = np.zeros_like(lambdas)
    Rint1 = np.zeros_like(lambdas)
    approxI1 = np.zeros_like(lambdas)

    theta2_min = np.zeros_like(lambdas)
    D2_min = np.zeros_like(lambdas)
    alpha2_deg = np.zeros_like(lambdas)
    phi2_deg = np.zeros_like(lambdas)
    Rint2 = np.zeros_like(lambdas)
    approxI2 = np.zeros_like(lambdas)

    for i, (lam, n) in enumerate(zip(lambdas, n_vals)):
        t1, D1 = find_min_theta(deviation_primary_array, n)
        theta1_min[i] = t1; D1_min[i] = D1
        alpha1_deg[i] = math.degrees(math.pi - D1)
        phi1 = math.degrees(phi_from_theta(t1, n)); phi1_deg[i] = phi1
        R_entry = fresnel_unpolarized(1.0, n, t1)
        R_internal = fresnel_unpolarized(n, 1.0, math.radians(phi1))
        T_entry = max(0.0, 1.0 - R_entry)
        approxI1[i] = T_entry * R_internal * T_entry
        Rint1[i] = R_internal

        t2, D2 = find_min_theta(deviation_secondary_array, n)
        theta2_min[i] = t2; D2_min[i] = D2
        alpha2_deg[i] = math.degrees(D2 - math.pi)
        phi2 = math.degrees(phi_from_theta(t2, n)); phi2_deg[i] = phi2
        R_entry2 = fresnel_unpolarized(1.0, n, t2)
        R_internal2 = fresnel_unpolarized(n, 1.0, math.radians(phi2))
        T_entry2 = max(0.0, 1.0 - R_entry2)
        approxI2[i] = T_entry2 * (R_internal2**2) * T_entry2
        Rint2[i] = R_internal2

    phi_crit_deg = np.degrees(np.arcsin(np.clip(1.0 / n_vals, -1.0, 1.0)))

    return dict(
        lambdas=lambdas, freqs_thz=freqs_thz, n_vals=n_vals,
        theta1_min=theta1_min, D1_min=D1_min, alpha1_deg=alpha1_deg, phi1_deg=phi1_deg, Rint1=Rint1, approxI1=approxI1,
        theta2_min=theta2_min, D2_min=D2_min, alpha2_deg=alpha2_deg, phi2_deg=phi2_deg, Rint2=Rint2, approxI2=approxI2,
        phi_crit_deg=phi_crit_deg
    )

def compute_ray_path(theta_inc_deg, lam_nm, drop_radius=1.0, n=1.33, order=1):
    """
    Compute a ray path for a single incoming parallel ray at incidence angle theta_inc_deg (measured from
    the droplet normal at the incident point). We place the droplet centre at origin and drop on right.
    Returns a list of segments as ((x0,y0),(x1,y1), 'color', 'label') for plotting.
    order=1 for primary (1 internal reflection), 2 for secondary.
    """
    # We will compute intersections parametrically for a circle radius drop_radius center (0,0)
    # For clarity: we will launch a ray from left towards droplet with direction set by theta_inc (with respect to normal at impact).
    # Simpler approach for schematic: pick an incident point at circle boundary given polar angle psi such that the incoming ray direction forms
    # the specified angle with the normal. Derive geometry quickly for schematic; this is visual explanatory code not a full raytracer.
    psi = math.radians(90.0 - theta_inc_deg)
    x0 = -2.0*drop_radius
    y0 = 0.0 + 0.8*drop_radius*math.sin(psi)
    xi = drop_radius * math.cos(psi); yi = drop_radius * math.sin(psi)
    dir_vec = np.array([xi - x0, yi - y0])
    dir_vec = dir_vec / np.linalg.norm(dir_vec)
    segs = []
    #segment to intersection
    segs.append(((x0, y0), (xi, yi), wavelength_to_rgb(lam_nm), 'inc'))
    # refracted direction: apply Snell at incidence angle between dir and normal
    normal = np.array([xi, yi]) / np.sqrt(xi*xi + yi*yi)
    # angle between -dir_vec and normal
    cos_inc = -np.dot(dir_vec, normal)
    if cos_inc < -1: cos_inc = -1
    if cos_inc > 1: cos_inc = 1
    theta_inc = math.acos(cos_inc)
    # inside angle phi
    sin_phi = math.sin(theta_inc) / n
    if abs(sin_phi) > 1.0:
        # TIR: treat as reflection
        phi = None
        # reflect inside as mirror
        refl_dir = dir_vec - 2*np.dot(dir_vec, normal)*normal
        # interior segment
        p2 = (xi + refl_dir[0]*0.5, yi + refl_dir[1]*0.5)
        segs.append(((xi, yi), p2, wavelength_to_rgb(lam_nm), 'tir'))
        return segs
    phi = math.asin(sin_phi)
    # compute refracted vector inside using rotation approach
    # compute transmitted dir using Snell's vector form
    # rotate normal by phi to get interior direction approx:
    # find tangent vector
    tangent = np.array([-normal[1], normal[0]])
    # interior direction points roughly: -cos(phi)*normal + sin(phi)*tangent
    interior_dir = -math.cos(phi)*normal + math.sin(phi)*tangent
    interior_dir = interior_dir / np.linalg.norm(interior_dir)
    # next internal reflection(s):
    p_curr = np.array([xi, yi])
    dir_curr = interior_dir
    for refl in range(order):
        # intersect with circle again: param t satisfying |p_curr + t*dir_curr|^2 = R^2
        # solve quadratic for t>1e-6
        a = np.dot(dir_curr, dir_curr)
        b = 2*np.dot(p_curr, dir_curr)
        c0 = np.dot(p_curr, p_curr) - drop_radius**2
        disc = b*b - 4*a*c0
        if disc < 0:
            break
        tpos = (-b + math.sqrt(disc)) / (2*a)
        tneg = (-b - math.sqrt(disc)) / (2*a)
        t = tpos if tpos > 1e-6 else tneg
        if t is None:
            break
        p_hit = p_curr + t*dir_curr
        # append interior segment
        segs.append(((p_curr[0], p_curr[1]), (p_hit[0], p_hit[1]), wavelength_to_rgb(lam_nm), 'int'))
        # compute normal at hit
        normal2 = p_hit / np.linalg.norm(p_hit)
        # reflect dir
        dir_reflected = dir_curr - 2*np.dot(dir_curr, normal2)*normal2
        # set p_curr slightly inside
        p_curr = p_hit + 1e-6*dir_reflected
        dir_curr = dir_reflected / np.linalg.norm(dir_reflected)
    #intersect circle to exit
    # solve quadratic
    a = np.dot(dir_curr, dir_curr)
    b = 2*np.dot(p_curr, dir_curr)
    c0 = np.dot(p_curr, p_curr) - drop_radius**2
    disc = b*b - 4*a*c0
    if disc >= 0:
        tpos = (-b + math.sqrt(disc)) / (2*a)
        tneg = (-b - math.sqrt(disc)) / (2*a)
        t = tpos if tpos > 1e-6 else tneg
        if t is not None:
            p_exit = p_curr + t*dir_curr
            segs.append(((p_curr[0], p_curr[1]), (p_exit[0], p_exit[1]), wavelength_to_rgb(lam_nm), 'exit_in'))
            # refract out: compute angle with normal and apply Snell back (approx)
            normal_exit = p_exit / np.linalg.norm(p_exit)
            cos_in = -np.dot(dir_curr, normal_exit)
            if cos_in < -1: cos_in = -1
            if cos_in > 1: cos_in = 1
            theta_in = math.acos(cos_in)
            sin_out = n * math.sin(theta_in)
            if abs(sin_out) <= 1.0:
                # transmitted to outside, compute outgoing direction approx
                # compute outside vector using reverse relation with tangent
                # For schematic, reflect outgoing away by reversing interior refraction approximately
                outside_dir = -math.cos(theta_in)/1.0 * normal_exit + math.sin(theta_in) * np.array([-normal_exit[1], normal_exit[0]])
                p_far = (p_exit[0] + outside_dir[0]*2.0, p_exit[1] + outside_dir[1]*2.0)
                segs.append(((p_exit[0], p_exit[1]), (p_far[0], p_far[1]), wavelength_to_rgb(lam_nm), 'out'))
    return segs

#GUI
class App:
    def __init__(self, root):
        self.root = root
        root.title("Descartes Rainbow — Interactive modelling")
        root.geometry("1200x800")
        # menu frame
        menu = ttk.Frame(root)
        menu.pack(side='top', fill='x')
        # Notebook style menu buttons
        self.current_frame = None

        btn1 = ttk.Button(menu, text="Challenge 1b (n vs f)", command=self.show_1b)
        btn1.pack(side='left', padx=4, pady=4)
        btn2 = ttk.Button(menu, text="Challenge 11a (ε vs θ + schematic)", command=self.show_11a)
        btn2.pack(side='left', padx=4, pady=4)
        btn3 = ttk.Button(menu, text="Challenge 11b (elevation vs frequency)", command=self.show_11b)
        btn3.pack(side='left', padx=4, pady=4)
        btn4 = ttk.Button(menu, text="Challenge 11c (φ vs freq + critical)", command=self.show_11c)
        btn4.pack(side='left', padx=4, pady=4)
        btn5 = ttk.Button(menu, text="Challenge 11d (sea-level primary & secondary)", command=self.show_11d)
        btn5.pack(side='left', padx=4, pady=4)

        # container for pages
        self.container = ttk.Frame(root)
        self.container.pack(side='top', fill='both', expand=True)

        # precompute standard data
        self.data = compute_all()  # default lam range
        # build pages
        self.build_1b()
        self.build_11a()
        self.build_11b()
        self.build_11c()
        self.build_11d()

        # start showing 1b by default
        self.show_1b()

    def clear_container(self):
        for child in self.container.winfo_children():
            child.pack_forget()

    #1b again
    def build_1b(self):
        frame = ttk.Frame(self.container)
        self.page_1b = frame
        # figure
        fig, ax = plt.subplots(figsize=(8,5))
        self.fig1b = fig; self.ax1b = ax
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
        self.canvas1b = canvas
        # controls
        ctrl = ttk.Frame(frame)
        ctrl.pack(side='bottom', fill='x')
        ttk.Button(ctrl, text="Refresh data & plot", command=self.update_1b).pack(side='left', padx=6, pady=6)
        ttk.Button(ctrl, text="Save PNG", command=lambda: self.save_fig(self.fig1b)).pack(side='left', padx=6, pady=6)
        # explanation
        self.text1b = tk.Text(frame, height=6, wrap='word')
        self.text1b.pack(side='bottom', fill='x')
        self.text1b.insert('1.0', "Challenge 1b: Refractive index model (Task 1b). The Sellmeier-style formula used is:\n(n^2 - 1) = 1/sqrt(1.731 - 0.261*(f/1e15)^2). Code uses frequency (Hz) -> n.")
        self.update_1b()

    def update_1b(self):

        lam_min, lam_max = 405.0, 790.0
        ns = 200
        lambdas = np.linspace(lam_min, lam_max, ns)
        freqs_thz = wavelength_nm_to_frequency_thz(lambdas)
        n_vals = refractive_index_water(freqs_thz*1e12)
        self.data = compute_all(lam_min, lam_max, ns=ns)
        self.ax1b.clear()
        colors = [wavelength_to_rgb(l) for l in lambdas]
        self.ax1b.scatter(freqs_thz, n_vals, c=colors, s=8)
        # smooth interpolation line
        idx = np.argsort(freqs_thz)
        self.ax1b.plot(freqs_thz[idx], n_vals[idx], color='gray', alpha=0.3)
        self.ax1b.set_xlabel("Frequency (THz)")
        self.ax1b.set_ylabel("Refractive index n")
        self.ax1b.set_title("Challenge 1b: Refractive index of water vs Frequency (405-790 nm)")
        self.ax1b.grid(True)
        self.canvas1b.draw()

    #11a
    def build_11a(self):
        frame = ttk.Frame(self.container)
        self.page_11a = frame
        # top: two axes (left: epsilon vs theta; right: schematic animation)
        top = ttk.Frame(frame)
        top.pack(side='top', fill='both', expand=True)
        left_fig, self.ax11a = plt.subplots(figsize=(6,4))
        right_fig, self.ax_schem = plt.subplots(figsize=(4,4))
        self.fig11a = left_fig
        self.canvas11a = FigureCanvasTkAgg(left_fig, master=top)
        self.canvas11a.get_tk_widget().pack(side='left', fill='both', expand=True)
        self.canvas_schem = FigureCanvasTkAgg(right_fig, master=top)
        self.canvas_schem.get_tk_widget().pack(side='right', fill='both', expand=True)
        # controls
        ctrl = ttk.Frame(frame)
        ctrl.pack(side='bottom', fill='x')
        ttk.Label(ctrl, text="Wavelength (nm)").pack(side='left', padx=4)
        self.wlam_var = tk.DoubleVar(value=550.0)
        ttk.Scale(ctrl, from_=405.0, to=790.0, orient='horizontal', variable=self.wlam_var, command=self.update_schematic).pack(side='left', fill='x', expand=True, padx=4)
        ttk.Button(ctrl, text="Autoplay colours", command=self.autoplay_schematic).pack(side='left', padx=4)
        ttk.Button(ctrl, text="Refresh plot", command=self.update_11a).pack(side='left', padx=4)
        ttk.Button(ctrl, text="Save ε(θ) PNG", command=lambda: self.save_fig(self.fig11a)).pack(side='left', padx=4)
        ttk.Button(ctrl, text="Save schematic PNG", command=lambda: self.save_fig(self.canvas_schem.figure)).pack(side='left', padx=4)
        # explanation
        self.text11a = tk.Text(frame, height=8, wrap='word')
        self.text11a.pack(side='bottom', fill='x')
        self.text11a.insert('1.0', "Challenge 11a: Plot ε(θ) = D(θ) vs θ for primary and secondary. Minima correspond to rainbow angles (focusing). Equations used are in the code comments.")
        # initialise plots
        self.update_11a()
        self.update_schematic()
        self.anim_running = False

    def update_11a(self):
        # plot epsilon vs theta for primary & secondary using data
        data = self.data
        lambdas = data['lambdas']; n_vals = data['n_vals']
        # choose subset colors
        idxs = np.linspace(0, len(lambdas)-1, min(12, len(lambdas))).astype(int)
        thetas = np.linspace(0.0, np.pi/2 - 1e-6, 2000)
        self.ax11a.clear()
        for i in idxs:
            lam = lambdas[i]
            n = n_vals[i]
            D1 = deviation_primary_array(thetas, n)
            D2 = deviation_secondary_array(thetas, n)
            self.ax11a.plot(np.degrees(thetas), np.degrees(D1), color=wavelength_to_rgb(lam), alpha=0.6, label=f"{int(lam)}nm" if i==idxs[0] else "")
            self.ax11a.plot(np.degrees(thetas), np.degrees(D2), color=wavelength_to_rgb(lam), alpha=0.25, linestyle='--')
        # mark minima
        for i in idxs:
            lam = lambdas[i]
            n = n_vals[i]
            t1, D1 = find_min_theta(deviation_primary_array, n)
            t2, D2 = find_min_theta(deviation_secondary_array, n)
            self.ax11a.plot(np.degrees(t1), np.degrees(D1), 'o', color=wavelength_to_rgb(lam))
            self.ax11a.text(np.degrees(t1)+0.2, np.degrees(D1)+0.4, f"{int(lam)} nm", fontsize=7, color=wavelength_to_rgb(lam))
        self.ax11a.set_xlabel("Incident angle θ (deg)")
        self.ax11a.set_ylabel("Deviation D(θ) (deg)")
        self.ax11a.set_title("Challenge 11a: Deviation D(θ) for primary (solid) and secondary (dashed)")
        self.ax11a.set_xlim(0, 90); self.ax11a.set_ylim(0, 220)
        self.ax11a.grid(True)
        self.canvas11a.draw()

    def animate_schematic(self, frames=60):
        # animate the ray schematic for single wavelength cycling from violet->red
        fig = self.canvas_schem.figure
        ax = fig.axes[0]
        ax.clear()
        ax.set_xlim(-1.5,1.5); ax.set_ylim(-1.5,1.5); ax.set_aspect('equal'); ax.axis('off')
        drop = plt.Circle((0,0),1.0, edgecolor='k', facecolor='none', linewidth=1.0)
        ax.add_patch(drop)
        # choose wavelength sequence
        lam0 = self.wlam_var.get()
        lam_seq = np.linspace(405, 790, frames)
        line_artists = []
        def update(frame):
            ax.clear()
            ax.add_patch(plt.Circle((0,0),1.0, edgecolor='k', facecolor='none'))
            lam = lam_seq[frame]
            segs = compute_ray_path(30.0, lam, drop_radius=1.0, n=1.33, order=1)
            # draw segments
            for (p0,p1,col,tag) in segs:
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=col, linewidth=2)
            ax.set_title(f"Schematic: primary ray path at {int(lam)} nm")
            return line_artists
        self.anim = animation.FuncAnimation(fig, update, frames=len(lam_seq), interval=100, blit=False, repeat=True)
        self.canvas_schem.draw()

    def update_schematic(self, event=None):
        # draw a single schematic for selected wavelength, primary & secondary
        lam = self.wlam_var.get()
        fig = self.canvas_schem.figure
        ax = fig.axes[0] if fig.axes else fig.add_subplot(111)
        ax.clear()
        ax.set_xlim(-1.5,1.5); ax.set_ylim(-1.5,1.5); ax.set_aspect('equal'); ax.axis('off')
        drop = plt.Circle((0,0),1.0, edgecolor='black', facecolor='none')
        ax.add_patch(drop)
        # primary schematic
        segs = compute_ray_path(30.0, lam, drop_radius=1.0, n=1.33, order=1)
        for (p0,p1,col,tag) in segs:
            ax.plot([p0[0],p1[0]],[p0[1],p1[1]], color=col, linewidth=2)
            ax.scatter([p0[0]], [p0[1]], color=col, s=6)
        # add explanatory labels for angles (we'll annotate one angle)
        ax.text(-1.3,1.2,f"Wavelength: {int(lam)} nm", fontsize=9)
        ax.set_title("Schematic (primary) — single wavelength")
        self.canvas_schem.draw()

    def autoplay_schematic(self):
        # run animation in separate thread to avoid freezing Tk
        if hasattr(self, 'anim') and self.anim:
            try:
                self.anim.event_source.stop()
            except Exception:
                pass
        def run_anim():
            self.animate_schematic()
        t = threading.Thread(target=run_anim, daemon=True)
        t.start()

    #11b
    def build_11b(self):
        frame = ttk.Frame(self.container)
        self.page_11b = frame
        fig, ax = plt.subplots(figsize=(9,5))
        self.fig11b = fig; self.ax11b = ax
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
        ctrl = ttk.Frame(frame); ctrl.pack(side='bottom', fill='x')
        ttk.Button(ctrl, text="Plot elevation vs frequency", command=self.update_11b).pack(side='left', padx=6, pady=6)
        ttk.Button(ctrl, text="Save PNG", command=lambda: self.save_fig(self.fig11b)).pack(side='left', padx=6)
        self.text11b = tk.Text(frame, height=6); self.text11b.pack(side='bottom', fill='x')
        self.text11b.insert('1.0', "Challenge 11b: Plot rainbow elevation (angular radius) vs frequency using minimised D(θ).")
        self.update_11b()

    def update_11b(self):
        d = self.data
        freqs = d['freqs_thz']; lambdas = d['lambdas']
        a1 = d['alpha1_deg']; a2 = d['alpha2_deg']
        self.ax11b.clear()
        # plot lines
        self.ax11b.plot(freqs, a1, '-k', label='Primary α1')
        self.ax11b.plot(freqs, a2, '--k', label='Secondary α2')
        # coloured markers
        for lam,f,a in zip(lambdas, freqs, a1):
            self.ax11b.plot(f, a, 'o', color=wavelength_to_rgb(lam))
        # set axes and labels (match screenshot-like formatting)
        self.ax11b.set_xlabel("Frequency (THz)")
        self.ax11b.set_ylabel("Elevation / angular radius (deg)")
        self.ax11b.set_title("Challenge 11b: Rainbow elevation vs frequency (coloured by wavelength)")
        self.ax11b.grid(True)
        self.ax11b.legend()
        self.canvas11b = FigureCanvasTkAgg(self.fig11b, master=self.page_11b)
        self.canvas11b.draw()
        self.canvas11b.get_tk_widget().pack_forget()  # avoid duplicate
        self.canvas11b.get_tk_widget().pack(side='top', fill='both', expand=True)

    # 11c
    def build_11c(self):
        frame = ttk.Frame(self.container)
        self.page_11c = frame
        fig, ax = plt.subplots(figsize=(9,5))
        self.fig11c = fig; self.ax11c = ax
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
        ctrl = ttk.Frame(frame); ctrl.pack(side='bottom', fill='x')
        ttk.Button(ctrl, text="Plot φ vs freq & critical angle", command=self.update_11c).pack(side='left', padx=6, pady=6)
        ttk.Button(ctrl, text="Save PNG", command=lambda: self.save_fig(self.fig11c)).pack(side='left', padx=6)
        self.text11c = tk.Text(frame, height=6); self.text11c.pack(side='bottom', fill='x')
        self.text11c.insert('1.0', "Challenge 11c: Plot internal refraction angle φ at minima vs frequency and overlay critical angle φ_c = arcsin(1/n). Also plot reflectance estimates.")
        self.update_11c()

    def update_11c(self):
        d = self.data
        freqs = d['freqs_thz']; lambdas = d['lambdas']
        phi1 = d['phi1_deg']; phi2 = d['phi2_deg']; phi_crit = d['phi_crit_deg']
        self.ax11c.clear()
        self.ax11c.plot(freqs, phi1, '-b', label='Primary φ (deg)')
        self.ax11c.plot(freqs, phi2, '-g', label='Secondary φ (deg)')
        self.ax11c.plot(freqs, phi_crit, '--r', label='Critical angle φ_c (deg)')
        self.ax11c.set_xlabel("Frequency (THz)")
        self.ax11c.set_ylabel("Angle φ (deg)")
        self.ax11c.set_title("Challenge 11c: Internal φ and critical angle vs frequency")
        self.ax11c.grid(True); self.ax11c.legend()
        # show approximate R_internal lines in inset
        self.ax11c_twin = self.ax11c.twinx()
        self.ax11c_twin.plot(freqs, d['Rint1'], ':', color='purple', label='R_internal primary')
        self.ax11c_twin.plot(freqs, d['Rint2'], ':', color='magenta', label='R_internal secondary')
        self.ax11c_twin.set_ylabel('Reflectance (approx)')
        self.canvas11c = FigureCanvasTkAgg(self.fig11c, master=self.page_11c)
        self.canvas11c.draw()
        self.canvas11c.get_tk_widget().pack_forget()
        self.canvas11c.get_tk_widget().pack(side='top', fill='both', expand=True)

    # `11d`
    def build_11d(self):
        frame = ttk.Frame(self.container)
        self.page_11d = frame
        # sky figure
        fig, ax = plt.subplots(figsize=(9,6))
        self.fig11d = fig; self.ax11d = ax
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
        ctrl = ttk.Frame(frame); ctrl.pack(side='bottom', fill='x')
        ttk.Label(ctrl, text="Sun elevation (deg)").pack(side='left')
        self.sun_elev_var = tk.DoubleVar(value=5.0)
        ttk.Scale(ctrl, from_=0.0, to=45.0, variable=self.sun_elev_var, orient='horizontal', command=lambda e: self.update_11d()).pack(side='left', fill='x', expand=True, padx=6)
        ttk.Button(ctrl, text="Render primary+secondary arcs", command=self.update_11d).pack(side='left', padx=6)
        ttk.Button(ctrl, text="Save PNG", command=lambda: self.save_fig(self.fig11d)).pack(side='left', padx=6)
        self.text11d = tk.Text(frame, height=6); self.text11d.pack(side='bottom', fill='x')
        self.text11d.insert('1.0', "Challenge 11d: Sea-level view: primary & secondary rainbow arcs computed from α1(λ) and α2(λ) and placed relative to sun elevation.")
        self.update_11d()

    def update_11d(self):
        d = self.data
        lambdas = d['lambdas']; alpha1 = d['alpha1_deg']; alpha2 = d['alpha2_deg']
        sun_h = float(self.sun_elev_var.get())
        ax = self.ax11d
        ax.clear()
        ax.set_xlim(-1.2,1.2); ax.set_ylim(-0.1,1.4); ax.axis('off')
        ax.set_title(f"Sea-level rainbow (sun elevation = {sun_h:.1f}°)")
        # map α - sun_h to arc radii
        top_elev = alpha1 - sun_h
        visible = top_elev > 0.1
        if not np.any(visible):
            ax.text(0,0.6,"Rainbow below horizon at this sun elevation", ha='center')
            self.canvas11d.draw()
            return
        tmin = np.min(top_elev[visible]); tmax = np.max(top_elev[visible])
        rmin, rmax = 0.25, 1.0
        rs1 = rmin + (top_elev - tmin) * (rmax - rmin) / max(1e-6,(tmax - tmin))
        theta = np.linspace(-np.pi/3, np.pi/3, 300)
        # primary: inner (violet) to outer (red)
        for lam, r in zip(lambdas, rs1):
            col = wavelength_to_rgb(lam)
            ax.plot(r*np.cos(theta), r*np.sin(theta)+0.2, color=col, linewidth=2.0)
        # secondary offset bit larger
        top2 = alpha2 - sun_h
        if np.any(top2 > 0.1):
            tmin2 = np.min(top2[top2>0.1]); tmax2 = np.max(top2[top2>0.1])
            rs2 = rmin + (top2 - tmin2) * (rmax - rmin) / max(1e-6,(tmax2 - tmin2))
            for lam, r in zip(lambdas, rs2):
                col = wavelength_to_rgb(lam)
                ax.plot((r+0.07)*np.cos(theta), (r+0.07)*np.sin(theta)+0.2, color=col, linewidth=1.6, alpha=0.9, linestyle='--')
        # label antisolar point
        ax.scatter(0,0.2, color='k'); ax.text(0.02,0.1,'Antisolar (observer)', fontsize=9)
        self.canvas11d.draw()

    def build_11d(self):
        # Create figure and axis
        fig11d, ax11d = plt.subplots(figsize=(6, 4))
        ax11d.set_title("Rainbow Elevation vs Solar Elevation")
        ax11d.set_xlabel("Solar Elevation (degrees)")
        ax11d.set_ylabel("Rainbow Elevation (degrees)")

        # Store ax and fig for later updates
        self.fig11d = fig11d
        self.ax11d = ax11d

        # Create the Tkinter canvas
        self.canvas11d = FigureCanvasTkAgg(fig11d, master=self.frame_11d)
        self.canvas11d.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initial draw
        self.update_11d()

    def update_11d(self):
        self.ax11d.clear()
        self.ax11d.set_title("Rainbow Elevation vs Solar Elevation")
        self.ax11d.set_xlabel("Solar Elevation (degrees)")
        self.ax11d.set_ylabel("Rainbow Elevation (degrees)")

        # Example plot — replace with your computed data
        solar_elevations = np.linspace(0, 45, 100)
        rainbow_elevations_primary = 42 - solar_elevations  # placeholder
        rainbow_elevations_secondary = 50 - solar_elevations  # placeholder

        self.ax11d.plot(solar_elevations, rainbow_elevations_primary, label="Primary Rainbow")
        self.ax11d.plot(solar_elevations, rainbow_elevations_secondary, label="Secondary Rainbow")
        self.ax11d.legend()

        self.canvas11d.draw()

    def save_fig(self, fig):
        if isinstance(fig, plt.Figure):
            f = fig
        else:
            f = fig
        fname = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG','*.png')])
        if not fname:
            return
        f.savefig(fname, dpi=200, bbox_inches='tight')
        messagebox.showinfo("Saved", f"Saved to {fname}")

#beta
    def show_1b(self):
        self.clear_current()
        self.page_1b.pack(fill='both', expand=True)

    def show_11a(self):
        self.clear_current()
        self.page_11a.pack(fill='both', expand=True)

    def show_11b(self):
        self.clear_current()
        self.page_11b.pack(fill='both', expand=True)

    def show_11c(self):
        self.clear_current()
        self.page_11c.pack(fill='both', expand=True)

    def show_11d(self):
        self.clear_current()
        self.page_11d.pack(fill='both', expand=True)

    def clear_current(self):
        for child in self.container.winfo_children():
            child.pack_forget()
#run
def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == '__main__':
    main()
