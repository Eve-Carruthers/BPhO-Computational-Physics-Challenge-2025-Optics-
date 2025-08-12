#Fix convex/concave lens stuff

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy import ndimage

# image loading and conversions
def load_image_float(path, max_dim=1200):

    img = Image.open(path).convert("RGBA")
    if max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        newsize = (int(img.size[0]*scale), int(img.size[1]*scale))
        img = img.resize(newsize, Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    return arr

def save_image_float(arr, path):
    a = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(a).save(path)
def reflect_across_horizontal_mirror(img, y_mirror_plot):
    H, W = img.shape[0], img.shape[1]
    x_coords = np.arange(W)
    y_coords = np.arange(H)
    X, Yplot = np.meshgrid(x_coords, y_coords)
    Ys_src_plot = 2.0 * y_mirror_plot - Yplot
    Xs_src_plot = X
    src_rows = (H - 1) - Ys_src_plot
    src_cols = Xs_src_plot
    coords = np.vstack([src_rows.ravel(), src_cols.ravel()])
    out = np.zeros_like(img)
    for c in range(img.shape[2]):
        sampled = ndimage.map_coordinates(img[:,:,c], coords, order=1, mode='constant', cval=0.0)
        out[:,:,c] = sampled.reshape((H, W))
    return out

def thin_lens_map_image(img, f, u, background_value=1.0):

    if abs(u - f) < 1e-12:
        raise ValueError("Object at focal plane (u == f): image is formed at infinity and cannot be rendered.")

    v = 1.0 / (1.0/f - 1.0/u)   # image distance
    m = - v / u                 # transverse magnification
    H, W = img.shape[0], img.shape[1]
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0

    #image-plane in plot coords
    x_coords = np.arange(W)
    y_coords = np.arange(H)
    X_img, Y_img_plot = np.meshgrid(x_coords, y_coords)
    x_img_rel = X_img - cx
    y_img_rel = Y_img_plot - cy

    if abs(m) < 1e-12:
        # magnification ~ 0: degenerate
        out = np.ones_like(img) * background_value
        return out, v, m

    # inverse mapping
    x_obj_rel = x_img_rel / m
    y_obj_rel = y_img_rel / m

    # convert to source array indices
    src_cols = cx + x_obj_rel
    src_rows = (H - 1) - (cy + y_obj_rel)
    coords = np.vstack([src_rows.ravel(), src_cols.ravel()])

    out = np.ones_like(img) * background_value
    for c in range(img.shape[2]):
        sampled = ndimage.map_coordinates(img[:,:,c], coords, order=1, mode='constant', cval=background_value)
        out[:,:,c] = sampled.reshape((H, W))
    return out, v, m

def spherical_mirror_map_image(img, R, u, background_value=1.0):
    """
    Map an object image using spherical mirror with radius R.
    For a spherical mirror, focal length f = R/2 (signed), so we reuse thin_lens_map_image with f = R/2.
    Positive R = concave (f positive), negative R = convex (f negative).
    """
    f = R / 2.0
    return thin_lens_map_image(img, f, u, background_value=background_value)

def anamorphic_map(img, circle_center, circle_radius_px, Rf, theta_span=np.pi/2, theta_offset=0.0, background_value=1.0):
    """
    Task 10 Anamorphic mapping: map source unit circle -> sector (inverse mapping).
    - circle_center: (cx, cy) in plotting coordinates (origin lower)
    - circle_radius_px: radius in pixels of the unit circle region
    - Rf: radial scaling factor for target sector (Rf>1 enlarges)
    - theta_span: angular width of sector (radians)
    - theta_offset: rotation of sector center
    Returns mapped image same shape.
    """
    H, W = img.shape[0], img.shape[1]
    cx, cy = circle_center
    # target grid
    x_coords = np.arange(W)
    y_coords = np.arange(H)
    X_t, Y_t_plot = np.meshgrid(x_coords, y_coords)
    xr = X_t - cx
    yr = Y_t_plot - cy
    r_t = np.sqrt(xr**2 + yr**2) / circle_radius_px  # normalized target radius
    theta_t = np.arctan2(yr, xr)  # -pi..pi

    # Inverse mapping
    theta_src = (theta_t - theta_offset) * (2.0 * np.pi / theta_span)
    r_src = r_t / Rf

    # Only valid mapping for r_src in [0,1] (i.e., original unit circle)
    # >Convert back to source plot coords
    Xs_plot = cx + (r_src * circle_radius_px) * np.cos(theta_src)
    Ys_plot = cy + (r_src * circle_radius_px) * np.sin(theta_src)

    src_rows = (H - 1) - Ys_plot
    src_cols = Xs_plot
    coords = np.vstack([src_rows.ravel(), src_cols.ravel()])
    out = np.ones_like(img) * background_value
    for c in range(img.shape[2]):
        sampled = ndimage.map_coordinates(img[:,:,c], coords, order=1, mode='constant', cval=background_value)
        out[:,:,c] = sampled.reshape((H, W))
    return out
# Tkinter GUI

class BPhOGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BPhO Imaging Tasks 5–10 — Tkinter GUI")
        self.geometry("1200x800")

        # store loaded image array (RGBA float [0,1])
        self.img = None
        self.img_path = None

        # top frame: file open & save buttons
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        open_btn = ttk.Button(top, text="Open Image", command=self.open_image)
        open_btn.pack(side=tk.LEFT, padx=4)

        save_btn = ttk.Button(top, text="Save Current View", command=self.save_current_view)
        save_btn.pack(side=tk.LEFT, padx=4)

        help_btn = ttk.Button(top, text="Help / Physics", command=self.show_help)
        help_btn.pack(side=tk.LEFT, padx=4)

        # Notebook (tabs) for Tasks 5-10
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create tabs
        self.tab5 = ttk.Frame(self.notebook); self.notebook.add(self.tab5, text="Task 5: Mirror")
        self.tab6 = ttk.Frame(self.notebook); self.notebook.add(self.tab6, text="Task 6/7: Thin Lens")
        self.tab8 = ttk.Frame(self.notebook); self.notebook.add(self.tab8, text="Task 8: Concave Mirror")
        self.tab9 = ttk.Frame(self.notebook); self.notebook.add(self.tab9, text="Task 9: Convex Mirror")
        self.tab10 = ttk.Frame(self.notebook); self.notebook.add(self.tab10, text="Task 10: Anamorphosis")

        # Build UI for each tab
        self.build_tab5()
        self.build_tab6_7()
        self.build_tab8()
        self.build_tab9()
        self.build_tab10()

        # status bar
        self.status_var = tk.StringVar(value="Open an image to begin.")
        status = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status.pack(side=tk.BOTTOM, fill=tk.X)

        # keep current displayed figure reference for saving
        self.current_canvas = None
        self.current_fig = None

    def open_image(self):
        filetypes = [("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Select object image", filetypes=filetypes)
        if not path:
            return
        try:
            img = load_image_float(path, max_dim=1200)
        except Exception as e:
            messagebox.showerror("Open Image Error", str(e))
            return
        self.img = img
        self.img_path = path
        self.status_var.set(f"Loaded image: {os.path.basename(path)}  (shape: {img.shape[1]} x {img.shape[0]})")
        # render initial views in all tabs
        self.update_tab5()
        self.update_tab6_7()
        self.update_tab8()
        self.update_tab9()
        self.update_tab10()

    def save_current_view(self):
        if self.current_fig is None:
            messagebox.showinfo("Save", "No current view to save. Display a task first.")
            return
        path = filedialog.asksaveasfilename(title="Save current view as PNG", defaultextension=".png",
                                            filetypes=[("PNG image","*.png")])
        if not path:
            return
        # Save as a PNG
        try:
            self.current_fig.savefig(path, dpi=150, bbox_inches='tight')
            messagebox.showinfo("Saved", f"Saved current view to {path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def show_help(self):
        help_text = (
            "Physics and conventions used:\n\n"
            "Thin lens / spherical mirror formula (Gaussian):\n"
            "  1/u + 1/v = 1/f\n"
            "Transverse magnification: m = -v/u\n\n"
            "Sign conventions: u>0 object left of lens; v>0 image right of lens. For mirrors f=R/2 (signed).\n\n"
            "Mapping: for an object plane at distance u, transverse coords scale by m. "
            "This app uses inverse mapping and bilinear interpolation (scipy.ndimage.map_coordinates) to resample."
        )
        messagebox.showinfo("Help — Physics & Conventions", help_text)

#Task 5
    def build_tab5(self):
        frm = self.tab5
        # left side: figure canvas
        left = ttk.Frame(frm)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        fig, ax = plt.subplots(figsize=(6,5))
        canvas = FigureCanvasTkAgg(fig, master=left)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.tab5_fig = fig; self.tab5_ax = ax; self.tab5_canvas = canvas

        # right side: controls
        right = ttk.Frame(frm, width=300)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Label(right, text="Plane mirror controls").pack(padx=6, pady=6)
        ttk.Label(right, text="Mirror vertical position (pixels)").pack(padx=6, pady=(8,0))
        self.mirror_y_var = tk.DoubleVar(value=100.0)
        mirror_slider = ttk.Scale(right, from_=0.0, to=600.0, orient=tk.HORIZONTAL, variable=self.mirror_y_var, command=lambda e: self.update_tab5())
        mirror_slider.pack(fill=tk.X, padx=6, pady=6)
        ttk.Button(right, text="Center mirror", command=lambda: self.set_mirror_center()).pack(padx=6, pady=6)

    def set_mirror_center(self):
        if self.img is None:
            return
        H = self.img.shape[0]; cy=(H-1)/2.0
        self.mirror_y_var.set(cy)
        self.update_tab5()

    def update_tab5(self):
        fig = self.tab5_fig; ax = self.tab5_ax; canvas = self.tab5_canvas
        ax.clear()
        if self.img is None:
            # placeholder
            ax.text(0.5, 0.5, "Open an image to run Task 5", ha='center', va='center', transform=ax.transAxes)
            canvas.draw()
            return
        H, W = self.img.shape[0], self.img.shape[1]
        # read mirror slider (value is in pixels, plotting coords origin lower)
        y_mirror = float(self.mirror_y_var.get())
        # clamp mirror into image
        if y_mirror < 0.0: y_mirror = 0.0
        if y_mirror > H-1: y_mirror = H-1
        # compute reflection
        reflected = reflect_across_horizontal_mirror(self.img, y_mirror)
        # plot: show reflected image, overlay mirror line and annotations
        ax.imshow(reflected, origin='lower', extent=[0, W, 0, H])
        ax.set_xlim(0, W); ax.set_ylim(0, H)
        ax.set_xlabel("x (pixels)"); ax.set_ylabel("y (pixels)")
        ax.set_title("Task 5 — Virtual image in a plane mirror")
        # mirror line
        ax.axhline(y_mirror, color='red', linestyle='--', linewidth=1.4)
        ax.annotate("Mirror line (y = {:.1f} px)".format(y_mirror), xy=(W*0.02, y_mirror+5),
                    color='red', fontsize=9)
        # optical axis center marker
        ax.plot((W-1)/2.0, (H-1)/2.0, marker='+', color='white')
        ax.annotate("optical axis center", xy=((W-1)/2.0, (H-1)/2.0+10), color='white', fontsize=8)
        canvas.draw()

        # set current fig/canvas for saving externally
        self.current_canvas = canvas
        self.current_fig = fig

#Task 6/7
    def build_tab6_7(self):
        frm = self.tab6
        left = ttk.Frame(frm)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        fig, ax = plt.subplots(figsize=(6,5))
        canvas = FigureCanvasTkAgg(fig, master=left)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.tab6_fig = fig; self.tab6_ax = ax; self.tab6_canvas = canvas

        right = ttk.Frame(frm, width=320)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Label(right, text="Thin lens controls (Task 6 & 7)").pack(padx=6, pady=6)
        ttk.Label(right, text="Focal length f (px)").pack(anchor=tk.W, padx=6)
        self.f_var = tk.DoubleVar(value=120.0)
        f_scale = ttk.Scale(right, from_=10.0, to=600.0, orient=tk.HORIZONTAL, variable=self.f_var, command=lambda e: self.update_tab6_7())
        f_scale.pack(fill=tk.X, padx=6, pady=4)

        ttk.Label(right, text="Object distance u (px)").pack(anchor=tk.W, padx=6)
        self.u_var = tk.DoubleVar(value=240.0)
        u_scale = ttk.Scale(right, from_=10.0, to=1200.0, orient=tk.HORIZONTAL, variable=self.u_var, command=lambda e: self.update_tab6_7())
        u_scale.pack(fill=tk.X, padx=6, pady=4)

        ttk.Button(right, text="Set u = 1.5 f (real image)", command=lambda: self.set_u_to_fraction(1.5)).pack(fill=tk.X, padx=6, pady=(8,2))
        ttk.Button(right, text="Set u = 0.8 f (virtual image)", command=lambda: self.set_u_to_fraction(0.8)).pack(fill=tk.X, padx=6, pady=(2,8))

        # info area
        self.lens_info = tk.StringVar(value="v = ?, m = ?")
        ttk.Label(right, textvariable=self.lens_info, wraplength=300).pack(padx=6, pady=6)

    def set_u_to_fraction(self, frac):
        f = float(self.f_var.get())
        self.u_var.set(f * frac)
        self.update_tab6_7()

    def update_tab6_7(self):
        fig = self.tab6_fig; ax = self.tab6_ax; canvas = self.tab6_canvas
        ax.clear()
        if self.img is None:
            ax.text(0.5, 0.5, "Open an image to run Tasks 6 & 7", ha='center', va='center', transform=ax.transAxes)
            canvas.draw()
            return
        f = float(self.f_var.get())
        u = float(self.u_var.get())
        try:
            mapped, v, m = thin_lens_map_image(self.img, f, u, background_value=1.0)
            H, W = self.img.shape[0], self.img.shape[1]
            ax.imshow(mapped, origin='lower', extent=[0, W, 0, H])
            ax.set_xlim(0, W); ax.set_ylim(0, H)
            ax.set_xlabel("x (pixels)"); ax.set_ylabel("y (pixels)")
            title = f"Thin lens (ideal): f={f:.2f} px, u={u:.2f} px, v={v:.2f} px, m={m:.3f}"
            ax.set_title("Task 6/7 — " + title)
            # annotate inversion/u-f relation
            ax.annotate("Optical axis (centre)", xy=((W-1)/2.0, (H-1)/2.0+10), color='white', fontsize=8)
            ax.axvline((W-1)/2.0, color='white', linestyle='--', linewidth=0.8)
            # show focal points on axis: lens at x=0 in physics; here x axis is transverse only. We'll annotate f value numerically.
            self.lens_info.set(f"Thin lens formula: 1/u + 1/v = 1/f\nComputed: v = {v:.3f} px, m = {m:.6f}")
        except ValueError as e:
            # object at focal plane
            H, W = self.img.shape[0], self.img.shape[1]
            ax.imshow(np.ones_like(self.img), origin='lower', extent=[0, W, 0, H])
            ax.set_title("Object at focal plane — image at infinity (cannot render)")
            self.lens_info.set(str(e))
        canvas.draw()

        self.current_fig = fig
        self.current_canvas = canvas

#Task 8
    def build_tab8(self):
        frm = self.tab8
        left = ttk.Frame(frm)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        fig, ax = plt.subplots(figsize=(6,5))
        canvas = FigureCanvasTkAgg(fig, master=left)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.tab8_fig = fig; self.tab8_ax = ax; self.tab8_canvas = canvas

        right = ttk.Frame(frm, width=300)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Label(right, text="Concave mirror controls (Task 8)").pack(padx=6, pady=6)
        ttk.Label(right, text="Radius of curvature R (px)").pack(anchor=tk.W, padx=6)
        self.R8_var = tk.DoubleVar(value=240.0)
        R8_scale = ttk.Scale(right, from_=20.0, to=1200.0, orient=tk.HORIZONTAL, variable=self.R8_var, command=lambda e: self.update_tab8())
        R8_scale.pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(right, text="Object distance u (px)").pack(anchor=tk.W, padx=6)
        self.u8_var = tk.DoubleVar(value=400.0)
        u8_scale = ttk.Scale(right, from_=10.0, to=2000.0, orient=tk.HORIZONTAL, variable=self.u8_var, command=lambda e: self.update_tab8())
        u8_scale.pack(fill=tk.X, padx=6, pady=4)
        self.mirror8_info = tk.StringVar(value="v = ?, m = ?")
        ttk.Label(right, textvariable=self.mirror8_info, wraplength=300).pack(padx=6, pady=6)

    def update_tab8(self):
        fig = self.tab8_fig; ax = self.tab8_ax; canvas = self.tab8_canvas
        ax.clear()
        if self.img is None:
            ax.text(0.5, 0.5, "Open an image to run Task 8", ha='center', va='center', transform=ax.transAxes)
            canvas.draw()
            return
        R = float(self.R8_var.get())
        u = float(self.u8_var.get())
        try:
            mapped, v, m = spherical_mirror_map_image(self.img, R, u)
            H, W = self.img.shape[0], self.img.shape[1]
            ax.imshow(mapped, origin='lower', extent=[0, W, 0, H])
            ax.set_title(f"Task 8 — Concave mirror (R={R:.2f} px, f={R/2.0:.2f} px, u={u:.2f} px, v={v:.2f} px, m={m:.3f})")
            ax.set_xlabel("x (pixels)"); ax.set_ylabel("y (pixels)")
            self.mirror8_info.set(f"Mirror formula used: 1/u + 1/v = 1/f (f = R/2). Computed v={v:.3f} px, m={m:.6f}")
        except ValueError as e:
            ax.imshow(np.ones_like(self.img), origin='lower')
            ax.set_title("Object at focal plane — image at infinity (cannot render)")
            self.mirror8_info.set(str(e))
        canvas.draw()
        self.current_fig = fig; self.current_canvas = canvas

# Task 9
    def build_tab9(self):
        frm = self.tab9
        left = ttk.Frame(frm)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        fig, ax = plt.subplots(figsize=(6,5))
        canvas = FigureCanvasTkAgg(fig, master=left)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.tab9_fig = fig; self.tab9_ax = ax; self.tab9_canvas = canvas

        right = ttk.Frame(frm, width=300)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Label(right, text="Convex mirror controls (Task 9)").pack(padx=6, pady=6)
        ttk.Label(right, text="Radius of curvature R (px) — negative for convex").pack(anchor=tk.W, padx=6)
        self.R9_var = tk.DoubleVar(value=-240.0)
        R9_scale = ttk.Scale(right, from_=-2000.0, to=-20.0, orient=tk.HORIZONTAL, variable=self.R9_var, command=lambda e: self.update_tab9())
        R9_scale.pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(right, text="Object distance u (px)").pack(anchor=tk.W, padx=6)
        self.u9_var = tk.DoubleVar(value=400.0)
        u9_scale = ttk.Scale(right, from_=10.0, to=2000.0, orient=tk.HORIZONTAL, variable=self.u9_var, command=lambda e: self.update_tab9())
        u9_scale.pack(fill=tk.X, padx=6, pady=4)
        self.mirror9_info = tk.StringVar(value="v = ?, m = ?")
        ttk.Label(right, textvariable=self.mirror9_info, wraplength=300).pack(padx=6, pady=6)

    def update_tab9(self):
        fig = self.tab9_fig; ax = self.tab9_ax; canvas = self.tab9_canvas
        ax.clear()
        if self.img is None:
            ax.text(0.5, 0.5, "Open an image to run Task 9", ha='center', va='center', transform=ax.transAxes)
            canvas.draw()
            return
        R = float(self.R9_var.get())
        u = float(self.u9_var.get())
        try:
            mapped, v, m = spherical_mirror_map_image(self.img, R, u)
            H, W = self.img.shape[0], self.img.shape[1]
            ax.imshow(mapped, origin='lower', extent=[0, W, 0, H])
            ax.set_title(f"Task 9 — Convex mirror (R={R:.2f} px -> f={R/2.0:.2f} px) u={u:.2f} px, v={v:.2f} px, m={m:.3f}")
            ax.set_xlabel("x (pixels)"); ax.set_ylabel("y (pixels)")
            self.mirror9_info.set(f"Mirror formula used: 1/u + 1/v = 1/f (f = R/2). Computed v={v:.3f} px, m={m:.6f}")
        except ValueError as e:
            ax.imshow(np.ones_like(self.img), origin='lower')
            ax.set_title("Object at focal plane — image at infinity (cannot render)")
            self.mirror9_info.set(str(e))
        canvas.draw()
        self.current_fig = fig; self.current_canvas = canvas

# Task 10
    def build_tab10(self):
        frm = self.tab10
        left = ttk.Frame(frm)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        fig, ax = plt.subplots(figsize=(6,6))
        canvas = FigureCanvasTkAgg(fig, master=left)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.tab10_fig = fig; self.tab10_ax = ax; self.tab10_canvas = canvas

        right = ttk.Frame(frm, width=300)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Label(right, text="Anamorphic mapping controls (Task 10)").pack(padx=6, pady=6)
        ttk.Label(right, text="Circle radius (px)").pack(anchor=tk.W, padx=6)
        self.circle_r_var = tk.DoubleVar(value=120.0)
        cr_scale = ttk.Scale(right, from_=10.0, to=800.0, orient=tk.HORIZONTAL, variable=self.circle_r_var, command=lambda e: self.update_tab10())
        cr_scale.pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(right, text="Rf (radial scale factor)").pack(anchor=tk.W, padx=6)
        self.Rf_var = tk.DoubleVar(value=1.6)
        Rf_scale = ttk.Scale(right, from_=0.2, to=3.0, orient=tk.HORIZONTAL, variable=self.Rf_var, command=lambda e: self.update_tab10())
        Rf_scale.pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(right, text="Sector angular span (degrees)").pack(anchor=tk.W, padx=6)
        self.theta_deg_var = tk.DoubleVar(value=90.0)
        theta_scale = ttk.Scale(right, from_=10.0, to=360.0, orient=tk.HORIZONTAL, variable=self.theta_deg_var, command=lambda e: self.update_tab10())
        theta_scale.pack(fill=tk.X, padx=6, pady=4)

        ttk.Button(right, text="Set circle base = lower-center", command=lambda: self.set_circle_base()).pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(right, text="Note: red star marks base (center of cylinder)").pack(padx=6, pady=(2,6))

    def set_circle_base(self):
        # set circle to be near base of object (lower centre)
        if self.img is None:
            return
        H = self.img.shape[0]; W = self.img.shape[1]
        # center x at centre, y at 20% up from bottom (as earlier spec)
        cx = (W - 1) / 2.0
        cy = H * 0.2
        # set radius approx quarter image width
        r = min(W, H) * 0.25
        self.circle_r_var.set(r)
        # store center in instance for update
        self.circle_center = (cx, cy)
        self.update_tab10()

    def update_tab10(self):
        fig = self.tab10_fig; ax = self.tab10_ax; canvas = self.tab10_canvas
        ax.clear()
        if self.img is None:
            ax.text(0.5, 0.5, "Open an image to run Task 10", ha='center', va='center', transform=ax.transAxes)
            canvas.draw()
            return
        H, W = self.img.shape[0], self.img.shape[1]
        # default circle center at lower-center unless previously set
        if not hasattr(self, 'circle_center'):
            cx = (W - 1) / 2.0
            cy = H * 0.2
            self.circle_center = (cx, cy)
        circle_r = float(self.circle_r_var.get())
        Rf = float(self.Rf_var.get())
        theta_span = np.deg2rad(float(self.theta_deg_var.get()))
        mapped = anamorphic_map(self.img, self.circle_center, circle_r, Rf, theta_span, theta_offset=0.0)
        ax.imshow(mapped, origin='lower', extent=[0, W, 0, H])
        ax.set_xlim(0, W); ax.set_ylim(0, H)
        ax.set_xlabel("x (pixels)"); ax.set_ylabel("y (pixels)")
        ax.set_title(f"Task 10 — Anamorphic mapping (circle r={circle_r:.1f}px; Rf={Rf:.2f}; span={np.rad2deg(theta_span):.0f}°)")
        # draw circle in overlay where source region is
        circ = Circle(self.circle_center, circle_r, fill=False, edgecolor='yellow', linewidth=1.2)
        ax.add_patch(circ)
        # draw base red star
        ax.plot(self.circle_center[0], self.circle_center[1], marker='*', color='red', markersize=12)
        ax.annotate("base (red star)", xy=(self.circle_center[0]+8, self.circle_center[1]+8), color='red', fontsize=9)
        canvas.draw()
        self.current_fig = fig; self.current_canvas = canvas

#Run!
def main():
    app = BPhOGUI()
    app.mainloop()

if __name__ == "__main__":
    main()

