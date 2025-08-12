import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

def get_angles(x, y, Y, L):
    dx1 = x
    dy1 = y
    dx2 = L - x
    dy2 = Y
    theta = np.arctan2(dy1, dx1)
    phi = np.arctan2(dy2, dx2)
    return np.degrees(theta), np.degrees(phi)

def travel_time_reflection(x_vals, y, L, c):
    d1 = np.sqrt(x_vals**2 + y**2)
    d2 = np.sqrt((L - x_vals)**2 + y**2)
    return (d1 + d2) / c

def travel_time_refraction(x_vals, y, Y, L, c1, c2):
    d1 = np.sqrt(x_vals**2 + y**2) / c1
    d2 = np.sqrt((L - x_vals)**2 + Y**2) / c2
    return d1 + d2
#plotting
def plot_fermat(ax1, ax2, mode, y, Y, L, c1, c2):
    ax1.clear()
    ax2.clear()
    x_vals = np.linspace(0.01, L - 0.01, 1000)

    if mode == 'Reflection':
        times = travel_time_reflection(x_vals, y, L, c1)
    else:
        times = travel_time_refraction(x_vals, y, Y, L, c1, c2)

    min_index = np.argmin(times)
    x_min = x_vals[min_index]
    t_min = times[min_index]

    # Time vs x
    ax1.plot(x_vals, times, label='Travel Time')
    ax1.plot(x_min, t_min, 'ro', label='Minimum Time')
    ax1.axvline(x=x_min, color='r', linestyle='--')
    ax1.set_xlabel('x (Contact Point)')
    ax1.set_ylabel('Total Travel Time')
    ax1.set_title(f'{mode}: Fermat’s Principle')
    ax1.grid(True)
    ax1.legend()

    #Annotations
    if mode == 'Reflection':
        theta, phi = get_angles(x_min, y, y, L)
        law_text = f"θ = {theta:.2f}°, φ = {phi:.2f}°\nEqual Angles ✅"
    else:
        theta, phi = get_angles(x_min, y, Y, L)
        snell_LHS = np.sin(np.radians(theta)) / c1
        snell_RHS = np.sin(np.radians(phi)) / c2
        law_text = f"sinθ/c₁ ≈ {snell_LHS:.3f}\nsinφ/c₂ ≈ {snell_RHS:.3f}\nSnell’s Law ✅"

    ax1.annotate(
        law_text,
        xy=(x_min, t_min),
        xytext=(x_min + 0.5, t_min + 0.05 * (max(times) - min(times))),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3)
    )

    # Ray path
    if mode == 'Reflection':
        A = (0, y)
        B = (L, y)
        S = (x_min, 0)
        ax2.plot([A[0], S[0], B[0]], [A[1], S[1], B[1]], 'b-o')
        ax2.axhline(0, color='gray', linestyle='--')
        ax2.set_title(f'Equal Angles Path')
    else:
        A = (0, y)
        B = (L, -Y)
        S = (x_min, 0)
        ax2.plot([A[0], S[0], B[0]], [A[1], S[1], B[1]], 'b-o')
        ax2.axhline(0, color='gray', linestyle='--')
        ax2.set_title(f'Snell’s Law Path')

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.grid(True)
    ax2.set_xlim(-1, L + 1)
    ax2.set_ylim(-max(Y, y) - 1, max(Y, y) + 1)
    ax2.axvline(x_min, color='red', linestyle='--', alpha=0.3)
#initial values
init_mode = 'Reflection'
init_y = 4
init_Y = 4
init_L = 8
init_c1 = 1.0
init_c2 = 2.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
plt.subplots_adjust(left=0.25, bottom=0.35)

# Initial plot
plot_fermat(ax1, ax2, init_mode, init_y, init_Y, init_L, init_c1, init_c2)

# Slider Axes
axcolor = 'lightgoldenrodyellow'
ax_y = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
ax_Y = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)
ax_L = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_c1 = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
ax_c2 = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)

# Sliders
s_y = Slider(ax_y, 'Height A (y)', 1, 10, valinit=init_y, valstep=0.5)
s_Y = Slider(ax_Y, 'Depth B (Y)', 1, 10, valinit=init_Y, valstep=0.5)
s_L = Slider(ax_L, 'Horizontal Dist (L)', 2, 20, valinit=init_L, valstep=1)
s_c1 = Slider(ax_c1, 'Speed c1', 0.1, 3.0, valinit=init_c1, valstep=0.1)
s_c2 = Slider(ax_c2, 'Speed c2', 0.1, 3.0, valinit=init_c2, valstep=0.1)

# modes
rax = plt.axes([0.025, 0.4, 0.15, 0.15], facecolor=axcolor)
radio = RadioButtons(rax, ('Reflection', 'Refraction'), active=0)

def update(val):
    plot_fermat(ax1, ax2, radio.value_selected, s_y.val, s_Y.val, s_L.val, s_c1.val, s_c2.val)
    fig.canvas.draw_idle()

def mode_update(label):
    plot_fermat(ax1, ax2, label, s_y.val, s_Y.val, s_L.val, s_c1.val, s_c2.val)
    fig.canvas.draw_idle()

s_y.on_changed(update)
s_Y.on_changed(update)
s_L.on_changed(update)
s_c1.on_changed(update)
s_c2.on_changed(update)
radio.on_clicked(mode_update)

plt.show()
