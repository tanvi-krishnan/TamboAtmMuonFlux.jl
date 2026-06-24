"""
Validate mceq_flux_tables.h5 against:
1. Gaisser parametrization (simple analytic formula)
2. Published sea-level muon flux values

Gaisser parametrization (from Gaisser "Cosmic Rays and Particle Physics"):
  Φ_μ(E, θ) ≈ 0.14 * E^{-2.7} / cm² s sr GeV  *
               [ 1/(1 + 1.1*E*cosθ/115) + 0.054/(1 + 1.1*E*cosθ/850) ]
  valid for E >> 1 GeV, cos θ > 0.2
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

table_path = "../data/mceq_flux_tables.h5"

with h5py.File(table_path, "r") as f:
    E_GeV     = f["E_GeV"][:]
    cos_theta = f["cos_theta"][:]
    h_km      = f["h_km"][:]
    flux      = f["flux"][:]

log10E = np.log10(E_GeV)
interp = RegularGridInterpolator(
    (log10E, cos_theta, h_km), flux,
    method="linear", bounds_error=False, fill_value=0.0,
)

def gaisser(E, ct):
    """Gaisser parametrization, sea level, GeV^-1 cm^-2 s^-1 sr^-1."""
    return (0.14 * E**(-2.7) *
            (1/(1 + 1.1*E*ct/115) + 0.054/(1 + 1.1*E*ct/850)))

E_fine = E_GeV  # use MCEq native grid to avoid aliasing
ct_vals = [0.3, 0.5, 0.7, 1.0]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ── Panel 1: E³·Φ vs E at sea level, compare MCEq vs Gaisser ────────────────
ax = axes[0]
for ct, col in zip(ct_vals, ['C0','C1','C2','C3']):
    theta_deg = np.degrees(np.arccos(ct))
    h_sl = h_km[0]  # ~0 km (sea level)

    # MCEq
    pts = np.column_stack([np.log10(E_fine), np.full_like(E_fine, ct), np.full_like(E_fine, h_sl)])
    phi_mceq = interp(pts)
    ax.loglog(E_fine, E_fine**3 * phi_mceq, color=col, lw=2, label=f"MCEq θ={theta_deg:.0f}°")

    # Gaisser
    phi_g = gaisser(E_fine, ct)
    ax.loglog(E_fine, E_fine**3 * phi_g, color=col, lw=1.5, ls='--')

ax.set_xlabel("E (GeV)")
ax.set_ylabel("E³ · Φ_μ  [GeV² cm⁻² s⁻¹ sr⁻¹]")
ax.set_title("MCEq (solid) vs Gaisser (dashed) — sea level")
ax.set_xlim(10, 1e6)
ax.set_ylim(1e-4, 1.0)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Panel 2: ratio MCEq / Gaisser ────────────────────────────────────────────
ax = axes[1]
for ct, col in zip(ct_vals, ['C0','C1','C2','C3']):
    theta_deg = np.degrees(np.arccos(ct))
    h_sl = h_km[0]

    pts = np.column_stack([np.log10(E_fine), np.full_like(E_fine, ct), np.full_like(E_fine, h_sl)])
    phi_mceq = interp(pts)
    phi_g    = gaisser(E_fine, ct)

    ratio = np.where(phi_g > 0, phi_mceq / phi_g, np.nan)
    ax.semilogx(E_fine, ratio, color=col, label=f"θ={theta_deg:.0f}°")

ax.axhline(1.0, color='k', ls='--', lw=1)
ax.axhspan(0.8, 1.2, alpha=0.1, color='k', label="±20%")
ax.set_xlabel("E (GeV)")
ax.set_ylabel("MCEq / Gaisser")
ax.set_title("Ratio at sea level")
ax.set_xlim(10, 1e6)
ax.set_ylim(1e-4, 1.0)
ax.set_ylim(0, 1.3)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle("Flux validation: MCEq (SIBYLL23D, H3a) vs Gaisser parametrization", fontsize=11)
plt.tight_layout()
out = "../data/flux_validation.png"
plt.savefig(out, dpi=150)
print(f"Saved: {out}")

# Print spot-check values at 1 TeV, vertical
E_check = 1000.0
ct_check = 1.0
h_sl = h_km[0]
pts = np.array([[np.log10(E_check), ct_check, h_sl]])
phi_mceq_val = interp(pts)[0]
phi_g_val    = gaisser(E_check, ct_check)
print(f"\nSpot check at E=1 TeV, vertical, sea level:")
print(f"  MCEq:    {phi_mceq_val:.4e} GeV^-1 cm^-2 s^-1 sr^-1")
print(f"  Gaisser: {phi_g_val:.4e} GeV^-1 cm^-2 s^-1 sr^-1")
print(f"  Ratio:   {phi_mceq_val/phi_g_val:.3f}")
