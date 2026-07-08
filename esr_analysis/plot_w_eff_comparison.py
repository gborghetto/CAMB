"""Compare equation-of-state definitions for the eq (4.5) time-dependent potential.

For a coupled model V(phi,a) = V0*exp(-lambda*phi) + (rho_m0/a^3)*(F(phi)-1) this plots:

  1. w_eff        -- the effective equation of state (draft eq 3.5/3.30), i.e. the
                     quantity satisfying d rho_DE/dt = -3H(1+w_eff) rho_DE that DESI/CPL
                     comparisons constrain. This is what get_dark_energy_rho_w now returns.
  2. w_old        -- the naive field w with the full effective potential in the pressure,
                     (K - V_eff)/(K + V_eff): the pre-fix CAMB output, bounded below by -1
                     whenever V_eff > 0 (misses phantom crossing by construction).
  3. w_uncoupled  -- the same bare potential V1(phi) = V0*exp(-lambda*phi) run as minimal
                     quintessence (coupling switched off via F(phi) identically 1, V0 retuned).

and the effective DE density rho_DE(z) of the coupled model against the uncoupled one
(both normalised to their common value today, since V0 is tuned to the same Omega_DE).

Example (test_108 phantom-crossing point is the default):
    python plot_w_eff_comparison.py
    python plot_w_eff_comparison.py --a0 0.02 --theta-i 1.13 -o my_comparison.png
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import sympy

import camb
from camb.dark_energy import QuintessenceInterp, load_esr_function_string

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ESR_FILE = os.path.join(SCRIPT_DIR, '..', 'function_library',
                                'V_maths', 'compl_4', 'unique_equations_4.txt')


def get_background(de, args):
    pars = camb.set_params(H0=args.H0, ombh2=args.ombh2, omch2=args.omch2,
                           num_massive_neutrinos=1, nnu=3.046)
    pars.DarkEnergy = de
    return camb.get_background(pars)


def run_coupled(args):
    de = QuintessenceInterp()
    esr_params = {}
    for i, val in enumerate([args.a0, args.a1, args.a2, args.a3]):
        if val is not None:
            esr_params[f'esr_param_a{i}'] = val
    de.set_params(esr_functions_file=args.esr_file, esr_potential_index=args.esr_index,
                  n=args.lam, theta_i=args.theta_i, **esr_params)
    return get_background(de, args)


def run_uncoupled(args):
    # esr_potential_index = -1 is the reserved "no coupling" case F(phi) = 1: the (F-1) term and
    # its derivatives vanish, so the class evolves minimal quintessence with V1(phi)=V0 exp(-lam phi)
    de = QuintessenceInterp()
    de.set_params(esr_functions_file=args.esr_file, esr_potential_index=-1,
                  n=args.lam, theta_i=args.theta_i)
    return get_background(de, args)


def potential_latex(args):
    """Full V(phi,a) of eq (4.5) with the numeric lambda and the substituted ESR F(phi)."""
    fd = load_esr_function_string(args.esr_file, args.esr_index)
    vals = [v for v in [args.a0, args.a1, args.a2, args.a3] if v is not None]
    # substitute parameter values (same symbol ordering as QuintessenceInterp.set_params)
    F = fd['expr_template'].subs(
        {p: sympy.Float(v, 3) for p, v in zip(fd['param_symbols'], vals)})
    x = [s for s in F.free_symbols if str(s) == 'x']
    if x:
        F = F.subs(x[0], sympy.Symbol('phi'))
    return (r'$V(\phi,a) \,=\, V_0\, e^{-%.3g\,\phi} \;+\; '
            r'\frac{\rho_m^0}{a^3}\left[\,%s \,-\, 1\,\right]$'
            % (args.lam, sympy.latex(F)))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--esr-file', default=DEFAULT_ESR_FILE)
    p.add_argument('--esr-index', type=int, default=8)
    p.add_argument('--a0', type=float, default=-0.020559)
    p.add_argument('--a1', type=float, default=None)
    p.add_argument('--a2', type=float, default=None)
    p.add_argument('--a3', type=float, default=None)
    p.add_argument('--lam', type=float, default=0.965081, help='exponent lambda of V1')
    p.add_argument('--theta-i', type=float, default=1.129909, help='initial field value')
    p.add_argument('--H0', type=float, default=66.264064)
    p.add_argument('--ombh2', type=float, default=0.022287)
    p.add_argument('--omch2', type=float, default=0.119094)
    p.add_argument('--zmax', type=float, default=3.0)
    p.add_argument('-o', '--output', default=os.path.join(SCRIPT_DIR, 'w_eff_comparison.png'))
    args = p.parse_args()

    z = np.linspace(0, args.zmax, 1200)
    a = 1 / (1 + z)

    # --- coupled model ---
    res_c = run_coupled(args)
    rho_c, w_eff = res_c.get_dark_energy_rho_w(a)          # w_eff: p = K - V1(phi), eq (3.30)
    phi, phidot = res_c.get_dark_energy_phi_phidot(a)      # phidot is d phi/d tau (conformal)
    V_eff = res_c.get_dark_energy_Vphi(a, phi, deriv=0)    # full V(phi,a), 8piG units (Mpc^-2)
    K = 0.5 * (phidot / a)**2                              # kinetic energy, same units
    w_old = (K - V_eff) / (K + V_eff)                      # pre-fix output: eq (2.7) with V_eff
    V0_c = res_c.Params.DarkEnergy.V0

    # --- uncoupled minimal quintessence with the same V1(phi) ---
    res_u = run_uncoupled(args)
    rho_u, w_unc = res_u.get_dark_energy_rho_w(a)          # F==1: both w definitions coincide
    V0_u = res_u.Params.DarkEnergy.V0

    # normalise densities to today (z[0] = 0); both are tuned to the same Omega_DE
    rho_c = rho_c / rho_c[0]
    rho_u = rho_u / rho_u[0]

    print(f'tuned V0 (coupled)   = {V0_c:.6e}')
    print(f'tuned V0 (uncoupled) = {V0_u:.6e}')
    print(f'w_eff(0) = {w_eff[0]:.4f},  w_old(0) = {w_old[0]:.4f},  w_uncoupled(0) = {w_unc[0]:.4f}')
    crossing = np.where(np.diff(np.sign(w_eff + 1)))[0]
    if len(crossing):
        print(f'w_eff crosses -1 at z = {z[crossing[0]]:.3f}')
    else:
        print('w_eff does not cross -1 in the plotted range')

    # --- plot ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(z, w_eff, color='#1f4e9c', lw=2.2,
            label=r'$w_{\rm eff}$ (eq. 3.5, coupled)')
    ax.plot(z, w_old, color='#c0392b', lw=2, ls='--',
            label=r'field $w$ with $V_{\rm eff}$ (old output)')
    ax.plot(z, w_unc, color='#3d8f5f', lw=2, ls='-.',
            label=r'uncoupled quintessence, same $V_1(\phi)$')
    ax.axhline(-1, color='k', lw=0.8, ls=':')
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$w_{\rm DE}(z)$')
    ax.set_xlim(0, args.zmax)
    ax.set_ylim(min(-1.1, 1.05 * w_eff.min()), -0.3)
    ax.legend(fontsize=8.5, loc='lower left', frameon=False)
    mant, expo = f'{V0_c:.2e}'.split('e')
    ax.set_title(rf'$\theta_i={args.theta_i:.3g}$, ESR idx {args.esr_index}, '
                 rf'tuned $V_0={mant}\times10^{{{int(expo)}}}\,{{\rm Mpc}}^{{-2}}$', fontsize=10)

    ax = axes[1]
    ax.plot(z, rho_c, color='#1f4e9c', lw=2.2, label=r'coupled: $\rho_{\rm DE}=K+V(\phi,a)$')
    ax.plot(z, rho_u, color='#3d8f5f', lw=2, ls='-.', label=r'uncoupled: $K+V_1(\phi)$')
    ax.axhline(0, color='k', lw=0.8, ls=':')
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$\rho_{\rm DE}(z)/\rho_{\rm DE,0}$')
    ax.set_xlim(0, args.zmax)
    ax.legend(fontsize=8.5, frameon=False)
    ax.set_title('effective DE density (both tuned to the same $\\Omega_{\\rm DE}$)', fontsize=10)

    fig.suptitle(potential_latex(args), fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(args.output, dpi=150)
    print('saved', args.output)


if __name__ == '__main__':
    main()
