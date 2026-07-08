#!/usr/bin/env python3
"""Comparison plots for the ESR coupling functions fitted by run_selected_potentials.py.

Reads the completed best-fits under chains/selected/<run>/<idx>/results.minimum.txt, runs CAMB
at each best-fit (analytic esr_potentials interface), and plots -- in the style of
plot_multi_potential_comparison_custom_DE_exp.py -- three panels:
  * w_eff(z)          effective dark-energy equation of state (eq 3.5/3.30, from CAMB)
  * H(z)/H_LCDM(z)    expansion history relative to LambdaCDM
  * F(phi)            the analytic coupling function over the field's visited range
against LambdaCDM and CPL references (with the CPL 95% band), labelling each with its
Delta chi^2 vs LambdaCDM and CPL.

Only functions whose fit completed (results.minimum.txt) AND improve on LambdaCDM
(Delta chi^2_LCDM < 0) are plotted; pass --keep-all to show every completed fit.

    python plot_selected_comparison.py            # best 5 fits -> esr_selected_comparison.pdf
    python plot_selected_comparison.py --best 8   # best 8
    python plot_selected_comparison.py --best 0   # all fits that beat LCDM
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CAMB_PATH = '/Users/amk_1/cosmocodes/ESR_V_Phi/CAMB'
if CAMB_PATH not in sys.path:
    sys.path.insert(0, CAMB_PATH)
import camb
from camb.dark_energy import load_esr_function_string

matplotlib.rcParams.update({'font.size': 15, 'text.usetex': False})

RUNNAME, COMPLEXITY = 'V_maths', 4
CHAINS_DIR = f'chains/selected/{RUNNAME}_compl{COMPLEXITY}'
ESR_FILE = os.path.join(CAMB_PATH, 'function_library', RUNNAME,
                        f'compl_{COMPLEXITY}', f'unique_equations_{COMPLEXITY}.txt')

# the three background likelihoods (their chi^2 columns sum to the total we rank on)
CHI2_COLS = ['chi2__cmb_lite_3d', 'chi2__bao.desi_dr2.desi_bao_all', 'chi2__sn.union3']


def read_minimum(path):
    """Parse a cobaya .minimum.txt (commented header + one row) into a dict."""
    data = pd.read_csv(path, sep=r'\s+', comment='#', header=None, engine='python')
    with open(path) as f:
        cols = f.readline().strip().replace('#', '').strip().split()
    data.columns = cols
    return data.iloc[0]


def total_chi2(row):
    return float(sum(row[c] for c in CHI2_COLS))


def load_reference_models():
    lcdm = cpl = None
    chi2_lcdm = chi2_cpl = None
    for f in ("LCDM_cmblite.minimum.txt", "LCDM_CMB_lite.minimum.txt"):
        if os.path.exists(f):
            lcdm = read_minimum(f); chi2_lcdm = total_chi2(lcdm)
            print(f"Loaded LCDM from {f}, chi2={chi2_lcdm:.4f}")
            break
    if os.path.exists("CPL_cmblite.minimum.txt"):
        cpl = read_minimum("CPL_cmblite.minimum.txt"); chi2_cpl = total_chi2(cpl)
        print(f"Loaded CPL from CPL_cmblite.minimum.txt, chi2={chi2_cpl:.4f}")
    return lcdm, cpl, chi2_lcdm, chi2_cpl


def load_model(idx):
    min_file = os.path.join(CHAINS_DIR, str(idx), 'results.minimum.txt')
    if not os.path.exists(min_file):
        return None
    row = read_minimum(min_file)
    fd = load_esr_function_string(ESR_FILE, idx, verbose=False)
    if not fd['valid']:
        return None

    H0, ombh2, omch2 = float(row['H0']), float(row['ombh2']), float(row['omch2'])
    n, theta_i = float(row['n']), float(row['theta_i'])
    esr_names = [str(p) for p in fd['param_symbols']]
    esr_vals = [float(row[f'esr_param_{p}']) for p in esr_names]
    esr_kwargs = {f'esr_param_{p}': v for p, v in zip(esr_names, esr_vals)}

    # analytic F(phi): substitute best-fit params into the ESR expression
    x = sp.Symbol('x', real=True)
    f_expr = fd['expr_template'].subs({s: v for s, v in zip(fd['param_symbols'], esr_vals)})
    f_func = sp.lambdify(x, f_expr, 'numpy')

    pars = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2,
                           num_massive_neutrinos=1, nnu=3.046,
                           dark_energy_model='QuintessenceInterp',
                           theta_i=theta_i, n=n,
                           esr_functions_file=ESR_FILE, esr_potential_index=idx, **esr_kwargs)
    try:
        res = camb.get_background(pars)
    except Exception as e:
        print(f"  index {idx}: CAMB failed at best-fit ({e}); skipping.")
        return None
    phi_traj, _ = res.get_dark_energy_phi_phidot(np.linspace(1e-3, 1.0, 500))
    return {
        'idx': idx, 'results': res, 'H0': H0,
        'func_string': fd['func_string'], 'param_names': esr_names, 'param_values': esr_vals,
        'n': n, 'theta_i': theta_i, 'chi2_total': total_chi2(row),
        'f_func': f_func, 'phi_range': (float(np.min(phi_traj)), float(np.max(phi_traj))),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-o', '--output', default='esr_selected_comparison.pdf')
    ap.add_argument('--zmax', type=float, default=2.4)
    ap.add_argument('--keep-all', action='store_true',
                    help="Plot every completed fit, even those no better than LCDM "
                         "(by default only fits that improve on LCDM are shown)")
    ap.add_argument('--best', type=int, default=5,
                    help="Plot only the best N fits by total chi^2 (default 5; 0 = all)")
    args = ap.parse_args()

    z_arr = np.linspace(0, args.zmax, 500)
    a_arr = 1 / (1 + z_arr)

    lcdm, cpl, chi2_lcdm, chi2_cpl = load_reference_models()

    # LCDM expansion history (denominator of the H-ratio panel)
    if lcdm is not None:
        H0_lcdm = float(lcdm['H0'])
        pars_lcdm = camb.set_params(H0=H0_lcdm, ombh2=float(lcdm['ombh2']), omch2=float(lcdm['omch2']))
        hubble_lcdm = camb.get_background(pars_lcdm).hubble_parameter(z_arr) / H0_lcdm
    else:
        H0_lcdm, hubble_lcdm = 67.0, np.ones_like(z_arr)

    # CPL reference curves
    if cpl is not None:
        w0, wa = float(cpl['w']), float(cpl['wa'])
        pars_cpl = camb.set_params(H0=float(cpl['H0']), ombh2=float(cpl['ombh2']),
                                   omch2=float(cpl['omch2']), dark_energy_model='ppf', w=w0, wa=wa)
        hubble_cpl = camb.get_background(pars_cpl).hubble_parameter(z_arr) / float(cpl['H0'])
        wde_cpl = w0 + wa * z_arr / (1 + z_arr)
    else:
        hubble_cpl, wde_cpl = hubble_lcdm.copy(), -np.ones_like(z_arr)

    # completed ESR models, in ascending index order
    print("\nLoading fitted ESR models...")
    indices = sorted(int(d) for d in os.listdir(CHAINS_DIR)
                     if os.path.exists(os.path.join(CHAINS_DIR, d, 'results.minimum.txt')))
    models = [m for m in (load_model(i) for i in indices) if m is not None]
    if not models:
        print(f"No completed fits with results.minimum.txt under {CHAINS_DIR}.")
        return
    for m in models:
        print(f"  idx {m['idx']:3d}: f(phi)={m['func_string']}, chi2_total={m['chi2_total']:.3f}")

    # Only plot fits that actually improve on LambdaCDM (Delta chi^2_LCDM < 0); this also drops
    # failed/boundary-railed minimisations that would distort the axes. --keep-all overrides.
    if chi2_lcdm is not None and not args.keep_all:
        kept = [m for m in models if m['chi2_total'] < chi2_lcdm]
        for m in models:
            if m not in kept:
                print(f"  -> excluding idx {m['idx']} ({m['func_string']}): "
                      f"Delta chi2_LCDM = {m['chi2_total'] - chi2_lcdm:+.2f} "
                      f"(not better than LCDM)")
        models = kept
    if not models:
        print("No completed fit improves on LCDM.")
        return

    # Rank by total chi^2 (lower is better) and keep the best N.
    models.sort(key=lambda m: m['chi2_total'])
    if args.best and len(models) > args.best:
        print(f"\nKeeping the best {args.best} of {len(models)} fits (by chi^2):")
        models = models[:args.best]
    for rank, m in enumerate(models, 1):
        dl = f"{m['chi2_total'] - chi2_lcdm:+.2f}" if chi2_lcdm is not None else "n/a"
        print(f"  #{rank} idx {m['idx']:3d}: {m['func_string']:22s} Delta chi2_LCDM={dl}")

    _palette = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
                "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990"]
    colors = [_palette[i % len(_palette)] for i in range(len(models))]

    # layout: w(z), H(z)/H_LCDM (top), F(phi) (middle), legend (bottom)
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.2, 0.45], hspace=0.4, wspace=0.3)
    ax_wde = fig.add_subplot(gs[0, 0])
    ax_hub = fig.add_subplot(gs[0, 1])
    ax_pot = fig.add_subplot(gs[1, :])
    ax_leg = fig.add_subplot(gs[2, :]); ax_leg.axis('off')

    for model, color in zip(models, colors):
        res = model['results']
        _, w_eff = np.array(res.get_dark_energy_rho_w(a_arr))  # w_eff (eq 3.5/3.30)
        hubble_ratio = (res.hubble_parameter(z_arr) / model['H0']) / hubble_lcdm

        dchi2_lcdm = model['chi2_total'] - chi2_lcdm if chi2_lcdm is not None else None
        dchi2_cpl = model['chi2_total'] - chi2_cpl if chi2_cpl is not None else None

        f_str = re.sub(r'\bx\b', r'\\phi', model['func_string'])
        parts = [f"n={model['n']:.3f}"] + [f"{p}={v:.3f}" for p, v
                                           in zip(model['param_names'], model['param_values'])]
        chi2_parts = []
        if dchi2_lcdm is not None:
            chi2_parts.append(rf"$\Delta\chi^2_{{\Lambda CDM}}={dchi2_lcdm:.2f}$")
        if dchi2_cpl is not None:
            chi2_parts.append(rf"$\Delta\chi^2_{{CPL}}={dchi2_cpl:.2f}$")
        label = rf"$f(\phi)={f_str}$ ({', '.join(parts)})" + "\n" + "  ".join(chi2_parts)

        ax_wde.plot(z_arr, w_eff, color=color, linewidth=2)
        ax_hub.plot(z_arr, hubble_ratio, color=color, linewidth=2)

        phi_lo, phi_hi = model['phi_range']
        pad = 0.05 * (phi_hi - phi_lo) + 1e-3
        phi_full = np.linspace(phi_lo - pad, phi_hi + pad, 500)
        ax_pot.plot(phi_full, model['f_func'](phi_full) * np.ones_like(phi_full),
                    color=color, linewidth=2, label=label)

    # CPL 95% bands
    if os.path.exists("cpl_w_hubble_bounds_cmblite.csv"):
        b = pd.read_csv("cpl_w_hubble_bounds_cmblite.csv")
        ax_wde.fill_between(b['z'], b['w_de_lower_95'], b['w_de_upper_95'], alpha=0.2, color='gray')
        hl = np.interp(b['z'], z_arr, hubble_lcdm)
        ax_hub.fill_between(b['z'], b['hubble_lower_95'] / hl, b['hubble_upper_95'] / hl,
                            alpha=0.2, color='gray')

    # reference lines
    ax_wde.axhline(-1, color='black', linestyle='-.', linewidth=2)
    ax_wde.plot(z_arr, wde_cpl, 'gray', linestyle='--', linewidth=2)
    ax_hub.axhline(1, color='black', linestyle='-.', linewidth=2)
    ax_hub.plot(z_arr, hubble_cpl / hubble_lcdm, 'gray', linestyle='--', linewidth=2)
    ax_pot.plot([], [], color='black', linestyle='-.', linewidth=2, label=r'$\Lambda$CDM')
    ax_pot.plot([], [], color='gray', linestyle='--', linewidth=2, label='CPL')

    ax_wde.set_xlabel(r'$z$'); ax_wde.set_ylabel(r'$w_\mathrm{eff}(z)$')
    ax_wde.set_xlim(0, args.zmax); ax_wde.set_ylim(-1.5, -0.3); ax_wde.grid(True, alpha=0.3)
    ax_hub.set_xlabel(r'$z$'); ax_hub.set_ylabel(r'$H(z)/H_{\Lambda\mathrm{CDM}}(z)$')
    ax_hub.set_xlim(0, args.zmax); ax_hub.grid(True, alpha=0.3)
    ax_pot.set_xlabel(r'$\phi$'); ax_pot.set_ylabel(r'$f(\phi)$'); ax_pot.grid(True, alpha=0.3)

    handles, labels = ax_pot.get_legend_handles_labels()
    ax_leg.legend(handles, labels, loc='upper center', fontsize=11, ncol=3,
                  borderaxespad=0, frameon=True)
    plt.suptitle(r"$V(a,\phi) = V_0\,e^{-\lambda\phi} + \dfrac{V_1}{a^3}\,f(\phi) - \dfrac{V_1}{a^3}$",
                 fontsize=16, y=0.94)
    plt.savefig(args.output, bbox_inches='tight')
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
