import sys
import os

# Ensure the correct CAMB version (V_phi_interp) is loaded.
# Run this script with:
#   PYTHONPATH=/scratch/s.2362709/CAMB_V_phi_interp:$PYTHONPATH python plot_...py
# or set CAMB_PATH below if running interactively.
CAMB_PATH = '/Users/giuliaborghetto/CAMB_V_phi_interp'
if CAMB_PATH not in sys.path:
    sys.path.insert(0, CAMB_PATH)

import camb
print(f"Using CAMB from: {camb.__file__}")
from camb.dark_energy import load_esr_function_string
import numpy as np
import pandas as pd
import sympy as sp
from pathlib import Path
import matplotlib.pyplot as plt
import re
import matplotlib
matplotlib.rcParams.update({'font.size': 15, 'text.usetex': False})


def load_reference_models():
    chi2_lcdm, chi2_cpl = None, None
    lcdm_data, cpl_data = None, None

    for lcdm_file in ["LCDM_cmblite.minimum.txt", "LCDM_CMB_lite.minimum.txt"]:
        if Path(lcdm_file).exists():
            lcdm_data = pd.read_csv(lcdm_file, sep=r'\s+', comment='#', header=None, engine='python')
            with open(lcdm_file, 'r') as f:
                cols = f.readline().strip().replace('#', '').strip().split()
            lcdm_data.columns = cols
            chi2_lcdm = (lcdm_data['chi2__cmb_lite_3d'].values[0] +
                         lcdm_data['chi2__bao.desi_dr2.desi_bao_all'].values[0] +
                         lcdm_data['chi2__sn.union3'].values[0])
            print(f"Loaded LCDM from {lcdm_file}, χ²={chi2_lcdm:.4f}")
            break

    for cpl_file in ["CPL_cmblite.minimum.txt"]:
        if Path(cpl_file).exists():
            cpl_data = pd.read_csv(cpl_file, sep=r'\s+', comment='#', header=None, engine='python')
            with open(cpl_file, 'r') as f:
                cols = f.readline().strip().replace('#', '').strip().split()
            cpl_data.columns = cols
            chi2_cpl = (cpl_data['chi2__cmb_lite_3d'].values[0] +
                        cpl_data['chi2__bao.desi_dr2.desi_bao_all'].values[0] +
                        cpl_data['chi2__sn.union3'].values[0])
            print(f"Loaded CPL from {cpl_file}, χ²={chi2_cpl:.4f}")
            break

    return lcdm_data, cpl_data, chi2_lcdm, chi2_cpl


def load_esr_model(runname, complexity, potential_idx, file_type="original"):
    esr_functions_file = f'../function_library/{runname}/compl_{complexity}/unique_equations_{complexity}.txt'
    function_dict = load_esr_function_string(esr_functions_file, potential_idx)

    if not function_dict['valid']:
        print(f"Skipping invalid function at index {potential_idx}")
        return None

    chains_dir = f"./chains/{potential_idx}/results"
    suffix_map = {"rerun": "_rerun", "iminuit": "_iminuit", "bobyqa": "_bobyqa", "original": ""}
    min_file_path = Path(chains_dir + suffix_map.get(file_type, "") + ".minimum.txt")

    if not min_file_path.exists():
        print(f"Results file not found: {min_file_path}")
        return None

    data = pd.read_csv(min_file_path, sep=r'\s+', comment='#', header=None, engine='python')
    with open(min_file_path, 'r') as f:
        cols = f.readline().strip().replace('#', '').strip().split()
    data.columns = cols

    H0      = data['H0'].values[0]
    ombh2   = data['ombh2'].values[0]
    omch2   = data['omch2'].values[0]
    n       = data['n'].values[0]
    theta_i = data['theta_i'].values[0] if 'theta_i' in data.columns else 0.0

    chi2_total = (data['chi2__cmb_lite_3d'].values[0] +
                  data['chi2__bao.desi_dr2.desi_bao_all'].values[0] +
                  data['chi2__sn.union3'].values[0])

    esr_param_names = [str(p) for p in function_dict['param_symbols']]
    esr_params = []
    camb_esr_params = {}
    for param in esr_param_names:
        camb_param = f"esr_param_{param}"
        val = data[camb_param].values[0] if camb_param in data.columns else 0.0
        esr_params.append(val)
        camb_esr_params[camb_param] = val

    # Analytic F(phi): substitute the fitted parameters into the ESR expression and lambdify.
    # This is exactly the function CAMB now evaluates in Fortran (esr_potentials), so no
    # interpolation table is built here -- f_func is used only for the f(phi) plot panel.
    x = sp.Symbol('x', real=True)
    f_expr = function_dict['expr_template'].subs(
        {s: v for s, v in zip(function_dict['param_symbols'], esr_params)})
    f_func = sp.lambdify(x, f_expr, 'numpy')

    # Run CAMB (analytic interface: index selects the compiled function; no phi range needed)
    try:
        pars = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2,
                               dark_energy_model='QuintessenceInterp',
                               theta_i=theta_i, n=n,
                               esr_functions_file=esr_functions_file,
                               esr_potential_index=potential_idx,
                               **camb_esr_params)
        results = camb.get_background(pars)
    except Exception as e:
        print(f"  Index {potential_idx}: CAMB failed ({e}); skipping.")
        return None
    omrh2 = results.Params.omnuh2
    print(f"  Index {potential_idx}: succeeded")

    # phi range actually visited by the field (for the f(phi) plot panel)
    phi_traj, _ = results.get_dark_energy_phi_phidot(np.linspace(1e-3, 1.0, 500))
    V1 = results.Params.DarkEnergy.V1
    V0 = results.Params.DarkEnergy.V0
    n = results.Params.DarkEnergy.n
    return {
        'V_function':    function_dict['func_string'],
        'param_names':   esr_param_names,
        'param_values':  esr_params,
        'H0': H0, 'ombh2': ombh2, 'omch2': omch2, 'omrh2': omrh2,
        'theta_i': theta_i, 'n': n,
        'chi2_total':    chi2_total,
        'results':       results,
        'pars':          pars,
        'complexity':    complexity,
        'potential_idx': potential_idx,
        'file_type':     file_type,
        'f_func':        f_func,
        'phi_range':     (float(np.min(phi_traj)), float(np.max(phi_traj))),
        'V0':            V0,
        'V1':            V1,
        'n':             n,
    }


def hubble_check(model, a_arr):
    res   = model['results']
    f_func = model['f_func']
    V1     = model['V1']
    H0    = model['H0']
    V0    = model['V0']
    n     = model['n']

    grhob       = res.grhob
    grhoc       = res.grhoc
    grhog       = res.grhog
    grhornomass = res.grhornomass
    grhocrit    = res.grhocrit

    phi, phidot = res.get_dark_energy_phi_phidot(a_arr)

    f_phi = f_func(phi) * np.ones_like(phi)

    # V_eff = Fortran VofPhi(deriv=0): V0*exp(-n*phi) + V1/a^3 * (f(phi) - 1)
    V_full  = V0 * np.exp(-n * phi) + V1 / a_arr**3 * (f_phi - 1)
    grhov_t = 0.5 * phidot**2 + a_arr**2 * V_full

    grho_no_de = (grhob + grhoc) * a_arr + (grhog + grhornomass)
    grhoa2     = grho_no_de + grhov_t * a_arr**2

    c_kms   = 2.99792458e5
    H_check = c_kms * np.sqrt(np.abs(grhoa2) / 3) / a_arr**2
    return H_check


def main():
    potential_specs = [
        (4, 121, "original"),
        (4, 78, "original"),
        (4, 67, "original"),
        (4, 8, "original"),
        # (4, 123, "original"),
    ]

    runname = "V_maths"
    z_arr   = np.linspace(0, 2.4, 500)
    a_arr   = 1 / (1 + z_arr)

    lcdm_data, cpl_data, chi2_lcdm, chi2_cpl = load_reference_models()

    # LCDM Hubble
    if lcdm_data is not None:
        H0_lcdm    = lcdm_data['H0'].values[0]
        ombh2_lcdm = lcdm_data['ombh2'].values[0]
        omch2_lcdm = lcdm_data['omch2'].values[0]
        pars_lcdm  = camb.set_params(H0=H0_lcdm, ombh2=ombh2_lcdm, omch2=omch2_lcdm)
        res_lcdm   = camb.get_background(pars_lcdm)
        hubble_lcdm = res_lcdm.hubble_parameter(z_arr) / H0_lcdm
    else:
        H0_lcdm = 67.0
        hubble_lcdm = np.ones_like(z_arr)

    # CPL
    if cpl_data is not None:
        H0_cpl    = cpl_data['H0'].values[0]
        ombh2_cpl = cpl_data['ombh2'].values[0]
        omch2_cpl = cpl_data['omch2'].values[0]
        w0 = cpl_data['w'].values[0]
        wa = cpl_data['wa'].values[0]
        pars_cpl  = camb.set_params(H0=H0_cpl, ombh2=ombh2_cpl, omch2=omch2_cpl,
                                     dark_energy_model='ppf', w=w0, wa=wa)
        res_cpl   = camb.get_background(pars_cpl)
        hubble_cpl = res_cpl.hubble_parameter(z_arr) / H0_cpl
        wde_cpl    = w0 + wa * z_arr / (1 + z_arr)
    else:
        w0, wa = -1.0, 0.0
        hubble_cpl  = hubble_lcdm.copy()
        wde_cpl     = -np.ones_like(z_arr)

    # Load ESR models
    print("\nLoading ESR models...")
    esr_models = []
    for complexity, potential_idx, file_type in potential_specs:
        model = load_esr_model(runname, complexity, potential_idx, file_type)
        if model is not None:
            esr_models.append(model)
            print(f"  Loaded: f(phi)={model['V_function']}, χ²={model['chi2_total']:.4f}")

    if not esr_models:
        print("No valid models loaded.")
        return

    # Distinct, colourblind-friendly palette
    _palette = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
                "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990"]
    colors = [_palette[i % len(_palette)] for i in range(len(esr_models))]

    # ── figure layout: w(z), H(z)/H_LCDM, V(a,phi), legend ──
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.2, 0.45],
                          hspace=0.4, wspace=0.3)
    ax_wde  = fig.add_subplot(gs[0, 0])
    ax_hub  = fig.add_subplot(gs[0, 1])
    ax_pot  = fig.add_subplot(gs[1, :])
    ax_leg  = fig.add_subplot(gs[2, :])
    ax_leg.axis('off')
    ax_dict = {'w_de': ax_wde, 'hubble': ax_hub, 'potential': ax_pot}

    for i, (model, color) in enumerate(zip(esr_models, colors)):
        res = model['results']

        # get_dark_energy_rho_w returns w_eff directly (eq 3.5/3.30: p = phidot^2/2 - V1(phi),
        # i.e. the equation of state satisfying d rho_DE/dt = -3H(1+w_eff) rho_DE)
        _, w_eff     = np.array(res.get_dark_energy_rho_w(a_arr))
        hubble       = res.hubble_parameter(z_arr) / model['H0']
        hubble_ratio = hubble / hubble_lcdm
        H_check = hubble_check(model, a_arr) / model['H0']

        delta_chi2_lcdm = model['chi2_total'] - chi2_lcdm if chi2_lcdm is not None else None
        delta_chi2_cpl  = model['chi2_total'] - chi2_cpl  if chi2_cpl  is not None else None

        # Label
        f_str = re.sub(r'\bx\b', r'\\phi', model['V_function'])
        param_parts = [f"n={model['n']:.3f}"]
        if model['param_names'] and model['param_values']:
            param_parts += [f"{pname}={v:.3f}" for pname, v in
                            zip(model['param_names'], model['param_values'])]
        param_str = ', '.join(param_parts)
        chi2_parts = []
        if delta_chi2_lcdm is not None:
            chi2_parts.append(rf"$\Delta\chi^2_{{\Lambda CDM}}={delta_chi2_lcdm:.2f}$")
        if delta_chi2_cpl is not None:
            chi2_parts.append(rf"$\Delta\chi^2_{{CPL}}={delta_chi2_cpl:.2f}$")
        label = rf"$f(\phi)={f_str}$ ({param_str})" + "\n" + "\n".join(chi2_parts)

        # w_eff only
        ax_dict['w_de'].plot(z_arr, w_eff, color=color, linewidth=2, linestyle='-')

        ax_dict['hubble'].plot(z_arr, hubble_ratio, color=color, linewidth=2)
        #ax_dict['hubble'].plot(z_arr, H_check/ hubble_lcdm, color=color, linewidth=2, linestyle='--')

        # Plot f(phi) vs phi over the range the field actually visits (analytic F)
        phi_lo, phi_hi = model['phi_range']
        pad = 0.05 * (phi_hi - phi_lo) + 1e-3
        phi_full = np.linspace(phi_lo - pad, phi_hi + pad, 500)
        f_vals = model['f_func'](phi_full) * np.ones_like(phi_full)
        ax_dict['potential'].plot(phi_full, f_vals, color=color, linewidth=2, label=label)

    # CPL uncertainty bands
    cpl_bounds_file = "cpl_w_hubble_bounds_cmblite.csv"
    if Path(cpl_bounds_file).exists():
        cpl_bounds = pd.read_csv(cpl_bounds_file)
        ax_dict['w_de'].fill_between(cpl_bounds['z'],
                                      cpl_bounds['w_de_lower_95'],
                                      cpl_bounds['w_de_upper_95'],
                                      alpha=0.2, color='gray')
        hubble_lcdm_interp = np.interp(cpl_bounds['z'], z_arr, hubble_lcdm)
        ax_dict['hubble'].fill_between(cpl_bounds['z'],
                                        cpl_bounds['hubble_lower_95'] / hubble_lcdm_interp,
                                        cpl_bounds['hubble_upper_95'] / hubble_lcdm_interp,
                                        alpha=0.2, color='gray')

    # Reference lines
    ax_dict['w_de'].axhline(-1, color='black', linestyle='-.', linewidth=2)
    ax_dict['w_de'].plot(z_arr, wde_cpl, 'gray', linestyle='--', linewidth=2)
    ax_dict['hubble'].axhline(1, color='black', linestyle='-.', linewidth=2)
    ax_dict['hubble'].plot(z_arr, hubble_cpl / hubble_lcdm, 'gray', linestyle='--', linewidth=2)

    # Legend proxy entries
    ax_dict['potential'].plot([], [], color='black', linestyle='-.', linewidth=2, label=r'$\Lambda$CDM')
    ax_dict['potential'].plot([], [], color='gray',  linestyle='--', linewidth=2, label='CPL')


    # Formatting
    ax_dict['w_de'].set_xlabel(r'$z$')
    ax_dict['w_de'].set_ylabel(r'$w_\mathrm{eff}(z)$')
    ax_dict['w_de'].set_xlim(0, 2.4)
    ax_dict['w_de'].grid(True, alpha=0.3)
    ax_dict['w_de'].set_ylim([-1.5,-0.3])

    ax_dict['hubble'].set_xlabel(r'$z$')
    ax_dict['hubble'].set_ylabel(r'$H(z)/H_{\Lambda\mathrm{CDM}}(z)$')
    ax_dict['hubble'].set_xlim(0, 2.4)
    ax_dict['hubble'].grid(True, alpha=0.3)

    ax_dict['potential'].set_xlabel(r'$\phi$')
    ax_dict['potential'].set_ylabel(r'$f(\phi)$')
    ax_dict['potential'].grid(True, alpha=0.3)

    handles, labels = ax_dict['potential'].get_legend_handles_labels()
    ax_leg.legend(handles, labels, loc='upper center', fontsize=11,
                  ncol=3, borderaxespad=0, frameon=True)

    plt.suptitle(
        r"$V(a,\phi) = V_0\,e^{-\lambda\phi} + \dfrac{V_1}{a^3}\,f(\phi) - \dfrac{V_1}{a^3} $",
        fontsize=16, y=0.94
    )
    plt.savefig("esr_V_phi_new3.pdf", bbox_inches='tight')
    print("Saved: esr_V_phi_new3.pdf")


if __name__ == "__main__":
    main()