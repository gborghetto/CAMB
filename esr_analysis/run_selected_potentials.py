#!/usr/bin/env python3
"""Run the Cobaya analyses for a hand-picked set of ESR coupling functions F(phi).

These use the analytic esr_potentials interface (no interpolation tables): the CAMB theory
selects each function by (library, index), and Cobaya fits the cosmology + the function
parameters against CMBlite + DESI DR2 BAO + Union3, minimising chi^2.

The ~25 functions below were chosen from V_maths/compl_4 to (i) be analytic and singularity-free
over the visited field range, (ii) tune to the correct Omega_de and reach the DE regime at a
representative point, and (iii) span the operator classes: exponential, power / Gaussian-exponent,
polynomial, rational, and oscillatory (axion-like). All carry a free parameter a0 -- a parameter-free
F(phi) cannot represent a coupled quintessence model.

For each function the run is two-phase: a short exploratory MCMC chain, then a bobyqa
minimize that (sharing the chain's output prefix) starts from the MCMC MAP and reuses its
learned covariance -- more robust than minimising from a cold start.

Usage:
    # MCMC + minimize for every selected function
    python run_selected_potentials.py

    # shorter/longer exploratory chain
    python run_selected_potentials.py --mcmc-samples 1000

    # quick pipeline check: evaluate chi^2 once at each function's fiducial point
    python run_selected_potentials.py --test

    # a single function, forcing overwrite
    python run_selected_potentials.py --only 67 --force

Prerequisites (paths configurable at the top of this file):
  * this CAMB build (analytic interface) importable  -> CAMB_PATH
  * the cmb_lite_3d likelihood module on the path     -> CMB_LITE_DIR
  * cobaya with the desi_dr2 and union3 likelihoods (already installed)
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- paths (edit these for your machine) ------------------------------------
CAMB_PATH = '/Users/amk_1/cosmocodes/ESR_V_Phi/CAMB'
CMB_LITE_DIR = '/Users/amk_1/cosmocodes/ESR_V_Phi/cmblite'  # holds cmb_lite_3d.py (thetastar/100 fix)
# cobaya packages path with the DESI DR2 BAO + Union3 data. Leave None to use your cobaya config
# (e.g. the OneDrive location); or set COBAYA_PACKAGES_PATH in the environment to override.
PACKAGES_PATH = None #os.environ.get('COBAYA_PACKAGES_PATH') or None
RUNNAME = 'V_maths'
COMPLEXITY = 4
OUTPUT_ROOT = 'chains/selected'
# ----------------------------------------------------------------------------

for _p in (CAMB_PATH, CMB_LITE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from camb.dark_energy import load_esr_function_string  # noqa: E402
from cobaya.run import run  # noqa: E402

ESR_FILE = os.path.join(
    CAMB_PATH, 'function_library', RUNNAME, f'compl_{COMPLEXITY}',
    f'unique_equations_{COMPLEXITY}.txt')

# Selected functions: index -> (short label, fiducial {a0, theta_i, n} that is known to tune).
# The fiducial seeds the minimiser's reference point and is the evaluation point for --test.
# ~25 parameterised V_maths/compl_4 functions that tune to Omega_de and reach the DE regime at a
# representative point (found by scanning CAMB backgrounds; see the git history / memory). The
# fiducial (a0, theta_i, n) seeds the MCMC/minimiser. All carry a free a0.
SELECTED = {
    # 2:   {'label': '|a0|/phi^2',            'a0':  1.30, 'theta_i': 1.5, 'n': 0.9},
    8:   {'label': 'exp(a0*phi)',           'a0': -0.30, 'theta_i': 0.6, 'n': 0.9},
    15:  {'label': 'a0*exp(-phi)',          'a0':  0.70, 'theta_i': 1.0, 'n': 0.9},
    33:  {'label': 'a0*phi^2',              'a0':  0.30, 'theta_i': 1.5, 'n': 0.9},
    # 36:  {'label': '|a0|/phi',              'a0':  1.30, 'theta_i': 1.0, 'n': 0.9},
    # 37:  {'label': '|a0|**exp(phi)',        'a0':  0.30, 'theta_i': 1.5, 'n': 0.9},
    # 41:  {'label': '|sin(a0)|**phi',        'a0':  1.30, 'theta_i': 1.5, 'n': 0.9},
    43:  {'label': 'phi^2*|a0|',            'a0':  0.30, 'theta_i': 1.5, 'n': 0.9},
    44:  {'label': 'exp(phi)*|a0|',         'a0':  0.70, 'theta_i': 1.0, 'n': 0.5},
    53:  {'label': 'sin(a0)/phi',           'a0':  1.30, 'theta_i': 1.0, 'n': 0.9},
    60:  {'label': 'sin(a0/phi)',           'a0':  1.30, 'theta_i': 1.5, 'n': 0.9},
    # 67:  {'label': '|a0|**sin(phi)',        'a0':  1.30, 'theta_i': 0.6, 'n': 0.9},
    69:  {'label': 'a0 + sin(phi)',         'a0':  1.30, 'theta_i': 1.0, 'n': 0.9},
    # 78:  {'label': '|a0|**(phi^2)',         'a0':  0.70, 'theta_i': 1.5, 'n': 0.9},
    79:  {'label': 'exp(-phi)*|a0|',        'a0':  0.70, 'theta_i': 1.0, 'n': 0.9},
    # 81:  {'label': 'exp(|a0|**phi)',        'a0':  0.70, 'theta_i': 1.5, 'n': 0.5},
    88:  {'label': 'a0 + phi^2',            'a0':  0.70, 'theta_i': 1.0, 'n': 0.9},
    # 95:  {'label': 'exp(a0/phi)',           'a0':  0.30, 'theta_i': 1.5, 'n': 0.5},
    98:  {'label': 'a0*exp(phi)',           'a0':  0.70, 'theta_i': 1.0, 'n': 0.5},
    106: {'label': 'a0 + exp(phi)',         'a0':  0.30, 'theta_i': 1.0, 'n': 0.9},
    # 109: {'label': 'a0/phi^2',              'a0':  1.30, 'theta_i': 1.5, 'n': 0.9},
    110: {'label': '(a0 - phi)^2',          'a0': -0.30, 'theta_i': 1.0, 'n': 0.9},
    121: {'label': '|a0|**(2*phi)',         'a0':  0.70, 'theta_i': 0.6, 'n': 0.9},
    # 123: {'label': 'a0 - sin(phi)',         'a0':  1.30, 'theta_i': 0.6, 'n': 0.9},
    125: {'label': '(a0 + phi)^2',          'a0':  0.30, 'theta_i': 1.0, 'n': 0.9},
}


def build_info(index, fiducial, variable_params, fixed_params, phase, mcmc_samples=500):
    # phase: 'evaluate' (chi^2 at fiducial), 'mcmc' (short exploratory chain), or
    # 'minimize' (bobyqa refinement). mcmc and minimize share one output prefix so the
    # minimiser auto-loads the chain's covariance and starts from its MAP.
    ref = fiducial
    info = {
        "theory": {
            "camb": {
                "path": CAMB_PATH,
                "stop_at_error": False,
                "extra_args": {
                    "halofit_version": "mead",
                    "bbn_predictor": "PArthENoPE_880.2_standard.dat",
                    "lens_potential_accuracy": 1,
                    "num_massive_neutrinos": 1,
                    "nnu": 3.046,
                    "dark_energy_model": 'QuintessenceInterp',
                    "esr_functions_file": ESR_FILE,
                    "esr_potential_index": index,
                },
            }
        },
        "likelihood": {
            "cmb_lite_3d": None,
            "bao.desi_dr2.desi_bao_all": None,
            "sn.union3": None,
        },
        "params": {
            "omk": 0.,
            "mnu": 0.06,
            "omch2": {"latex": r"\Omega_\mathrm{c} h^2", "prior": {"min": 0.11, "max": 0.13},
                      "ref": {"dist": "norm", "loc": 0.1191, "scale": 0.001}, "proposal": 0.0001},
            "ombh2": {"latex": r"\Omega_\mathrm{b} h^2", "prior": {"min": 0.018, "max": 0.025},
                      "ref": {"dist": "norm", "loc": 0.02228, "scale": 0.0001}, "proposal": 0.0001},
            "H0":    {"latex": r"H_0", "prior": {"min": 64, "max": 68},
                      "ref": {"dist": "norm", "loc": 66.0, "scale": 0.25}, "proposal": 0.05},
            "theta_i": {"latex": r"\phi_i", "prior": {"min": 0., "max": 5.},
                        "ref": {"dist": "norm", "loc": ref['theta_i'], "scale": 0.1}, "proposal": 0.1},
            "n":     {"latex": r"\lambda", "prior": {"min": -3, "max": 3},
                      "ref": {"dist": "norm", "loc": ref['n'], "scale": 0.1}, "proposal": 0.05},
            "chi2__BAO": {"latex": r"\chi^2_\mathrm{BAO}", "derived": True},
            "chi2__CMB": {"latex": r"\chi^2_\mathrm{CMB}", "derived": True},
            "chi2__SN":  {"latex": r"\chi^2_\mathrm{SN}",  "derived": True},
            "chi2_total": {"latex": r"\chi^2_\mathrm{total}",
                           "derived": "lambda chi2__BAO, chi2__SN, chi2__CMB: "
                                      "chi2__BAO + chi2__SN + chi2__CMB"},
            "omegam": {"latex": r"\Omega_\mathrm{m}",
                       "derived": "lambda omch2, ombh2, H0: (omch2 + ombh2) / (H0 / 100.0)**2"},
            "rdrag":    {"latex": r"r_\mathrm{drag}", "derived": True},
            "hrd":      {"latex": r"hr_d", "derived": "lambda H0, rdrag: H0/100 * rdrag"},
            "thetastar":{"latex": r"\theta_*", "derived": True},
        },
        "output": os.path.join(OUTPUT_ROOT, f"{RUNNAME}_compl{COMPLEXITY}", str(index), "results"),
    }

    # ESR function parameters: sampled ones get a prior (ref at the fiducial), fixed ones pinned to 0.
    for p in variable_params:
        loc = fiducial.get(p, 0.0)
        info['params'][f"esr_param_{p}"] = {
            "latex": rf"\mathrm{{{p}}}", "prior": {"min": -5.0, "max": 5.0},
            "ref": {"dist": "norm", "loc": loc, "scale": 0.1}, "proposal": 0.1}
    for p in fixed_params:
        info['params'][f"esr_param_{p}"] = 0.0

    if phase == 'evaluate':
        # Evaluate chi^2 once at the fiducial: pin every sampled quantity to its ref value.
        info['sampler'] = {"evaluate": None}
        for name in ("omch2", "ombh2", "H0", "theta_i", "n"):
            info['params'][name] = info['params'][name]['ref']['loc']
        for p in variable_params:
            info['params'][f"esr_param_{p}"] = fiducial.get(p, 0.0)
        info['output'] = None  # don't write chains for a smoke test
    elif phase == 'mcmc':
        # Short exploratory chain: caps at mcmc_samples (single chain, so R-1 won't trigger);
        # learn_proposal builds a covariance the minimiser will reuse.
        info['sampler'] = {"mcmc": {
            "max_samples": mcmc_samples,
            "burn_in": 0,
            "learn_proposal": True,
            # DE tuning legitimately rejects some proposals (-inf), and the thetastar constraint is
            # tight, so allow retries -- but abort promptly if truly stuck so we fall through to
            # minimize instead of spinning.
            "max_tries": 1000,
        }}
    elif phase == 'minimize':
        info['sampler'] = {"minimize": {"method": "bobyqa", "best_of": 4,
                                        "override_bobyqa": {"seek_global_minimum": True}}}
    else:
        raise ValueError(f"unknown phase {phase!r}")
    return info


def run_one(index, test, force, debug, mcmc_samples):
    spec = SELECTED[index]
    fd = load_esr_function_string(ESR_FILE, index, verbose=False)
    if not fd['valid']:
        print(f"[{index}] invalid function, skipping")
        return
    variable_params = fd['variable_params']
    fixed_params = fd['fixed_params']
    print(f"\n=== index {index}: {spec['label']} ===")
    print(f"    {fd['func_string']}   sampled params: {variable_params or 'none'}")

    if test:
        info = build_info(index, spec, variable_params, fixed_params, 'evaluate')
        try:
            _, sampler = run(info, force=force, debug=debug, packages_path=PACKAGES_PATH)
            logpost = sampler.products()["sample"]["minuslogpost"][0]
            print(f"    evaluate OK: -logpost = {logpost:.3f} (finite => pipeline works)")
        except Exception as e:
            print(f"    ERROR for index {index}: {e}")
        return

    # Two-phase: short MCMC to explore + build a covariance, then bobyqa minimize which
    # (sharing the output prefix) auto-starts from the chain's MAP and uses its covmat.
    out_prefix = os.path.join(OUTPUT_ROOT, f"{RUNNAME}_compl{COMPLEXITY}", str(index), "results")
    if os.path.exists(out_prefix + '.minimum.txt') and not force:
        print(f"    minimum already exists ({out_prefix}.minimum.txt); use --force to redo. skipping.")
        return
    # Phase 1: MCMC. force=True gives a clean directory (also clears any stale .minimum so the
    # subsequent minimize -- which cannot resume -- starts fresh). Non-fatal: the CMBlite
    # thetastar constraint is razor-thin, so without a good covmat the chain can stall; if it
    # does, we still run minimize (bobyqa handles the tight likelihood from the fiducial).
    print(f"    [1/2] MCMC (<= {mcmc_samples} samples) ...")
    try:
        mcmc_info = build_info(index, spec, variable_params, fixed_params, 'mcmc', mcmc_samples)
        run(mcmc_info, force=force, debug=debug, packages_path=PACKAGES_PATH)
        seeded = "seeded from MCMC"
    except Exception as e:
        print(f"    MCMC did not complete ({str(e).splitlines()[0]}); minimising from the fiducial.")
        seeded = "from fiducial"
        # remove the partial/empty chain so minimize starts cleanly from the fiducial ref
        import glob
        for f in glob.glob(out_prefix + '.*'):
            if not f.endswith(('.input.yaml', '.updated.yaml')):
                try:
                    os.remove(f)
                except OSError:
                    pass

    # Phase 2: minimize, same output prefix. Do NOT force (that would delete the chain). If the
    # MCMC produced a (partial) chain, minimize auto-seeds from its MAP + covmat; otherwise it
    # starts from the fiducial ref plus best_of random restarts.
    print(f"    [2/2] minimize (bobyqa, {seeded}) ...")
    try:
        min_info = build_info(index, spec, variable_params, fixed_params, 'minimize')
        run(min_info, force=False, debug=debug, packages_path=PACKAGES_PATH)
        print(f"    done -> {out_prefix}.minimum.txt")
    except Exception as e:
        print(f"    ERROR minimising index {index}: {str(e).splitlines()[0]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", action="store_true",
                    help="Evaluate chi^2 once at each fiducial instead of the MCMC+minimize run")
    ap.add_argument("--only", type=int, default=None,
                    help="Run just this index from the selected set")
    ap.add_argument("--mcmc-samples", type=int, default=500,
                    help="Max samples for the exploratory MCMC phase (default 500)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing output")
    ap.add_argument("--debug", action="store_true", help="Cobaya debug output")
    args = ap.parse_args()

    if args.only is not None and args.only not in SELECTED:
        raise SystemExit(f"index {args.only} is not in the selected set: {list(SELECTED)}")

    indices = [args.only] if args.only is not None else list(SELECTED)
    print(f"ESR file: {ESR_FILE}")
    mode = 'evaluate (test)' if args.test else f'MCMC(<={args.mcmc_samples}) + minimize'
    print(f"Mode: {mode};  functions: {indices}")
    for idx in indices:
        run_one(idx, args.test, args.force, args.debug, args.mcmc_samples)


if __name__ == "__main__":
    main()
