#!/usr/bin/env python3
"""
Run Cobaya minimization for a single ESR potential function to test
the phi0 self-consistency tuning implementation.

Usage:
    python test_single_minimize.py --index 15 --complex 5
"""
import sys
sys.path.insert(0, '/Users/giuliaborghetto/CAMB_V_phi_interp')

import camb
#camb.set_feedback_level(1)

from cobaya.run import run
from cobaya.mpi import get_mpi_comm, more_than_one_process, is_main_process
from camb.dark_energy import load_esr_function_string
import argparse
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def create_cobaya_info_dict(esr_functions_file, potential_function_index,
                             complexity, runname,
                             esr_sampled_params=[], esr_fixed_params=[]):

    info = {
        "theory": {
            "camb": {
                "path": '/Users/giuliaborghetto/CAMB_V_phi_interp',
                "stop_at_error": False,
                "extra_args": {
                    "halofit_version": "mead",
                    "bbn_predictor": "PArthENoPE_880.2_standard.dat",
                    "lens_potential_accuracy": 1,
                    "num_massive_neutrinos": 1,
                    "nnu": 3.046,
                    "dark_energy_model": 'QuintessenceInterp',
                    "esr_functions_file": esr_functions_file,
                    "esr_potential_index": potential_function_index,
                }
            }
        },

        "likelihood": {
            "cmb_lite_3d": None,
            "bao.desi_dr2.desi_bao_all": None,
            "sn.union3": None
        },

        "params": {
            # Fixed
            "omk": 0.,
            "mnu": 0.06,
            #"V1": 4.67e-08,

            # Sampled
            "omch2": {
                "latex": r"\Omega_\mathrm{c} h^2",
                "prior": {"min": 0.11, "max": 0.13},
                "ref": {"dist": "norm", "loc": 1.191396612e-01, "scale": 0.001},
                "proposal": 0.0001
            },
            "ombh2": {
                "latex": r"\Omega_\mathrm{b} h^2",
                "prior": {"min": 0.018, "max": 0.025},
                "ref": {"dist": "norm", "loc": 2.227819962e-02, "scale": 0.0001},
                "proposal": 0.0001
            },
            "H0": {
                "latex": r"H_0",
                "prior": {"min": 64, "max": 68},
                "ref": {"dist": "norm", "loc": 6.595882213e+01, "scale": 0.25},
                "proposal": 0.05
            },
            "theta_i": {
                "latex": r"\phi_i",
                "prior": {"min": 0., "max": 5.},
                "ref": {"dist": "norm", "loc": 1., "scale": 0.5},
                "proposal": 0.1
            },
            "n": {
                "latex": r"n",
                "prior": {"min": -3, "max": 3},
                "ref": {"dist": "norm", "loc": 0.5, "scale": 0.05},
                "proposal": 0.05
            },

            # Derived
            "chi2__BAO": {"latex": r"\chi^2_\mathrm{BAO}", "derived": True},
            "chi2__CMB": {"latex": r"\chi^2_\mathrm{CMB}", "derived": True},
            "chi2__SN":  {"latex": r"\chi^2_\mathrm{SN}",  "derived": True},
            "chi2_total": {
                "latex": r"\chi^2_\mathrm{total}",
                "derived": "lambda chi2__BAO, chi2__SN, chi2__CMB: chi2__BAO + chi2__SN + chi2__CMB"
            },
            "omegam": {
                "latex": r"\Omega_\mathrm{m}",
                "derived": "lambda omch2, ombh2, H0: (omch2 + ombh2) / (H0 / 100.0)**2"
            },
            "rdrag":    {"latex": r"r_\mathrm{drag}", "derived": True},
            "hrd":      {"latex": r"hr_d", "derived": "lambda H0, rdrag: H0/100 * rdrag"},
            "thetastar":{"latex": r"\theta_*", "derived": True},
        },

        # "output": f"chains/tests/{runname}/compl_{complexity}/{potential_function_index}",
        "output": f"chains_test_{potential_function_index}/{potential_function_index}",

        "sampler": {
            # 'mcmc': {
            # 'drag': False,
            # 'oversample_power': 0.4,
            # 'proposal_scale': 1.9,
            # 'Rminus1_stop': 0.1,
            # 'Rminus1_cl_stop': 0.2,
            # 'max_tries': 100,
            # 'max_samples': 1000,
            # }
            "minimize": {
                "method": "bobyqa",
                "best_of": 6,
                "override_bobyqa": {"seek_global_minimum": True}
            }
        }
    }

    # Add ESR parameters
    for param in esr_sampled_params:
        info['params'][f"esr_param_{param}"] = {
            "latex": f"\\mathrm{{{param}}}",
            "prior": {"min": -5.0, "max": 5.0},
            "ref": {"dist": "norm", "loc": 0.0, "scale": 1.0},
            "proposal": 0.1
        }
    for param in esr_fixed_params:
        info['params'][f"esr_param_{param}"] = 0.0

    return info


def run_single_potential(esr_functions_file, potential_function_index,
                          complexity, runname, force=False, debug=False):

    function_dict = load_esr_function_string(
        esr_functions_file, potential_function_index, verbose=True)

    if not function_dict['valid']:
        print(f"Skipping invalid function at index {potential_function_index}")
        return

    func_string     = function_dict['func_string']
    param_symbols   = [str(p) for p in function_dict['param_symbols']]
    params_to_fix   = function_dict['fixed_params']
    variable_params = function_dict['variable_params']

    print(f"\nFunction: {func_string}")
    print(f"Parameters: {param_symbols},  fixed: {params_to_fix},  variable: {variable_params}")

    info = create_cobaya_info_dict(
        esr_functions_file, potential_function_index,
        complexity, runname,
        esr_sampled_params=variable_params,
        esr_fixed_params=params_to_fix
    )

    min_file_path = os.path.join(info['output'], 'minimum.txt')
    if os.path.exists(min_file_path) and not force:
        print(f"Minimum already exists at {min_file_path}, skipping.")
        return

    print(f"Running minimization for index {potential_function_index}...")
    try:
        updated_info, sampler = run(info, force=force, debug=debug,
                                    stop_at_error=False)
        print(f"Minimization complete for index {potential_function_index}")
    except Exception as e:
        print(f"Error during minimization: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index",   type=int, required=True,
                        help="Single ESR function index to run")
    parser.add_argument("--complex", type=int, default=5,
                        help="Complexity of the ESR function library")
    parser.add_argument("--force",   action="store_true", default=False,
                        help="Force overwrite of existing output")
    parser.add_argument("--debug",   action="store_true", default=False,
                        help="Debug mode")
    args = parser.parse_args()

    runname = "V_maths"
    complexity = args.complex
    esr_functions_file = (
        f'/Users/giuliaborghetto/CAMB_V_phi_interp/function_library/'
        f'{runname}/compl_{complexity}/unique_equations_{complexity}.txt'
    )

    print(f"ESR file: {esr_functions_file}")
    print(f"Running index {args.index}, complexity {complexity}")

    run_single_potential(
        esr_functions_file,
        potential_function_index=args.index,
        complexity=complexity,
        runname=runname,
        force=args.force,
        debug=args.debug,
    )