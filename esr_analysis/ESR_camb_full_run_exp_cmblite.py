#!/usr/bin/env python3

"""
Python script to run a Cobaya MCMC analysis for a Quintessence model using CMB-lite.

This script defines the model, parameters, likelihoods, and sampler settings
for a cosmological analysis using a custom Quintessence theory code,
and a combination of CMB-lite, BAO, and Supernova likelihoods.

Configured for custom_DE_exp runname.
"""
import sys
sys.path.insert(0, '/scratch/s.2362709/CAMB_V_phi_interp')

from cobaya.run import run
from cobaya.mpi import get_mpi_comm, more_than_one_process, is_main_process
import sympy
from camb.dark_energy import load_esr_function_string
import argparse
#from scipy import stats

#suppress runtime warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def create_cobaya_info_dict(esr_functions_file, potential_function_index, esr_sampled_params = [],esr_fixed_params=[]):
    """
    Create the Cobaya info dictionary with the necessary settings.
    This includes the theory, likelihoods, parameters, and sampler settings.
    """
    # Define the dictionary with all the settings for the run

    # Define the dictionary with all the settings for the run
    info = {
        # Theory
        "theory": {
            "camb": {
                "path": '/scratch/s.2362709/CAMB_V_phi_interp',
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

        # Likelihoods
        "likelihood": {
            "cmb_lite_3d": None,
            "bao.desi_dr2.desi_bao_all": None,
            "sn.union3": None
        },

        # Parameters
        "params": {
            # Fixed parameters
            "omk": 0.,
            'mnu': 0.06,

            # Sampled parameters
            "omch2": {
                "latex": r"\Omega_\mathrm{c} h^2",
                "prior": {
                    "min": 0.11,
                    "max": 0.13
                },
                "ref": {
                    "dist": "norm",
                    "loc": 1.191396612e-01,
                    "scale": 0.001
                },
                "proposal": 0.0001
            },
            "ombh2": {
                "latex": r"\Omega_\mathrm{b} h^2",
                "prior": {
                    "min": 0.018,
                    "max": 0.025
                },
                "ref": {
                    "dist": "norm",
                    "loc": 2.227819962e-02,
                    "scale": 0.0001
                },
                "proposal": 0.0001
            },
            "H0": {
                "latex": r"H_0",
                "prior": {
                    "min": 64,
                    "max": 68
                },
                "ref": {
                    "dist": "norm",
                    "loc": 6.595882213e+01,
                    "scale": 0.25
                },
                "proposal": 0.05
            },

            "theta_i": {
                "latex": r"\phi_i",
                "prior": {
                    "min": 0.,
                    "max": 5.,
                },
                "ref": {
                    "dist": "norm",
                    "loc": 0.,
                    "scale": 0.5
                },
                "proposal": 0.1
            },
            "n": {
                "latex": r"\lambda",
                "prior": {"min": -3, "max": 3},
                "ref": {"dist": "norm", "loc": 0.5, "scale": 0.05},
                "proposal": 0.05
            },
            # Note: the old c0/c1 knobs (second scale-factor power a**c1) are gone -- the
            # analytic interface has no such field, so they are not sampled.

            # Derived parameters
            "chi2__BAO": {
                "latex": r"\chi^2_\mathrm{BAO}",
                "derived": True
            },
            "chi2__CMB": {
                "latex": r"\chi^2_\mathrm{CMB}",
                "derived": True
            },
            "chi2__SN": {
                "latex": r"\chi^2_\mathrm{SN}",
                "derived": True
            },
            "chi2_total": {
                "latex": r"\chi^2_\mathrm{total}",
                "derived": "lambda chi2__BAO, chi2__SN, chi2__CMB: chi2__BAO + chi2__SN + chi2__CMB"
            },
            "omegam": {
                "latex": r"\Omega_\mathrm{m}",
                "derived": "lambda omch2, ombh2, H0: (omch2 + ombh2) / (H0 / 100.0)**2"
            },
            "rdrag": {
                "latex": r"r_\mathrm{drag}",
                "derived": True
            },
            "hrd": {
                "latex": r"hr_d",
                "derived": "lambda H0, rdrag: H0/100 * rdrag"
            },
            "thetastar": {
                "latex": r"\theta_*",
                "derived": True
            }
        },

        # Output settings
        "output": f"chains/camb_esr_cmblite/{runname}_new4/compl_{complexity}/{potential_function_index}/results",
    }

    # Update info with ESR potential parameters using esr_param_ prefix
    if esr_sampled_params:
        for param in esr_sampled_params:
            param_name = f"esr_param_{param}"
            info['params'][param_name] = {
                "latex": f"\\mathrm{{{param}}}",
                "prior": {
                    "min": -5.0,
                    "max": 5.0
                },
                "ref": {
                    "dist": "norm",
                    "loc": 0.0,
                    "scale": 1.
                },
                "proposal": 0.1
            }

    if esr_fixed_params:
        for param in esr_fixed_params:
            param_name = f"esr_param_{param}"
            info['params'][param_name] = 0.0  # Fixed to zero

    return info

def run_single_potential(esr_functions_file, potential_function_index, resume, test, force, debug):
    """
    Run a single potential function with the given parameters.
    """
    function_dict = load_esr_function_string(esr_functions_file, potential_function_index,verbose=True)

    is_valid = function_dict['valid']

    if not is_valid:
        if is_main_process():
            print(f"Skipping invalid function at index {potential_function_index}")
        return

    esr_param_names = [str(p) for p in function_dict['param_symbols']]
    params_to_fix = function_dict['fixed_params']
    variable_params = function_dict['variable_params']

    esr_function_string = function_dict['func_string']



    info = create_cobaya_info_dict(esr_functions_file, potential_function_index, esr_sampled_params=variable_params, esr_fixed_params=params_to_fix)

    if is_main_process():
        print(f"Using ESR potential function: {esr_function_string}, with parameters: {esr_param_names}, fixed: {params_to_fix}, variable: {variable_params}")
        print(f"Resume: {resume}, Test: {test}, Force: {force}, Debug: {debug}")
        print(info['params'])

    # mcmc_info = {"sampler": {
    #     "polychord": {
    #         "nlive": '6d',
    #         'precision_criterion': 0.2,
    #         'num_repeats': 'd'
    #     },
    # }}
    # mcmc_info = {"sampler": {
    #         "mcmc": {
    #             "drag": False,
    #             "oversample_power": 0.4,
    #             "proposal_scale": 1.9,
    #             "Rminus1_stop": 0.01,
    #             "Rminus1_cl_stop": 0.2,
    #             "max_tries": 100,
    #             "max_samples": 8000,
    #         }
    #     }
    # }
    # info.update(mcmc_info)

    # Run the sampler with the defined settings
    # try:
    #     updated_info, sampler = run(info, resume=resume, test=test, force=force, debug=debug)
    # except Exception as e:
    #     print(f"Error during Cobaya mcmc run: {e}")

    # =========================================================================
    # STAGE 1: Quick MCMC (500 samples) to find a viable parameter region
    # =========================================================================
    if is_main_process():
        print(f"\nSTAGE 1: Quick MCMC exploration (max 500 samples)")

    info['sampler'] = {
        'mcmc': {
            'drag': False,
            'oversample_power': 0.4,
            'proposal_scale': 1.9,
            'Rminus1_stop': 0.1,
            'Rminus1_cl_stop': 0.2,
            'max_tries': 100,
            'max_samples': 500,
        }
    }

    mcmc_success = False
    try:
        if is_main_process():
            print(f"Starting MCMC exploration...")
        updated_info, sampler = run(info, resume=resume, test=test, force=force, debug=debug)
        if is_main_process():
            print(f"MCMC exploration completed.")
        mcmc_success = True
    except Exception as e:
        if is_main_process():
            print(f"MCMC exploration failed: {e}. Will use default starting points for minimization.")

    # =========================================================================
    # STAGE 2: Minimization (resumes from MCMC output when available)
    # =========================================================================
    if is_main_process():
        print(f"\nSTAGE 2: Minimization")

    minimize_info = {"sampler": {
        "minimize": {
            "method": "bobyqa",
            "best_of": 10,
            "override_bobyqa": {"seek_global_minimum": True}
            }
        }
    }


    try:
        info.update(minimize_info)
        import os
        min_file_path = os.path.join(info['output'], 'minimum.txt')
        if not os.path.exists(min_file_path):
            print(f"Minimum file not found at {min_file_path}. Running minimization...")
            updated_info, sampler = run(info, resume=True, debug=debug, force=force, stop_at_error=False)
    except Exception as e:
        print(f"Error during minimization run: {e}")


# Main execution block
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", action="store_true", help="Resume from previous run", default=False
    )
    parser.add_argument(
        "--test", action="store_true", help="Test run", default=False
    )
    # default True for force per user request; using store_true keeps typical flag behavior
    parser.add_argument(
        "--force", action="store_true", help="Force overwrite of existing chains", default=False
    )
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode (prints more info)", default=False
    )
    parser.add_argument(
        "--complex", type=int, help="Complexity of the ESR functions to use", default=6, required=False
    )
    parser.add_argument(
        "--index1", type=int, help="Start index of the potential function to run", default=0, required=False
    )
    parser.add_argument(
        "--index2", type=int, help="End index of the potential function to run (exclusive)", default=None, required=False
    )

    args = parser.parse_args()
    # 1. Set up parameters for function generation
    complexity = int(args.complex)
    # Use a predefined runname from the ESR library
    runname = "V_maths"
    esr_functions_file = f'/scratch/s.2362709/CAMB_V_phi_interp/function_library/{runname}/compl_{complexity}/unique_equations_{complexity}.txt'
    with open(esr_functions_file, "r") as f:
        all_functions = [line.strip() for line in f.readlines() if line.strip()]
        num_esr_functions = len(all_functions)
    if is_main_process():
        print(f"Found {num_esr_functions} functions in {esr_functions_file}")

    resume = bool(args.resume)
    test = bool(args.test)
    force = bool(args.force)
    debug = bool(args.debug)

    index1 = int(args.index1)
    index2 = min(int(args.index2) if args.index2 is not None else num_esr_functions, num_esr_functions)

    if is_main_process():
        print(f"Running potentials from index {index1} to {index2} (exclusive)")

    for potential_function_index in range(index1, index2):
        if is_main_process():
            print(f"\n\nRunning potential function index: {potential_function_index} of {num_esr_functions}\n")
        if more_than_one_process():
            comm = get_mpi_comm()
            comm.Barrier()
        run_single_potential(esr_functions_file, potential_function_index, resume, test, force, debug)