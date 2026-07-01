import os
import pandas as pd
from pathlib import Path


def load_esr_function_string_simple(file_path, idx=0):
    """Loads a specific function string from a file."""
    try:
        with open(file_path, "r") as f:
            all_functions = [line.strip() for line in f.readlines() if line.strip()]
            if idx >= len(all_functions):
                return f"Index {idx} out of bounds"
            return all_functions[idx]
    except FileNotFoundError:
        return f"File not found: {file_path}"
    except Exception as e:
        return f"Error: {e}"


def load_benchmark_chi2(model_path):
    try:
        min_file_path = Path(model_path + ".minimum.txt")
        if not min_file_path.exists():
            print(f"Warning: Benchmark file not found at {min_file_path}")
            return None
        data = pd.read_csv(min_file_path, sep=r'\s+', comment='#', header=None, engine='python')
        with open(min_file_path, 'r') as f:
            header_line = f.readline().strip()
            column_names = header_line.replace('#', '').strip().split()
        data.columns = column_names
        chi2_total = (data['chi2__cmb_lite_3d'].iloc[0] +
                      data['chi2__bao.desi_dr2.desi_bao_all'].iloc[0] +
                      data['chi2__sn.union3'].iloc[0])
        return chi2_total
    except Exception as e:
        print(f"Warning: Could not load benchmark from {model_path}. Error: {e}")
        return None




def analyze_cobaya_runs(base_directory="./chains/camb_esr_cmblite/V_maths_new3/",
                        chi2_lcdm=None, chi2_cpl=None, top_n=20):

    all_results = []
    runname = "V_maths"

    if not os.path.isdir(base_directory):
        print(f"Error: Base directory not found at '{os.path.abspath(base_directory)}'")
        return

    print(f"Searching for results in: {os.path.abspath(base_directory)}")

    for root, dirs, files in os.walk(base_directory):
        target_filenames = ["results.minimum.txt", "results_rerun.minimum.txt",
                            "results_iminuit.minimum.txt", "results_bobyqa.minimum.txt"]

        for target_filename in target_filenames:
            if target_filename in files:
                min_file_path = Path(root) / target_filename

                try:
                    p = min_file_path
                    potential_index_str = p.parent.name
                    if potential_index_str == 'results':
                        potential_index_str = p.parent.parent.name

                    potential_index = int(potential_index_str)

                    complexity_dir = (p.parent.parent if potential_index_str == p.parent.name
                                      else p.parent.parent.parent)
                    complexity = int(complexity_dir.name.split('_')[-1])

                    if "rerun" in target_filename:
                        file_type = "rerun"
                    elif "iminuit" in target_filename:
                        file_type = "iminuit"
                    elif "bobyqa" in target_filename:
                        file_type = "bobyqa"
                    else:
                        file_type = "original"

                    source_id = f"{runname}/compl_{complexity}/{potential_index}_{file_type}"

                    # Load B(a) function string
                    function_file_path = f'../function_library/{runname}/compl_{complexity}/unique_equations_{complexity}.txt'
                    func_string = load_esr_function_string_simple(function_file_path, idx=potential_index)

                    data = pd.read_csv(min_file_path, sep=r'\s+', comment='#',
                                       header=None, engine='python')

                    if data.empty:
                        continue

                    with open(min_file_path, 'r') as f:
                        header_line = f.readline().strip()
                        column_names = header_line.replace('#', '').strip().split()

                    data.columns = column_names
                    data['source_run'] = source_id
                    data['potential_index'] = potential_index
                    data['complexity'] = complexity
                    data['V_function'] = func_string
                    data['file_type'] = file_type

                    data['chi2_total'] = (data['chi2__cmb_lite_3d'] +
                                          data['chi2__bao.desi_dr2.desi_bao_all'] +
                                          data['chi2__sn.union3'])

                    if chi2_lcdm is not None:
                        data['d_chi2_lcdm'] = data['chi2_total'] - chi2_lcdm
                    if chi2_cpl is not None:
                        data['d_chi2_cpl'] = data['chi2_total'] - chi2_cpl

                    all_results.append(data)

                except Exception as e:
                    print(f"Could not process {min_file_path}. Error: {e}")

    if not all_results:
        print("\nNo valid results found.")
        return

    full_results_df = pd.concat(all_results, ignore_index=True)
    top_results = full_results_df.sort_values(by='chi2_total').head(top_n)
    # Add this temporarily to analyse_esr_results_exp_cmblite.py
    from camb.dark_energy import load_esr_function_string  # the real one, not the simple version

    for _, row in top_results.iterrows():
        idx = int(row['potential_index'])
        complexity = int(row['complexity'])
        function_file_path = f'../function_library/V_maths/compl_{complexity}/unique_equations_{complexity}.txt'

        # What the analysis thinks the function is (raw line lookup)
        simple = load_esr_function_string_simple(function_file_path, idx=idx)
        # What the runner actually validated and ran
        full = load_esr_function_string(function_file_path, idx=idx)

        stored = row['V_function']
        if stored != simple:
            print(f"MISMATCH at index {idx}: stored='{stored}' vs file-line='{simple}'")
        else:
            print(f"OK index {idx}: '{stored}'")

    print("\n" + "="*140)
    print(f"TOP {top_n} V_MATHS CMB-LITE RESULTS")
    print("="*140)

    display_columns = [
        'source_run', 'complexity', 'potential_index', 'file_type',
        'chi2_total', 'd_chi2_lcdm', 'd_chi2_cpl',
        'H0', 'ombh2', 'omch2', 'theta_i', 'n', 'V_function'
    ]

    param_columns = sorted([col for col in top_results.columns if col.startswith('esr_param_')])
    display_columns.extend(param_columns)

    existing_columns = [col for col in display_columns if col in top_results.columns]
    pd.set_option('display.max_rows', top_n)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', None)

    float_formatters = {col: '{:.4g}'.format for col in top_results.select_dtypes(include='float').columns}
    print(top_results[existing_columns].to_string(index=False, formatters=float_formatters))
    print("="*140)

    output_file = os.path.join(base_directory, f"top_{top_n}_results_V_maths_new3.csv")
    top_results[existing_columns].to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    print(f"\nSUMMARY:")
    print(f"Total runs analyzed: {len(full_results_df)}")
    print(f"Best chi2_total: {top_results['chi2_total'].min():.4f}")
    if chi2_lcdm is not None:
        print(f"Best Δχ²_ΛCDM: {top_results['d_chi2_lcdm'].min():.4f}")
    if chi2_cpl is not None:
        print(f"Best Δχ²_CPL: {top_results['d_chi2_cpl'].min():.4f}")

    if len(top_results) > 0:
        best = top_results.iloc[0]
        print(f"\nBEST RESULT:")
        print(f"B(a) function: {best['V_function']}")
        print(f"n = {best['n']:.4f}, theta_i = {best['theta_i']:.4f}")
        print(f"χ²_CMB-lite: {best['chi2__cmb_lite_3d']:.4f}")
        print(f"χ²_BAO:      {best['chi2__bao.desi_dr2.desi_bao_all']:.4f}")
        print(f"χ²_SN:       {best['chi2__sn.union3']:.4f}")
        print(f"χ²_total:    {best['chi2_total']:.4f}")

    print("="*140)
    return top_results


if __name__ == "__main__":

    print("Loading benchmark models...")
    chi2_lcdm = load_benchmark_chi2("./LCDM_CMB_lite")
    chi2_cpl  = load_benchmark_chi2("./CPL_cmblite")
    print(f"ΛCDM χ² = {chi2_lcdm}, CPL χ² = {chi2_cpl}")

    results = analyze_cobaya_runs(chi2_lcdm=chi2_lcdm, chi2_cpl=chi2_cpl, top_n=20)

    if results is not None and len(results) > 0:
        print(f"\nDone. Found {len(results)} top results.")
    else:
        print("\nNo results found.")