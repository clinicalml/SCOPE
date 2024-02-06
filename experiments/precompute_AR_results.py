from mm79.train_modules.utils import get_results_sweep
from multiprocessing import Pool
import subprocess
"""
THIS SCRIPT IS USED TO PRECOMPUTE THE RESULTS OF THE AR SWEEP SO THE NOTEBOOK ONLY HAS TO READ THE RESULTS FROM THE DATAFRAME.
We can specifiy different t_cond and t_horizon to evaluate and they will be saved in the dataframe.
"""
sweep_name = "k5cq332uzw"
force_recompute = False
MM1 = False # Wether to evaluate on MM1 or not.
#constraints = {"fold": [0, 1, 2, 3, 4], "emission_proba": [False,True]}

t_conds = [6] #[1, 6, 12]
t_horizons = [6] #[6, 12]
subgroup_strats = ["myeloma-type"]

for t_cond in t_conds:
    for t_horizon in t_horizons:
        for subgroup_strat in subgroup_strats:

            args_list = [f"--t_cond={t_cond}", f"--t_horizon={t_horizon}", f"--subgroup_strat={subgroup_strat}",
                         f"--force_recompute={force_recompute}", f"--sweep_name={sweep_name}", f"--MM1={MM1}"]
            program = "precompute_results.py"
            process = subprocess.Popen(
                ["python", program]+args_list)
            process.wait()
# process.wait()

#evaluation_params = {"t_cond": 1, "t_horizon": 6, "subgroup_strat":subgroup_strat}
# df = get_results_sweep(sweep_name, constraints=constraints,
#                        evaluation_params=evaluation_params, force_recompute = force_recompute)

#evaluation_params = {"t_cond": 1, "t_horizon": 12, "subgroup_strat":subgroup_strat}
# df = get_results_sweep(sweep_name, constraints=constraints,
#                      evaluation_params=evaluation_params, force_recompute = force_recompute)

#evaluation_params = {"t_cond": 6, "t_horizon": 6,  "subgroup_strat":subgroup_strat}
# df = get_results_sweep(sweep_name, constraints=constraints,
#                      evaluation_params=evaluation_params, force_recompute = force_recompute)

#evaluation_params = {"t_cond": 6, "t_horizon": 12,  "subgroup_strat":subgroup_strat}
# df = get_results_sweep(sweep_name, constraints=constraints,
#                        evaluation_params=evaluation_params, force_recompute = force_recompute)

#evaluation_params = {"t_cond": 12, "t_horizon": 6,  "subgroup_strat":subgroup_strat}
# df = get_results_sweep(sweep_name, constraints=constraints,
#                        evaluation_params=evaluation_params, force_recompute = force_recompute)
