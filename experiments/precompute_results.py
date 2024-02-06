from mm79.train_modules.utils import get_results_sweep
from mm79.utils.utils import str2bool
from argparse import ArgumentParser
"""
THIS SCRIPT IS USED TO PRECOMPUTE THE RESULTS OF THE AR SWEEP SO THE NOTEBOOK ONLY HAS TO READ THE RESULTS FROM THE DATAFRAME.
We can specifiy difffernet t_cond and t_horizon to evaluate and they will be saved in the dataframe.
"""


def get_results(sweep_name, t_cond, t_horizon, subgroup_strat, force_recompute, MM1):

    constraints = {"fold": [0, 1, 2, 3, 4], "emission_proba": [False, True]}
    evaluation_params = {"t_cond": t_cond,
                         "t_horizon": t_horizon, "subgroup_strat": subgroup_strat}
    return get_results_sweep(sweep_name, constraints=constraints,
                             evaluation_params=evaluation_params, force_recompute=force_recompute, MM1=MM1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--t_cond', type=int,
                        help='t_cond value')
    parser.add_argument('--t_horizon', type=int,
                        help='t_horizon value')
    parser.add_argument('--subgroup_strat', type=str,
                        help='the subgroup strategy to use')
    parser.add_argument('--sweep_name', type=str,
                        help='the name of the sweep to use')
    parser.add_argument('--force_recompute', type=str2bool,
                        help='wether to force recomputing the results.')
    parser.add_argument('--MM1', type=str2bool,
                        help='wether to evaluate on the MM1 dataset.', default=False)

    args = parser.parse_args()
    get_results(**vars(args))
