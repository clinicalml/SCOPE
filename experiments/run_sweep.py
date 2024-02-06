import yaml
import itertools
import string
import random
import os
from mm79 import EXPERIMENT_DIR
import subprocess
from argparse import ArgumentParser


def run_sweep(sweep_config):
    with open(f"configs/{sweep_config}") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    keys, values = zip(*data.items())
    permutations_dicts = [dict(zip(keys, v))
                          for v in itertools.product(*values)]

    alphabet = string.ascii_lowercase + string.digits
    sweep_name = ''.join(random.choice(alphabet) for i in range(10))

    with open(os.path.join(EXPERIMENT_DIR, "configs", "records", f'{sweep_name}.yml'), 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    for permutation_dict in permutations_dicts:
        program = permutation_dict.pop("program")
        args_list = [f"--{k}={v}" for k, v in permutation_dict.items()]
        process = subprocess.Popen(
            ["python", program]+args_list+["--sweep_name", sweep_name])
        process.wait()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config_name', type=str,
                        help='name of the config to use (without the .yaml extension)')

    args = parser.parse_args()
    run_sweep(args.config_name+".yaml")
