import wandb
import subprocess
import argparse
import yaml

# parse arguments
parser = argparse.ArgumentParser(description='inputs')
parser.add_argument('--YAML_FILE', type=str)
parser.add_argument('--PROJ_NAME', type=str)
args = parser.parse_args()

# gather nodes allocated to current slurm job
result = subprocess.run(
    ['scontrol', 'show', 'hostnames'], stdout=subprocess.PIPE)
node_list = result.stdout.decode('utf-8').split('\n')[:-1]


def run(args):
    """ https://github.com/elyall/wandb_on_slurm
    Initialize and run wandb sweep on slurm

    Args:
        YAML_FILE:    name of config yaml file 
        PROJ_NAME:      name of sweep project
    Returns:
        None -> write file 
    """
    # extract yaml file to sweep
    with open(args.YAML_FILE) as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)

    # initialize sweep and get sweep id
    wandb.init(project=args.PROJ_NAME)
    sweep_id = wandb.sweep(config_dict, project=args.PROJ_NAME)

    # run on slurm
    sp = []
    for node in node_list:
        sp.append(subprocess.Popen(['srun',
                                    '--nodes=1',
                                    '--ntasks=1',
                                    '-w',
                                    node,
                                    'start_agent.sh',  # written in build_sh.py
                                    sweep_id,
                                    args.PROJ_NAME]))
    exit_codes = [p.wait() for p in sp]  # wait for processes to finish
    return exit_codes


if __name__ == '__main__':
    run(args)
