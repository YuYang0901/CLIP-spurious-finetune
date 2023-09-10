import argparse
import os

parser = argparse.ArgumentParser(description='Run OWL-ViT detection on ImageNet')
parser.add_argument('--device', type=int, nargs='+', default=[7])
parser.add_argument('--split', choices=['train', 'validation'], default='train')
parser.add_argument('--class_rank_range', type=int, nargs='+', default=[0, 0])
parser.add_argument('--num_classes', type=int, default=5)
parser.add_argument('--command_launcher', type=str, default='multi_gpu')
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--skip_confirmation', action='store_true')
args = parser.parse_args()

device_str = ",".join(map(str, args.device))
os.environ["CUDA_VISIBLE_DEVICES"] = device_str

import copy
import hashlib
import json
import shlex
import shutil
import subprocess
import time

import numpy as np
import torch
import tqdm


def local_launcher(commands):
    """Launch commands serially on the local machine."""
    for cmd in commands:
        subprocess.call(cmd, shell=True)


def multi_gpu_launcher(commands):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    print('WARNING: using experimental multi_gpu_launcher.')
    try:
        # Get list of GPUs from env, split by ',' and remove empty string ''
        # To handle the case when there is one extra comma: `CUDA_VISIBLE_DEVICES=0,1,2,3, python3 ...`
        available_gpus = [x for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x != '']
    except Exception:
        # If the env variable is not set, we use all GPUs
        available_gpus = [str(x) for x in range(torch.cuda.device_count())]
    n_gpus = len(available_gpus)
    procs_by_gpu = [None]*n_gpus

    while len(commands) > 0:
        for idx, gpu_idx in enumerate(available_gpus):
            proc = procs_by_gpu[idx]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this GPU; launch a command.
                if (proc is not None) and (proc.poll() is not None):
                    proc.kill()
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(
                    f'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 {cmd} --device {gpu_idx} --gradcam --gradcam_save', shell=True)
                procs_by_gpu[idx] = new_proc
                break
            else:
                try:
                    proc.communicate(timeout=1)
                except subprocess.TimeoutExpired:
                    continue
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()

REGISTRY = {
    'local': local_launcher,
    'multi_gpu': multi_gpu_launcher
}


class Job:
    NOT_LAUNCHED = 'Not launched'
    INCOMPLETE = 'Incomplete'
    DONE = 'Done'

    def __init__(self, train_args, sweep_output_dir):
        args_str = json.dumps(train_args, sort_keys=True)
        args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
        self.output_dir = os.path.join(sweep_output_dir, args_hash)

        self.train_args = copy.deepcopy(train_args)
        command = ['python', 'tools/explain_predictions_imagenet.py']
        for k, v in sorted(self.train_args.items()):
            if isinstance(v, list):
                v = ' '.join([str(v_) for v_ in v])
            elif isinstance(v, str):
                v = shlex.quote(v)
            command.append(f'--{k} {v}')
        command.append('--compute_mi')
        self.command_str = ' '.join(command)

        if os.path.exists(os.path.join(self.output_dir, 'done')):
            self.state = Job.DONE
        elif os.path.exists(self.output_dir):
            self.state = Job.INCOMPLETE
        else:
            self.state = Job.NOT_LAUNCHED

    def __str__(self):
        job_info = (self.train_args['class_rank_range'])
        return '{}: {} {}'.format(
            self.state,
            self.output_dir,
            job_info)

    @staticmethod
    def launch(jobs, launcher_fn):
        print('Launching...')
        jobs = jobs.copy()
        np.random.shuffle(jobs)
        print('Making job directories:')
        for job in tqdm.tqdm(jobs, leave=False):
            os.makedirs(job.output_dir, exist_ok=True)
        commands = [job.command_str for job in jobs]
        launcher_fn(commands)
        print(f'Launched {len(jobs)} jobs!')

    @staticmethod
    def delete(jobs):
        print('Deleting...')
        for job in jobs:
            shutil.rmtree(job.output_dir)
        print(f'Deleted {len(jobs)} jobs!')


def make_args_list(args):
    args_list = []    
    for class_idx_start in np.arange(args.class_rank_range[0], args.class_rank_range[1], args.num_classes):
        train_args = {}
        train_args['split'] = args.split
        train_args['class_range'] = [str(class_idx_start), str(class_idx_start+args.num_classes)]
        args_list.append(train_args)
    return args_list

def ask_for_confirmation():
    response = input('Are you sure? (y/n) ')
    if not response.lower().strip()[:1] == "y":
        print('Nevermind!')
        exit(0)


if __name__ == "__main__":
    args_list = make_args_list(args)

    jobs = [Job(train_args, args.output_dir) for train_args in args_list]

    for job in jobs:
        print(job)
    print("{} jobs: {} done, {} incomplete, {} not launched.".format(
        len(jobs),
        len([j for j in jobs if j.state == Job.DONE]),
        len([j for j in jobs if j.state == Job.INCOMPLETE]),
        len([j for j in jobs if j.state == Job.NOT_LAUNCHED]))
    )

    to_launch = [j for j in jobs if j.state == Job.NOT_LAUNCHED]
    print(f'About to launch {len(to_launch)} jobs.')
    if not args.skip_confirmation:
        ask_for_confirmation()
    launcher_fn = REGISTRY[args.command_launcher]
    Job.launch(to_launch, launcher_fn)

