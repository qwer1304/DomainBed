# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
import glob
import copy
import warnings

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

if __name__ == "__main__":

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state_all()
        else:
            cuda_rng_state = None

        checkpoint_file = os.path.join(args.output_dir, 'checkpoints/', filename)
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": copy.deepcopy(algorithm.state_dict()),
            "start_step": start_step,
            "optimizer_dict": copy.deepcopy(algorithm.optimizer.state_dict()),
            "rng_dict": {
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": cuda_rng_state,
                "numpy_rng_state": np.random.get_state(),
                "python_rng_state": random.getstate(),
            },
        }
        torch.save(save_dict, checkpoint_file) # Saves an object to a disk file.
        
    def load_checkpoint(filename):
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            
        save_dict = torch.load(filename, weights_only=False, map_location=device) # Loads an object saved with torch.save() from a file.
        """
        dataset.input_shape = save_dict['model_input_shape']
        dataset.num_classes = save_dict['model_num_classes']
        """
        hparams = save_dict['model_hparams']
        algorithm_dict = save_dict['model_dict']
        start_step = save_dict['start_step']
        optimizer_dict = save_dict['optimizer_dict']
        rng_dict = save_dict['rng_dict']

        if False: # To make this work need to distinguish default values vs user passed ones
            # Merge: command-line overrides checkpoint
            cmd_args_dict = vars(args)
            merged_args_dict = {**save_dict['args'], **{k: v for k, v in cmd_args_dict.items() if v is not None}}
        else:
            merged_args_dict = save_dict['args']
        new_args = argparse.Namespace(**merged_args_dict)
        
        return new_args, hparams, algorithm_dict, start_step, optimizer_dict, rng_dict

    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--load_from_checkpoint', action='store_true',    
        help='Resume from checkpoint. Provide extra info in checkpoint_... arguments, if needed.')
    parser.add_argument('--checkpoint_save_step_file_prefix', type=str, default="model_step",   
        help='Filename prefix of the step save checkpoint in output dir. If not given model_step is used.')
    parser.add_argument('--checkpoint_save_final_file', type=str, default="model.pkl",   
        help='Filename of the final save checkpoint. If not given model.pkl is used. Saved in output dir.')
    parser.add_argument('--checkpoint_load_file', type=str, default=None,   
        help='Filename (including path) of the load checkpoint. If not provided, latest model_step*.pkl file in output/checkpoints dir is used.')
    parser.add_argument('--checkpoint_use_current_args', action='store_true',    
        help='Use args from this command line instead from those in the checkpoint.')
    parser.add_argument('--checkpoint_dont_reload_optimizer', action='store_true',    
        help='Dont reload optimzer state from checkpoint.')
    parser.add_argument('--colwidth', type=int, default=12,    
        help='Column width of the print row.')
    parser.add_argument('--print_results_of_last_step', action='store_true',    
        help='Print the last result instead of averaging over all steps from the last print.')
    parser.add_argument('--set_seed_every_epoch', action='store_true',    
        help='Set seeds to the current epoch every epoch start.')
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    from_checkpoint = False
    if args.load_from_checkpoint:
        def latest_file(pattern):
            files = glob.glob(pattern)
            if not files:
                return None
            latest = max(files, key=os.path.getmtime)
            return latest

        if args.checkpoint_load_file is None:
            filename = os.path.join(args.output_dir, 'checkpoints', 'model_step*.pkl')
        else:
            filename = args.checkpoint_load_file # filename + path provided
        filename = latest_file(filename)
        if filename is not None:
            if args.checkpoint_use_current_args:
                _, hparams, algorithm_dict, start_step, otimizer_dict, rng_dict = load_checkpoint(filename)
            else:
               args, hparams, algorithm_dict, start_step, otimizer_dict, rng_dict = load_checkpoint(filename)
            print("Loading from", filename)           
            from_checkpoint = True
        else:
            warnings.warn(f"Warning: Loading from checkpoint, but no file {filename} exists! Defaulting to initial clean state.")
    if not from_checkpoint:
        if (args.checkpoint_load_file is not None) or (args.checkpoint_use_current_args):
            warnings.warn("Checkpoint load related options given, but loading from checkpoint not requested.")
        start_step = 0
        algorithm_dict = None
        optimizer_dict = None
        rng_dict = None
        from_checkpoint = False

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints/'), exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # RNG
    if not from_checkpoint:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device=="cuda":
            torch.cuda.manual_seed_all(args.seed)
    else:
        # Restore RNG states
        torch.set_rng_state(rng_dict['rng_state'].cpu())
        if device=="cuda":
            torch.cuda.set_rng_state_all([t.cpu() for t in rng_dict['cuda_rng_state']])
        np.random.set_state(rng_dict['numpy_rng_state'])
        random.setstate(rng_dict['python_rng_state'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
        
    if not from_checkpoint:
        if args.hparams_seed == 0:
            hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
        else:
            hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
                misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams and ((not from_checkpoint) or args.checkpoint_use_current_args):
        hparams.update(json.loads(args.hparams))

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    # Must come before algorithm initialization
    n_steps = args.steps or dataset.N_STEPS
    hparams["n_steps"] = n_steps

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes, len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        # Model
        algorithm.load_state_dict(algorithm_dict)

        # Optimizer
        if not args.checkpoint_dont_reload_optimizer:
            algorithm.optimizer.load_state_dict(otimizer_dict)

    def move_optimizer_to_device(optimizer, device):
        if hasattr(optimizer, 'move_to_device') and callable(optimizer.move_to_device):
            optimizer.move_to_device(device)

    algorithm.to(device)
    move_optimizer_to_device(algorithm.optimizer, device)

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    last_results_keys = None
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        epoch =  step / steps_per_epoch
        # Set seed every epoch start if requested to ensure 
        if args.set_seed_every_epoch and (epoch % 1 == 0):
            random.seed(int(epoch))
            np.random.seed(int(epoch))
            torch.manual_seed(int(epoch))
            if device=="cuda":
                torch.cuda.manual_seed_all(int(epoch))

        minibatches_device = [(x.to(device), y.to(device))
            for x,y in next(train_minibatches_iterator)]
        if args.task == "domain_adaptation":
            uda_device = [x.to(device)
                for x,_ in next(uda_minibatches_iterator)]
        else:
            uda_device = None
        step_vals = algorithm.update(minibatches_device, uda_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                if args.print_results_of_last_step:
                    results[key] = val[-1]
                else:
                    results[key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                acc = misc.accuracy(algorithm, loader, weights, device)
                results[name+'_acc'] = acc

            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=args.colwidth)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=args.colwidth)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'{args.checkpoint_save_step_file_prefix}{step}.pkl')

    save_checkpoint(args.checkpoint_save_final_file)

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
