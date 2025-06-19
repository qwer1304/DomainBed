# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools
import numpy as np
from collections import namedtuple
 
def get_test_records(records):
    """Given records with a common test env, get the test records (i.e. the
    records with *only* that single test env and no other test envs)"""
    return records.filter(lambda r: len(r['args']['test_envs']) == 1)

class SelectionMethod:
    """Abstract class whose subclasses implement strategies for model
    selection across hparams and timesteps."""

    def __init__(self):
        raise TypeError

    @classmethod
    def run_acc(self, test_env, run_records):
        """
        Given records from a run, return a {val_acc, test_acc} dict representing
        the best val-acc and corresponding test-acc for that run.
        """
        raise NotImplementedError

    @classmethod
    def hparams_accs(self, test_env, records):
        """
        Given all records from a single (dataset, algorithm, test env) triplet,
        return a sorted list of (run_acc, records, hparams_seed) tuples.
        """
        # group() returns a list of (group, group_records)
        return (records.group('args.hparams_seed')
            .map(lambda h_seed, run_records:
                (
                    self.run_acc(test_env, run_records),
                    run_records,
                    h_seed
                )
            ).filter(lambda x: x[0] is not None)
            .sorted(key=lambda x: x[0]['val_acc'])[::-1]
        )

    @classmethod
    def sweep_acc(self, test_env, records):
        """
        Given all records from a single (dataset, algorithm, test env) triplet,
        return the mean test acc of the k runs with the top val accs.
        """
        _hparams_accs = self.hparams_accs(test_env, records)
        if len(_hparams_accs):
            Sweep_point = namedtuple("Sweep_point", "seed step")
            sweep_point = Sweep_point(seed=_hparams_accs[0][2], step=_hparams_accs[0][0]['step'])
            return _hparams_accs[0][0]['test_acc'], sweep_point 
        else:
            return None, None

class OracleSelectionMethod(SelectionMethod):
    """Like Selection method which picks argmax(test_out_acc) across all hparams
    and checkpoints, but instead of taking the argmax over all
    checkpoints, we pick the last checkpoint, i.e. no early stopping."""
    name = "test-domain validation set (oracle)"

    @classmethod
    def run_acc(self, test_env, run_records):
        # filters records with a SINGLE test environment
        run_records = run_records.filter(lambda r:
            len(r['args']['test_envs']) == 1)
        if not len(run_records):
            return None
        test_env = run_records[0]['args']['test_envs'][0]
        test_out_acc_key = 'env{}_out_acc'.format(test_env)
        test_in_acc_key = 'env{}_in_acc'.format(test_env)
        chosen_record = run_records.sorted(lambda r: r['step'])[-1]
        return {
            'val_acc':  chosen_record[test_out_acc_key],
            'test_acc': chosen_record[test_in_acc_key],
            'step':     chosen_record['step']
        }

class IIDAccuracySelectionMethod(SelectionMethod):
    """Picks argmax(mean(env_out_acc for env in train_envs))"""
    name = "training-domain validation set"

    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {val_acc, test_acc} dict."""
        test_env = record['args']['test_envs'][0]
        val_env_keys = []
        for i in itertools.count():
            if f'env{i}_out_acc' not in record:
                break
            if i != test_env:
                val_env_keys.append(f'env{i}_out_acc')
        test_in_acc_key = 'env{}_in_acc'.format(test_env)
        return {
            'val_acc':  np.mean([record[key] for key in val_env_keys]),
            'test_acc': record[test_in_acc_key],
            'step':     record['step']
        }

    @classmethod
    def run_acc(self, test_env, run_records):
        # get_test_records filters records with a SINGLE test environment
        test_records = get_test_records(run_records)
        if not len(test_records):
            return None
        return test_records.map(self._step_acc).argmax('val_acc')

class IIDAutoLRAccuracySelectionMethod(SelectionMethod):
    """Picks argmax(mean(env_out_acc for env in train_envs))"""
    name = "auto lr training-domain validation set"

    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {val_acc, test_acc} dict."""
        test_env = record['args']['test_envs'][0]
        val_env_keys = []
        for i in itertools.count():
            if f'env{i}_out_acc' not in record:
                break
            if i != test_env:
                val_env_keys.append(f'env{i}_out_acc')
        test_in_acc_key = 'fd_env{}_in_acc'.format(test_env)
        return {
            'val_acc':  np.mean([record[key] for key in val_env_keys]),
            'test_acc': record[test_in_acc_key],
            'step':     record['step']
        }

    @classmethod
    def run_acc(self, test_env, run_records):
        # get_test_records filters records with a SINGLE test environment
        test_records = get_test_records(run_records)
        if not len(test_records):
            return None
        return test_records.map(self._step_acc).argmax('val_acc')


class LeaveOneOutSelectionMethod(SelectionMethod):
    """Picks (hparams, step) by leave-one-out cross validation."""
    name = "leave-one-domain-out cross-validation"

    @classmethod
    def _step_acc(self, test_env, records):
        """Return the {val_acc, test_acc} for a group of records corresponding
        to a single step."""
        # get_test_records() filters records with a SINGLE test env. How does it work here?
        if False:
            test_records = get_test_records(records)
            if len(test_records) != 1:
                return None
        else:
            test_records = records.filter(lambda r: len(r['args']['test_envs']) == 2)
            if len(test_records) != 1:
                return None

        # Assumes test record is the FIRST one in the pair of envs in a record.
        # Hence, to test different envs, we should have PERMUTATIONS of environments.
        # But sweep produces COMBINATIONS
        #test_env = test_records[0]['args']['test_envs'][0]

        # Find number of envs in the records by argmax_i env{i}_out_acc + 1
        n_envs = 0
        for i in itertools.count():
            if f'env{i}_out_acc' not in records[0]:
                break
            n_envs += 1

        """set val_accs for all envs that are in records with len(args.test_envs)==2
        by taking env{j}_in_acc for j in args.test_envs != test_env.
        Note that this scans ALL records provided with 2 test_envs.
        The records come from a group corresponding to a single test env which is either
        the first or the second in test_envs. So, records of the group corresponding to the
        2nd env in test_envs will be counted again as corresponding to those of the 1st env.
        To calculate val_acc of the 2nd test group it must exist in test_envs when swept
        with that env in the 1st position in test_envs. But sweep generates COMBINATIONS only."""
        val_accs = np.zeros(n_envs) - 1
        for r in records.filter(lambda r: len(r['args']['test_envs']) == 2):
            val_env = (set(r['args']['test_envs']) - set([test_env])).pop()
            val_accs[val_env] = r['env{}_in_acc'.format(val_env)]
        # leave ONLY envs that are NOT test_env
        val_accs = list(val_accs[:test_env]) + list(val_accs[test_env+1:])
        # Expects there're records with ALL other than test environment as the validation one.
        if False:
            if any([v==-1 for v in val_accs]):
                return None
            val_acc = np.sum(val_accs) / (n_envs-1)
        else:
            val_accs = [v for v in val_accs if v != -1]
            if not val_accs: # empty
                return None
            val_acc = np.sum(val_accs) / len(val_accs)
            
        return {
            'val_acc': val_acc,
            'test_acc': test_records[0]['env{}_in_acc'.format(test_env)]
        }

    @classmethod
    def run_acc(self, test_env, records):
        # records are all run records (i.e., for a single dataset, algorithm, test_env, hash_seed)
        # group() returns a list of (group, group_records)
        """step_accs = records.group('step').map(lambda step, step_records:
            {**self._step_acc(test_env, step_records), "step": step}
        ).filter_not_none()"""
        step_accs = records.group('step').map(lambda step, step_records:
            (self._step_acc(test_env, step_records), step)
        )
        step_accs = step.accs.filter(lambda r: r[0] is not None)
        step_accs = step_accs.map(lambda r: {**r[0], "step": r[1]})
        # step_acc() returns a dictionary with val_acc and test_acc keys
        # step_accs is a query (list) of run step_acc() results grouped according to step
        if len(step_accs):
            # argmax returns the dictionary with biggest val_acc.
            return step_accs.argmax('val_acc')
        else:
            return None
