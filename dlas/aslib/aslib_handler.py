""" This is a handler for the Algorithm Selection Library (ASlib), which is
currently available at http://www.coseal.net/aslib/.
It provides highly customized functionality, ommiting several aspects of the
original ASlib. It aims to provide information on the scenarios instance-files,
mapping instances to local files and calculate several measures such as
PAR10-scores or the percentage of solved instances, as well as information about
the distribution of solvers. """

import os
import re
import logging as log
import random
import pickle
import shutil

import arff
import numpy as np
from recordclass import recordclass

class ASlibHandler(object):
    """
    Provides problem-specific functionality for ASlib.
    An instance is defined by the name that is used by the scenario-files.
    Every instance has a local-path that corresponds to the file.
    Every instance has a dict with performance-data, that is saved as a
    dictionary with {solver_name : (status, performance (time), repetitions}.
    """

    def __init__(self, path="ASlib/", instance_path="instances/"):
        """ Initialize an ASlibHandler.

        Args:
        -----
            path: string
                Path to ASlib-scenarios
            instance_path: string
                Path to instances (or where they should be after matching)"""
        self.aslib_path = path
        self.instance_path = instance_path
        self.data = {}  # data[scen][aslib-inst] = (local-path-inst, [solver], cv) = (status, time, repetions))
        self.cutoff_times = {}  # scen -> cutoff

    def load_scenario(self, scen, extension="jpeg", match=None):
        """
        Load scenario data and attempt to match instances in scenario to local
        instance-files.

        Args:
        -----
            scen: string
                Scenario-name
            extension: string
                Extension for the names in aslib, default: jpeg.
            match: dict
                Dictionary with aslib- -> local-instances
        """
        log.info("Loading {} from \"{}\" into memory.".format(scen, self.aslib_path))
        path = os.path.join(os.getcwd(), self.aslib_path, scen, "algorithm_runs.arff")
        log.debug("Using {} with extension {}".format(path, extension))

        # Load scenario-data data[scen][inst][solver] = (status, time, repetions)
        dataset = arff.load(open(path, "r"))
        self.data[scen] = {}

        # Matching instances
        InstanceEntry = recordclass('InstanceEntry', 'local_path solvers CV_fold')
        for row in dataset["data"]:  # row[0] = instance in ASlib, row[1] = #repetions, row[2] = solver, row[3] = runtime, row[4] = status
            if match:
                local_path = match[row[0]]
            else:
                local_path = os.path.join(self.instance_path, scen, row[0])
            # Check if instance exists
            local_path = ".".join([local_path, extension])
            if not os.path.exists(local_path):
                raise FileNotFoundError("ASlib could not locate instance {} from "
                                        "scenario {} in given path {}.".format(
                                        row[0], scen, local_path))
            if row[0] not in self.data[scen]:
                # Instance has not been seen before, simply add (locPath, solverDict, invalid CV-fold)
                self.data[scen][row[0]] = InstanceEntry(local_path, {}, 1)
            # Add to solver-dictionary: (status, runtime, repitions)
            self.data[scen][row[0]][1][row[2]] = (row[4], row[3], row[1])

        log.info("Found {} instances, {} are solved at least once.".format(
            len(self.data[scen]),
            len([a for a in self.data[scen] if self.solved_by_any(scen,a)])))

        # Mark Cross-Validation-folds
        cv_path = os.getcwd() + "/" + self.aslib_path + scen + "/cv.arff"
        cv_data = arff.load(open(cv_path, "r"))
        for row in cv_data["data"]:
            self.data[scen][row[0]][2] = row[2]

        assert len(self.data[scen]) == len(set([a[0] for a in dataset["data"]]))

        # Read cutoff-time
        with open(os.path.join(self.aslib_path, scen, 'description.txt'), 'r') as fh:
            for line in fh.readlines():
                if line.startswith('algorithm_cutoff_time'):
                    self.cutoff_times[scen] = int(line.split(' : ')[1])

        return self.get_instances(scen), self.local_paths(scen, self.get_instances(scen))

    ## Scenario-wise functions
    def getCVfolds(self, scen):
        """ Get cross-validation-folds of scenario. ONLY WITH 10-FOLD!

        Returns:
        --------
            CVs: list of lists of strings
                folds as lists (in list) with aslib-instance-names
        """
        CVs = []
        for f in range(1, 11):
            CVs.append([i for i in self.data[scen] if self.data[scen][i][2] == f])
        return CVs

    def get_instances(self, scen, remove_unsolved=False):
        """
        Return list with all instances in this scenario in lexico-sorted order.

        Args:
        -----
            remove_unsolved: bool
                removes all instances for which no algorithm solved the instance

        Returns:
        --------
            instances: list of strings
                list with instances of the scenario
        """
        instances = self.data[scen].keys()
        if remove_unsolved:
            instances =[i for i in instances if self.solved_by_any(scen, i)]
        return sorted(instances)

    def get_solvers(self, scen):
        """ Return a list with solvers used in (a random instance of) scenario. """
        inst = random.choice(list(self.data[scen].keys()))
        return sorted(self.data[scen][inst][1].keys())

    def solver_stats(self, scen):
        """
        Evaluates statistics over the solvers.
        Returns a list with a string for every solver, containing information on
        par10-values and timeouts.

        Returns:
        --------
            stats: list of string
                statistics as specified above.
        """
        inst = self.get_instances(scen)
        lines = []
        for s in range(len(self.get_solvers(scen))):
            timeouts = (round(1-self.evaluate(scen, inst, [s for x in inst],
                                              "percent_solved")[0], 2))
            lines.append("Solver {}: {} PAR10 with {} unsuccessful runs.".format(
                s, self.evaluate(scen, inst, [s for x in inst], "par10"), timeouts))
        return lines


    ## Instance-wise functions:
    def solved_by_any(self, scen, inst):
        """ Returns True, if instance is solved by any solver. """
        return 1 in self.get_labels(scen, inst, label='status')

    def local_path(self, scen, inst):
        """ Returns the matched local (physical) path of the instance. """
        return self.data[scen][inst][0]

    def local_paths(self, scen, insts):
        """ Returns the matched local (physical) paths of the instances. """
        return [self.local_path(scen, inst) for inst in insts]

    def get_labels(self, scen, inst, label="status"):
        """ Iterate over all solvers (lexico) and requested labels

        Args:
        -----
            inst: string
                path to instance as in aslib.
            label: string
                one of
                    status -> [0, 1, 0, 1] (1 == 'ok')
                    parX   -> [0.23, 1.52, 0.92, 1.5]
                    rep    -> [1, 1, 1, 1]  (#repetitions)
        """
        result = []
        for solver in sorted(self.get_solvers(scen)):
            if label == "status":
                result.append(int(self.data[scen][inst][1][solver][0] == "ok"))
            elif label[:3] == "par":
                par_factor = int(label[3:])
                if self.data[scen][inst][1][solver][0] == "ok":
                    result.append(self.data[scen][inst][1][solver][1])
                else:
                    result.append(self.cutoff_times[scen]*par_factor)
            elif label == "rep":
                result.append(self.data[scen][inst][1][solver][2])
            else:
                raise ValueError("{} not recognised as label-option.".format(label))
        return result

    def mutate_scenario(self, old_scen, new_scen, startwith=None,
            exclude_solvers=[], ext=None):
        """ Creates new scenario from a loaded scenario, modifying the data.

        Args:
        -----
            old_scen, new_scen: strings
                old_scen from which to create the new_scen.
            startwith: string
                if set, only use instances starting with this string
                (recalculate CV-splits...)
            exclude_solvers: list(strings) | list(ints)
                if set, these solvers are excluded (either ints, interpreted as
                index in lexico-order from 0 or string, name)
            (TODO?) ext: string
                if set, rematch instances using only instances with this extension
        """

        if new_scen in self.data:
            raise NameError("{} already used as scenario-name".format(new_scen))

        # Copy scenario
        self.data[new_scen] = self.data[old_scen].deepcopy()

        # Exclude solvers
        for solver in exclude_solvers:
            for inst in self.data[old_scen]:
                del self.data[new_scen][inst]['solvers'][solver]

        if startwith:
            for inst in self.data[old_scen]:
                if not inst.startswith(startwith):
                    self.data[new_scen].pop(inst)

        if ext:
            raise NotImplementedError()
            for inst in self.data[old_scen]:
                if os.path.splitext(inst)[1] != ext:
                    self.data[new_scen][os.path.splitext(inst)[0]+"."+ext] = self.data[new_scen].pop(inst)

        return self.get_instances(new_scen), self.local_paths(new_scen, self.get_instances(new_scen))


    ## Scenario scoring/statistics functions
    def evaluate(self, scen, insts, solver_index, mode="par10",
                 ignore_unsolved = False):
        """
        Evaluates score depending on mode.

        ATTENTION: if mode is "misclassified", solver_index is a list of lists
        with direct output of neural network, i.e. values between 0 and 1.

        Args:
        -----
        scen: string
            name of scenario holding the instances
        insts: list of strings
            list of instance-names, as they exist in the ASlib-data
        solver_index: list of ints
            list of equal length as insts, containing the index of the
            chosen solver (corresponding, ordered!)
        mode: string
            metric to be evaluated, from [parX, percent_solved,
            misclassified], default par10.
        ignore_unsolved: bool
            if True, only consider instances solved at least once

        Returns:
        --------
        results: mean, std
            averaged score, depending on mode
        """
        assert len(insts) == len(solver_index)
        scores = []
        not_evaluated = 0
        for i, s in zip(insts, solver_index):
            if ignore_unsolved and not self.solved_by_any(scen, i):
                not_evaluated += 1
                continue

            if mode[:3] == "par":
                scores.append(self.get_labels(scen, i, label=mode)[s])
            elif mode == "percent_solved":
                scores.append(int(self.get_labels(scen, i, label="status")[s]))
            elif mode == "misclassified":
                # We have the whole range of solver predictions, e.g. [0.1, 0.3, 0.4]
                # Round <0.5 => 0; >0.5 => 1
                rounded_solvers = np.around(s)
                scores.append(sum([int(a != b) for a, b in
                    zip(self.get_labels(scen,i,label="status"), rounded_solvers)]))
            else:
                raise ValueError("{} is not regonized as a parameter for evaluation.".format(mode))
        assert len(insts) > not_evaluated
        return np.mean(scores), np.std(scores)

    def baseline(self, scen, insts=None, mode="par10"):
        """ Calculates a baseline for scenario.

        Args:
        -----
        insts: list of insts
            only consider these instances, if None, consider all
        mode: string
            metric to be evaluated, from [parX, percent_solved,
            misclassified], default par10.

        Returns:
        --------
        baseline: dict of tuples of means and stds
            VBS, BSS, RAND, GOOD_RAND
        """
        log.debug("Using {} for baseline-evaluation of {}.".format(mode, scen))
        baseline = {}
        if insts == None: insts = self.get_instances(scen, remove_unsolved=False)
        baseline["vbs"] = self.get_vbs_score(scen, insts, mode)
        baseline["bss"] = self.get_bss_score(scen, insts, mode)
        baseline["random"] = self.get_random_score(scen, insts, mode)
        baseline["good-random"] = self.get_good_random_score(scen, insts, mode)
        return baseline

    def get_good_random_score(self, scen, inst=None, mode="par10"):
        """ Consider only instances-solver pairs that actually get solved.

        Args:
        -----
        scen: string
            scenario to be evaluated
        inst: None or list of strings
            list of instances to be evaluated (default: all)
        mode: string
            metric to be evaluated, from [parX, percent_solved,
            misclassified], default par10.
        """
        goodRandInst = [[i for l in self.get_labels(scen,i,label="status") if l] for i in inst]
        goodRandSolv = [[l for l in range(len(self.get_solvers(scen))) if self.get_labels(scen,i,label="status")[l]] for i in inst]
        goodRandInst = [a for b in goodRandInst for a in b]
        goodRandSolv = [a for b in goodRandSolv for a in b] # write a function goddamn
        return self.evaluate(scen, goodRandInst, goodRandSolv, mode=mode)

    def get_random_score(self, scen, inst=None, mode="par10"):
        """ Return expected value (mean, std) for random selection by selecting each solver
        once per instance. """
        num_solvers = len(self.get_solvers(scen))
        rand_inst = [a for b in [inst for solver in
            range(num_solvers)] for a in b]
        rand_solv = [a for b in [[i for x in range(len(inst))] for i in
            range(num_solvers)] for a in b]
        return self.evaluate(scen, rand_inst, rand_solv, mode=mode)

    def get_bss_score(self, scen, inst=None, mode="par10"):
        """ Return score (mean, std) for best single solver. """
        if inst == None: inst = self.get_instances(scen)
        # Evaluate over CVs:
        result = [[],[]]  # inst, solvers
        for fold in self.getCVfolds(scen):
            train            = [i for i in inst if i not in fold]  # Only instance names
            solverTimesTrain = np.array([self.get_labels(scen, i, label="par10") for i in train]).transpose()
            b = np.argmin(np.sum(solverTimesTrain, axis=1))  # Index of best solver
            result[0].extend(fold)
            result[1].extend([b for i in fold])
        # TODO log distribution
        return self.evaluate(scen, result[0], result[1], mode=mode)

    def get_vbs_score(self, scen, inst=None, mode='par10'):
        """ Return virtually best score (mean, std), i.e. oracle performance. """
        if inst == None: inst = self.get_instances(scen)
        if mode == 'percent_solved':
            chosen = [self.get_labels(scen, i, mode).index(max(self.get_labels(scen, i, mode))) for i in inst]
        else:
            chosen = [self.get_labels(scen, i, mode).index(min(self.get_labels(scen, i, mode))) for i in inst]
        return self.evaluate(scen, inst, chosen, mode=mode)

    def solver_distribution(self, solvers, scen=None):
        """
        Parameters
        ----------
        solvers : list<int>
            list of chosen solvers

        Returns
        -------
        distribution : string
            distribution of solvers in list
        """
        res = ""
        stats = {s:solvers.count(s) for s in set(solvers)}
        log.debug(stats)
        total = sum(x[1] for x in stats.items())
        for k,v in reversed(sorted(stats.items(), key=lambda x: x[1])):
            if scen: s = self.get_solvers(scen)[k]
            else: s = k
            res += "{}: {} ({}%)|".format(s,v,round(v/float(total)*100,1))
        return res
