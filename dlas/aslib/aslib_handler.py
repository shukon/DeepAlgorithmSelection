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
#from recordclass import recordclass

import arff
import numpy as np

class ASlibHandler(object):
    """
    Provides problem-specific functionality for ASlib.
    An instance is defined by the name that is used by the scenario-files.
    Every instance has a local-path that corresponds to the file.
    Every instance has a dict with performance-data, that is saved as a
    dictionary with {solver_name : (status, performance (time), repetitions}.

    self.instances: a list with all the instance-files found.
    """

    source = ""  # Path from cwd to ASlib
    data = {}    # data[scen][aslib-inst] = (local-path-inst, [solver], cv) = (status, time, repetions))
    instances = {}
    times = {}
    scen_info = {
                "TSP"          : {"d":"TSP", "cutoffTime":3600 ,"state_of_art": None, "bss": 16.97333, "vbs": 10.80658},
                "TSP-MORPHED"  : {"d":"TSP", "cutoffTime":3600 ,"state_of_art": None, "bss": 0, "vbs": 0},
                "TSP-NETGEN"   : {"d":"TSP", "cutoffTime":3600 ,"state_of_art": None, "bss": 0, "vbs": 0},
                "TSP-RUE"      : {"d":"TSP", "cutoffTime":3600 ,"state_of_art": None, "bss": 0, "vbs": 0},
                "TSP-NO-EAXRESTART" : {"d":"TSP", "cutoffTime":3600,"state_of_art": None, "bss": 0, "vbs": 0},
                "TSP-MORPHED-NO-EAXRESTART" : {"d":"TSP", "cutoffTime":3600 ,"state_of_art": None, "bss": 0, "vbs": 0},
                "TSP-NETGEN-NO-EAXRESTART" : {"d":"TSP", "cutoffTime":3600 ,"state_of_art": None, "bss": 0, "vbs": 0},
		"TSP-RUE-NO-EAXRESTART" : {"d":"TSP", "cutoffTime":3600 ,"state_of_art": None, "bss": 0, "vbs": 0}} 

    def __init__(self, path="ASlib/", instance_path="instances/"):
        """ Initialize an ASlibHandler.

        Args:
            path -- string
                Path to ASlib-scenarios
            instance_path -- string
                Path to instances (or where they should be after matching)"""
        self.source = path
        self.instance_path = instance_path

    def load_scenario(self, scen):
        """
        Load scenario data and attempt to match instances in scenario to local
        instance-files.
        """
        # Find all instances in specified instance-directories.
        if not self.instances:
            self._load_all_inst()

        log.info("Loading {} from \"{}\" into memory.".format(scen, self.source))
        path = os.path.join(os.getcwd(), self.source, scen, "algorithm_runs.arff")
        log.debug("Using {}".format(path))

        # Save scenario-data data[scen][inst][solver] = (status, time, repetions)
        dataset = arff.load(open(path, "r"))
        self.data[scen] = {}

        # Matching instances
        InstanceEntry = recordclass('InstanceEntry', 'local_path solvers CV_fold')
        notfound = []
        for row in dataset["data"]:  # row[0] = instance in ASlib, row[1] = #repetions, row[2] = solver, row[3] = runtime, row[4] = status
            local_path = self._match(row[0], scen)
            if local_path:
                if row[0] not in self.data[scen]:
                    # Instance has not been seen before, simply add (locPath, solverDict, invalid CV-fold)
                    self.data[scen][row[0]] = InstanceEntry(local_path, {}, 1)
                # Add to solver-dictionary: (status, runtime, repitions)
                self.data[scen][row[0]][1][row[2]] = (row[4], row[3], row[1])
            else:
                notfound.append(row[0])
        notfound = list(set(notfound))

        log.info("All instances: {}. Solved at least once: {}.".format(
            len(self.data[scen]),
            len([a for a in self.data[scen] if self.solved_by_any(scen,a)])))

        # Report problems
        if len(notfound) > 0:
            error_path = "tmptxt/debug/notFoundDebug.txt"
            log.warning("Only matched {} of {} instances in {}. Saving not found files in \"{}\".".format(
                len(self.data[scen]), len(set([a[0] for a in dataset["data"]])), scen, error_path))
            with open(error_path, "w") as f:
                for n in notfound: f.write(n+"\n")
        else:
            log.info("All {} in {} matched.".format(len(set([a[0] for a in dataset["data"]])), scen))

        # Mark Cross-Validation-folds
        cv_path = os.getcwd() + "/" + self.source + scen + "/cv.arff"
        cv_data = arff.load(open(cv_path, "r"))
        for row in cv_data["data"]:
            self.data[scen][row[0]][2] = row[2]

        # copy all files to their "correct" location according to aslib-file.
        if True:
            self.copy_to_actual_path(scen)

        assert len(self.data[scen]) == len(set([a[0] for a in dataset["data"]]))
        return self.get_instances(scen), self.local_paths(scen, self.get_instances(scen))

    def _load_all_inst(self, inst_dir=os.path.join(os.getcwd(), "instances/"),
            accepted=re.compile(r"^.*\.(?!jpg$|png$)[^.]+$")):
        """ Find all instances in the specified instance-folders and load them
        for reference.

        Args:
            inst_dir : str, directory
                -- defines the path to the directory containing the instances
            accepted : compiled regular expression
                -- exclude unmatching files (i.e. jpgs or pdfs)
         """
        domains = list(set([self.scen_info[scen]['d'] for scen in self.scen_info]))
        # Loop over all domains
        for dom in domains:
            instances = []
            path = os.path.join(inst_dir, dom)
            cwd = os.path.abspath(os.getcwd())
            for root, dirs, files in os.walk(path):
                for f in files:
                    if accepted.match(f):
                        relFile = os.path.join(os.path.relpath(root, cwd), f) # save relative path
                        # Split up and save as 4-tuple (whole name, dir, name, ext)
                        ilocDir, ilocName = os.path.split(relFile)
                        ilocName, ilocExt = os.path.splitext(ilocName)
                        instances.append(tuple([relFile, ilocDir, ilocName, ilocExt]))
            log.info("{} instances found in {}".format(len(instances), path))
            savePath = "tmptxt/debug/{}_debugInst.txt".format(dom)
            log.debug("Saving instance list in {}".format(savePath))
            with open(savePath, "w") as f:
                for i in instances: f.write(i[0]+"\n")
            if dom in self.instances: self.instances[dom].extend(instances)
            else: self.instances[dom] = instances


    ## Scenario-wise functions
    def getCVfolds(self, scen):
        """ Get cross-validation-folds of scenario. ONLY WITH 10-FOLD!

        Returns:
            CVs -- list of lists of strings with aslib-instance-names
        """
        CVs = []
        for f in range(1, 11):
            CVs.append([i for i in self.data[scen] if self.data[scen][i][2] == f])
        return CVs

    def get_instances(self, scen, remove_unsolved=False):
        """
        Return list with all instances in this scenario in lexico-sorted order.

        remove_unsolved -- removes all instances for which no algorithm solved the instance
        """
        instances = self.data[scen].keys()
        if remove_unsolved:
            instances =[i for i in instances if self.solved_by_any(scen, i)]
        return sorted(instances)

    def get_solvers(self, scen):
        """ Return a list with solvers used in (a random instance of) scenario.
        """
        inst = random.choice(list(self.data[scen].keys()))
        return self.data[scen][inst][1].keys()

    def solver_stats(self, scen):
        inst = self.get_instances(scen)
        lines = []
        for s in range(len(self.get_solvers(scen))):
            timeouts = (round(1-self.evaluate(scen, inst, [s for x in inst],
                                              "percent_solved")[0], 2))
            lines.append("Solver {}: {} PAR10 with {} unsuccessful runs.".format(
                s, self.evaluate(scen, inst, [s for x in inst], "par10"), timeouts))
        return lines

    def copy_to_actual_path(self, scen):
        """ Copies all instances for scenario from matched local-paths to the
        aslib-path (in self.instance_path/scen/). """
        basepath = os.path.join(self.instance_path, scen)
        if not os.path.isdir(basepath):
            log.debug("Make {}".format(basepath))
            os.makedirs(basepath)
        loc_act_insts = zip(self.local_paths(scen, self.get_instances(scen)),
                            self.get_instances(scen))
        for loc, act in loc_act_insts:
            shutil.copy(loc, os.path.join(basepath, act))
            self.data[scen][act][0] = os.path.join(basepath, act)


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
            inst: path to instance as in .txt and aslib's
            label:
                status = [0, 1, 0, 1] -- 1 == 'ok'
                parX = [0.23, 1.52, 0.92, 1.5]
                rep = [1, 1, 1, 1]  -- #repetitions
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
                    result.append(self.scen_info[scen]["cutoffTime"]*par_factor)
            elif label == "rep":
                result.append(self.data[scen][inst][1][solver][2])
            else:
                raise ValueError("{} not recognised as label-option.".format(label))
        return result

    def _match(self, inst, scen, ext='.jpeg'):
        """ Matches an instance in a scenario to the local file path.

        Args:
            inst -- instance as specified in ASlib
            scen -- scenario

        Returns:
            local path, if found.
            If no match, returns None.
        """
        domain = self.scen_info[scen]["d"]
        # Split up instance for matching
        iDir, iName = os.path.split(inst)
        iName,iExt = os.path.splitext(iName)
        # locInst : physical (local) instance path <=> inst : aslib-path
        # locInst as 4-tuple (whole path, dir, name, ext):
        locInst = self.instances[domain]
        for tup in locInst:
            i, ilocDir, ilocName, ilocExt = tup
            if (ilocExt == ext and
                    iName == ilocName):
                return i
        return None  # no match

    def mutate_scenario(self, old_scen, new_scen, ext=None, startwith=None,
            exclude_solvers=[]):
        """ Creates new scenario from a loaded scenario, modifying the data.

        Args:
            old_scen, new_scen -- strings
                old_scen from which to create the new_scen.
            ext -- string
                if set, rematch instances using only instaces with this extension
            startwith -- string
                if set, only use instances starting with this string
                (recalculate CV-splits...)
            exclude_solvers -- list(strings) | list(ints)
                if set, these solvers are excluded (either ints, interpreted as
                index in lexico-order from 0 or string, name)
        """
        raise NotImplementedError()

    ## Scenario scoring/statistics functions
    def evaluate(self, scen, insts, solver_index, mode="par10",
                 ignore_unsolved = False):
        """
        Evaluates score depending on mode.

        ATTENTION: if mode is misclassified, solver_index is a list of lists
        with direct output of neural network, i.e. values between 0 and 1.

        Args:
            scen -- name of scenario holding the instances
            insts -- list of instance-names, as they exist in the ASlib-data
            solver_index -- list of equal length, containing the index of the
                            chosen solver (corresponding)
            mode -- domain: [parX, percent_solved, misclassified]
            ignore_unsolved -- If True, only consider instances solved at least once

        Returns: averaged score, depending on mode
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
            insts -- only consider these instances, if None, consider all
            mode -- metric to calculate baseline for (in parX, percent_solved, misclassified)

        Returns:
            as dict: VBS, BSS, RAND, GOOD_RAND
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
        """ Consider only instances-solver pairs that actually get solved. """
        goodRandInst = [[i for l in self.get_labels(scen,i,label="status") if l] for i in inst]
        goodRandSolv = [[l for l in range(len(self.get_solvers(scen))) if self.get_labels(scen,i,label="status")[l]] for i in inst]
        goodRandInst = [a for b in goodRandInst for a in b]
        goodRandSolv = [a for b in goodRandSolv for a in b] # write a function goddamn
        return self.evaluate(scen, goodRandInst, goodRandSolv, mode=mode)

    def get_random_score(self, scen, inst=None, mode="par10"):
        """ Return expected value for random selection by selecting each solver
        once per instance. """
        num_solvers = len(self.get_solvers(scen))
        rand_inst = [a for b in [inst for solver in
            range(num_solvers)] for a in b]
        rand_solv = [a for b in [[i for x in range(len(inst))] for i in
            range(num_solvers)] for a in b]
        return self.evaluate(scen, rand_inst, rand_solv, mode=mode)

    def get_bss_score(self, scen, inst=None, mode="par10"):
        """ Return score for best single solver. """
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
        if inst == None: inst = self.get_instances(scen)
        if mode == 'percent_solved':
            chosen = [self.get_labels(scen, i, mode).index(max(self.get_labels(scen, i, mode))) for i in inst]
        else:
            chosen = [self.get_labels(scen, i, mode).index(min(self.get_labels(scen, i, mode))) for i in inst]
        return self.evaluate(scen, inst, chosen, mode=mode)

    def solver_distribution(self, solvers):
        """ Given a number of chosen solvers, simply analyze the percentage each
        one takes up in the list."""
        res = ""
        stats = {s:solvers.count(s) for s in set(solvers)}
        print(stats)
        total = sum(x[1] for x in stats.items())
        for k,v in reversed(sorted(stats.items(), key=lambda x: x[1])):
            res += "{}: {} ({}%)|".format(k,v,round(v/float(total)*100,1))
        return res

if __name__ == "__main__":
    log.basicConfig(level = log.DEBUG)
    scens = ["TSP"] #, "TSP-MORPHED", "TSP-NETGEN", "TSP-RUE", "TSP-NO-EAXRESTART", "TSP-MORPHED-NO-EAXRESTART", "TSP-NETGEN-NO-EAXRESTART", "TSP-RUE-NO-EAXRESTART"]
    #with open("aslib_loaded.pickle", "rb") as f: a.data = pickle.load(f)
    aslib = ASlibHandler()
    for s in scens:
        aslib.load_scenario(s)
        log.info(aslib.baseline(s))
    with open("aslib_loaded.pickle", "wb") as f: pickle.dump(aslib.data, f)
