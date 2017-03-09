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
    scen_info = {"ASP-POTASSCO" : {"d":"ASP", "cutoffTime":600  ,"state_of_art": 115.5,  "bss": 534.1,   "vbs": 21.3},
                "CSP-2010"     : {"d":"CSP", "cutoffTime":5000 ,"state_of_art": 247.7,  "bss": 1087.4,  "vbs": 107.7},
                "CSP-MZN-2013" : {"d":"CSP", "cutoffTime":1800 ,"state_of_art": None,   "bss": None,    "vbs": None},
                "QBF-2011"     : {"d":"QBF", "cutoffTime":3600 ,"state_of_art": 910.0,  "bss": 9172.3,  "vbs": 95.9},
                "QBF-2014"     : {"d":"QBF", "cutoffTime":900  ,"state_of_art": 0,   "bss": 0,    "vbs": 0},
                "SAT11-HAND"   : {"d":"SAT", "cutoffTime":5000 ,"state_of_art": 4976.1, "bss": 17815.8, "vbs": 478.3},
                "SAT11-INDU"   : {"d":"SAT", "cutoffTime":5000 ,"state_of_art": 5395.9, "bss": 8985.6,  "vbs": 419.9},
                "SAT11-RAND"   : {"d":"SAT", "cutoffTime":5000 ,"state_of_art": 877.5,  "bss": 14938.6, "vbs": 227.3},
                "SAT12-ALL"    : {"d":"SAT", "cutoffTime":1200 ,"state_of_art": 804.5,  "bss": 2967.9,  "vbs": 93.7},
                "SAT12-HAND"   : {"d":"SAT", "cutoffTime":1200 ,"state_of_art": 886.3,  "bss": 3944.2,  "vbs": 113.2},
                "SAT12-INDU"   : {"d":"SAT", "cutoffTime":1200 ,"state_of_art": 774.6,  "bss": 1360.6,  "vbs": 88.1},
                "SAT12-RAND"   : {"d":"SAT", "cutoffTime":1200 ,"state_of_art": 425.5,  "bss": 568.5,   "vbs": 46.9},
                "SAT15-INDU"   : {"d":"SAT", "cutoffTime":3600 ,"state_of_art": None,   "bss": None,    "vbs": None},
                "MAXSAT12-PMS" : {"d":"SAT", "cutoffTime":2100 ,"state_of_art": 166.8,  "bss": 2111.6,  "vbs": 40.7},
                "MAXSAT15-PMS-INDU" : {"d":"SAT", "cutoffTime":1800 ,"state_of_art": None,   "bss": None,    "vbs": None},
                "PROTEUS-2014" : {"d":"CSP", "cutoffTime":3600 ,"state_of_art": 1321.7, "bss": 10756.3, "vbs": 26.3},
                "TSP"          : {"d":"TSP", "cutoffTime":3600 ,"state_of_art": None, "bss": 16.97333, "vbs": 10.80658},
                "TSP-MORPHED"  : {"d":"TSP", "cutoffTime":3600 ,"state_of_art": None, "bss": 0, "vbs": 0},
                "TSP-NETGEN"   : {"d":"TSP", "cutoffTime":3600 ,"state_of_art": None, "bss": 0, "vbs": 0},
                "TSP-RUE"      : {"d":"TSP", "cutoffTime":3600 ,"state_of_art": None, "bss": 0, "vbs": 0},
                "TSP-NO-EAXRESTART" : {"d":"TSP", "cutoffTime":3600,"state_of_art": None, "bss": 0, "vbs": 0},
                "TSP-MORPHED-NO-EAXRESTART" : {"d":"TSP", "cutoffTime":3600 ,"state_of_art": None, "bss": 0, "vbs": 0},
                "TSP-NETGEN-NO-EAXRESTART" : {"d":"TSP", "cutoffTime":3600 ,"state_of_art": None, "bss": 0, "vbs": 0},
                "TSP-RUE-NO-EAXRESTART" : {"d":"TSP", "cutoffTime":3600 ,"state_of_art": None, "bss": 0, "vbs": 0}}


    def __init__(self, path="ASlib/"):
        """ Path is the relative path to the ASlib-folder """
        self.source = path

    def load_scenario(self, scen):
        """
        Loads scenario data and attempts to match instances in scenario to local
        instance-files.
        """
        # Find all instances in specified instance-directories.
        if not self.instances:
            self._load_all_inst()

        log.info("Loading {} from \"{}\" into memory.".format(scen, self.source))
        path = os.path.join(os.getcwd(), self.source, scen, "algorithm_runs.arff")
        log.debug("Using {}".format(path))
        # Save scenario-data data[scen][inst][solver] = (status, time, repitions)
        dataset = arff.load(open(path, "r"))
        self.data[scen] = {}
        # Keep track of matched instances
        aslib_inst, local_inst, notfound = [], [], []
        # row[0] = instance in ASlib, row[1] = num of repitions, row[2] = solver, row[3] = runtime, row[4] = status
        for row in dataset["data"]:  #  dataset["data"] consists of 5-tuples with evaluations
            matched = self._match(row[0], scen)
            if matched:
                if row[0] not in self.data[scen]: # Instance has not been seen before, simply add (locPath, solverDict)
                    self.data[scen][row[0]] = [matched, {}, True, -1]
                    aslib_inst.append(row[0])
                    local_inst.append(matched)
                # data[scen][inst][solver-dict][solver] = (status, runtime, repitions) 
                self.data[scen][row[0]][1][row[2]] = (row[4], row[3], row[1])
            else:
                notfound.append(row[0])
        # Remove duplicates
        notfound = list(set(notfound))

        log.info("All instances: {}. Solved at least once: {}.".format(
            len(self.data[scen]),
            len([a for a in self.data[scen] if self.solved_by_any(scen,a)])))

        # Report problems
        if len(notfound) > 0:
            num = len(aslib_inst)
            error_path = "tmptxt/debug/notFoundDebug.txt"
            log.warning("Only matched {} of {} instances in {}. Saving not found files in \"{}\".".format(
                num, len(set([a[0] for a in dataset["data"]])), scen, error_path))
            with open(error_path, "w") as f:
                for n in notfound: f.write(n+"\n")
        else:
            log.info("All {} in {} matched.".format(len(set([a[0] for a in dataset["data"]])), scen))

        # Mark Cross-Validation-folds
        path = os.getcwd() + "/" + self.source + scen + "/cv.arff"
        cv_data = arff.load(open(path, "r"))
        for row in cv_data["data"]:
            self.data[scen][row[0]][3] = row[2]

        assert len(aslib_inst)==len(set([a[0] for a in dataset["data"]]))
        assert len(aslib_inst)==len(local_inst)
        return aslib_inst, local_inst

    def _load_all_inst(self, inst_dir=os.path.join(os.getcwd(), "instances/"),
            accepted=re.compile("^.*\.(?!jpg$|png$)[^.]+$")):
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
            # TODO copy all instances into new folder according to aslib-path
            # (would help a lot)
            savePath = "tmptxt/debug/{}_debugInst.txt".format(dom)
            log.debug("Saving instance list in {}".format(savePath))
            with open(savePath, "w") as f:
                for i in instances: f.write(i[0]+"\n")
            if dom in self.instances: self.instances[dom].extend(instances)
            else: self.instances[dom] = instances


    ## Scenario-wise functions
    def getCVfolds(self, scen):
        """ Get information of the cross-validation-folds of scenario.

        Returns:
            CVs -- list of lists of strings with aslib-instance-names
        """
        CVs = []
        for f in range(1, 11):
            CVs.append([i for i in self.data[scen] if self.data[scen][i][3] == f])
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

    def baseline(self, scen, insts=None, mode="par10"):
        """ Calculates a baseline for AS.
        Returns PAR-1/Par-10 for:
        Virtual Best Solver (vbs)
        Best Single Solver (bss)
        Worst Single Solver (wss)
        Random Solver (rand)
        """
        log.debug("Using {} for baseline-evaluation of {}.".format(mode, scen))
        baseline = {}
        # If no instances are specified, get all instances from scenario
        if insts == None: insts = self.get_instances(scen, remove_unsolved=False)
        baseline["vbs"] = self.evaluate(scen, insts, self.VBS(scen, insts), mode)
        baseline["bss"] = self.evaluate(scen, insts, self.BSS(scen, insts), mode)
        # To calculate random baseline, calculate expectat]
        num_solvers = len(self.get_solvers(scen))
        # Get all instances as often as there are solversion
        randInst = [a for b in [insts for solver in
            range(num_solvers)] for a in b]
        # Pick each solver once per instance
        randSolv = [a for b in [[i for x in range(len(insts))] for i in
            range(num_solvers)] for a in b]
        baseline["random"] = self.evaluate(scen, randInst, randSolv)
        # Get all instances as often as there are solvers
        goodRandInst = [[i for l in self.get_labels(scen,i,label="status") if l] for i in insts]
        goodRandSolv = [[l for l in range(len(self.get_solvers(scen))) if self.get_labels(scen,i,label="status")[l]] for i in insts]
        goodRandInst = [a for b in goodRandInst for a in b]
        goodRandSolv = [a for b in goodRandSolv for a in b] # write a function goddamn
        baseline["good-random"] = self.evaluate(scen, goodRandInst, goodRandSolv)
        return baseline

    def solver_distribution(self, scen):
        inst = self.get_instances(scen)
        lines = []
        for s in range(len(self.get_solvers(scen))):
            timeouts = (round(1-self.evaluate(scen,inst,[s for x in
                inst],"percent_solved")[0], 2))
            #timeouts = 1800- self.evaluate(scen,inst,[s for x in inst],"percentSolved")
            lines.append("Solver {}: {} PAR10 with {} unsuccessful runs.".format(s,
                self.evaluate(scen,inst,[s for x in inst],"par10"),timeouts))
        return lines


    ## Instance-wise functions:
    def solved_by_any(self, scen, inst):
        """ Returns True, if instance is solved by any solver. """
        for solver in self.data[scen][inst][1]:
            if self.data[scen][inst][1][solver][0] == "ok":
                return True
        return False

    def local_path(self, scen, inst):
        """ Returns the matched local (physical) path of the instance. """
        return self.data[scen][inst][0]

    def get_labels(self, scen, inst, label = "status"):
        """ Iterate over all solvers (lexico) and return list [0,1,1,0,1]
            indicating whether the solver solved the instance.

            Args:
            inst: path to instance as in .txt and aslib's
            label:
                status = [0,1,0,1]
                parX = [0.23, 1.52,...]
                rep = [1,1,1,1]
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

    def _match(self, inst, scen):
        """ Matches an instance in a scenario to the local file path.
        Highly hard-coded, could be handled in a more adaptive way.

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
            if domain == "QBF":
                if ilocName == inst or ilocName == iName+iExt or i == iName: return i
            elif domain == "CSP":
                if ilocName+ilocExt == inst: return i
                if i == inst: return i
            elif scen == "ASP-POTASSCO":
                if i == inst: return i
            elif domain == "SAT":
                if ilocName == iName+iExt: return i
                if ilocName+ilocExt == iName+iExt: return i
            elif domain == "TSP":
                if iName+".jpeg" == ilocName+ilocExt:
                    return i
        return None  # no match

    #def getEndings(self):
    #    for s in sorted(self.data.keys()):
    #        print([i for i in self.data[s] if self.localPath(s,i).endswith("txt")])
    #        endings = [self.data[s][i][0].split(".")[-1] for i in self.data[s]]
    #        log.debug("Endings for {}: {}".format(s, Counter(endings)))

    ## Scenario scoring/statistics functions
    def evaluate(self, scen, insts, solver_index, mode="par10",
                 ignore_unsolved = True):
        """
        Evaluates score depending on mode.

        ATTENTION: if mode is misclassified, solver_index is a list of lists
        with direct output of neural network, i.e. values between 0 and 1.

        Args:
            scen -- name of scenario holding the instances
            insts -- list of instance-names, as they exist in the ASlib-data
            solver_index -- list of equal length, containing the index of the
                            chosen solver (sorted)
            mode -- one in [par1, par10, percent_solved, misclassified]

        Returns: PAR-score
        """
        assert len(insts) == len(solver_index)
        scores = []
        notEvaluated = 0
        for i, s in zip(insts, solver_index):
            if ignore_unsolved and not self.solved_by_any(scen, i):
                notEvaluated += 1
                continue
            if mode[:3] == "par":
                scores.append(self.get_labels(scen, i, label=mode)[s])
            elif mode == "percent_solved":
                scores.append(int(self.get_labels(scen, i, label="status")[s]))
            elif mode == "misclassified":
                # We have the whole range of solver predictions. <0.5 => 0; >0.5 => 1
                rounded_solvers = np.around(s)
                scores.append(sum([int(a!=b) for a, b in
                    zip(self.get_labels(scen,i,label="status"),rounded_solvers)]))
            else: raise ValueError("{} is not regonized as a parameter for evaluation.".format(mode))
        if len(insts) > notEvaluated: return np.mean(scores), np.std(scores)
        else: raise Exception("No instances evaluated. Something terribly wrong.")

    def BSS(self, scen, inst = None, mode = "par10"):
        """ Returns index for bss, respectively. """
        if inst == None: inst = self.get_instances(scen)
        # Evaluate over CVs:
        result = [[],[]]  # inst, solvers
        for fold in self.getCVfolds(scen):
            train            = [i for i in inst if i not in fold]  # Only instance names
            solverTimesTrain = np.array([self.get_labels(scen, i, label="par10") for i in train]).transpose()
            b = np.argmin(np.sum(solverTimesTrain, axis=1))
            result[0].extend(fold)
            result[1].extend([b for i in fold])
        return result[1]

    def VBS(self, scen, inst = None):
        if inst == None: inst = self.get_instances(scen)
        return [self.get_labels(scen, i, "par10").index(min(self.get_labels(scen, i, "par10"))) for i in inst]

    def good_estimate(self, scen, inst = None, mode = "PAR10"):
        """ TODO What is this function doing? """
        if inst == None: inst = self.get_instances(scen)
        # Evaluate over CVs:
        result = [[],[]]  # inst, solvers
        for fold in self.getCVfolds(scen):
            train            = [i for i in inst if i not in fold]  # Only instance names
            solverStatusTrain = np.array([self.get_labels(scen, i, label="status") for i in train])
            solverStatusTrain = np.sum(solverStatusTrain, axis=0)
            b = np.argmax(solverStatusTrain)
            result[0].extend(fold)
            result[1].extend([b for i in fold])
        return result

    def percentageOfSolvers(self, solvers):
        """ TODO What is this function doing? """
        res = ""
        stats = {s:solvers.count(s) for s in set(solvers)}
        print(stats)
        total = sum(x[1] for x in stats.items())
        for k,v in reversed(sorted(stats.items(), key=lambda x: x[1])):
            res += "{}: {} ({}%)|".format(k,v,round(v/float(total)*100,1))
        return res

if __name__ == "__main__":
    log.basicConfig(level = log.DEBUG)
    scens = ["TSP", "TSP-MORPHED", "TSP-NETGEN", "TSP-RUE", "TSP-NO-EAXRESTART", "TSP-MORPHED-NO-EAXRESTART", "TSP-NETGEN-NO-EAXRESTART", "TSP-RUE-NO-EAXRESTART"]
    #with open("aslib_loaded.pickle", "rb") as f: a.data = pickle.load(f)
    aslib = ASlibHandler()
    for s in scens:
        aslib.load_scenario(s)
        log.info(aslib.baseline(s))
    with open("aslib_loaded.pickle", "wb") as f: pickle.dump(aslib.data, f)
