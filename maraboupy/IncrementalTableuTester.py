import os
import subprocess
import threading
from timeit import default_timer as timer
import tempfile

from maraboupy import MarabouCore

BUILD_SOLVE = "build"
BUILD_ADVERSARIAL = "build_adv"
DEFAULT_TIMEOUT = 600


def define_simple_network():
    network = MarabouCore.InputQuery()
    network.setNumberOfVariables(5)

    # x0
    network.setLowerBound(0, 0.1)
    network.setUpperBound(0, 1)

    # x1
    network.setLowerBound(1, 0.1)
    network.setUpperBound(1, 1)

    # y0
    network.setLowerBound(2, 0.1)
    network.setUpperBound(2, 50)

    # y1
    network.setLowerBound(3, 0)
    network.setUpperBound(3, 50)

    # y2
    network.setLowerBound(4, 0)
    network.setUpperBound(4, 50)

    # add equations, y0 will be larger then the rest
    eq = MarabouCore.Equation()
    eq.addAddend(5, 0)
    eq.addAddend(5, 1)
    eq.addAddend(-1, 2)  # y0
    eq.setScalar(0)
    network.addEquation(eq)

    eq = MarabouCore.Equation()
    eq.addAddend(1, 0)
    eq.addAddend(1, 1)
    eq.addAddend(-1, 3)  # y1
    eq.setScalar(0)
    network.addEquation(eq)

    eq = MarabouCore.Equation()
    eq.addAddend(1, 0)
    eq.addAddend(2, 1)
    eq.addAddend(-1, 4)  # y2
    eq.setScalar(0)
    network.addEquation(eq)

    max_idx = 2
    out_idx = [4, 3]
    return network, max_idx, out_idx


def solve_adversarial(query, max_idx, out_idx):
    vars1, stats1 = MarabouCore.solveAdversarial(query, max_idx, out_idx, "", 0)
    if len(vars1) > 0:
        print("SAT")
        print(vars1)
        return False
    else:
        print("UNSAT")
        return True


def run_process(args, cwd, timeout, s_input=None):
    """Runs a process with a timeout `timeout` in seconds. `args` are the
    arguments to execute, `cwd` is the working directory and `s_input` is the
    input to be sent to the process over stdin. Returns the output, the error
    output and the exit code of the process. If the process times out, the
    output and the error output are empty and the exit code is 124."""

    proc = subprocess.Popen(
        args,
        cwd=cwd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    out = ''
    err = ''
    exit_status = 124
    try:
        if timeout:
            timer = threading.Timer(timeout, lambda p: p.kill(), [proc])
            timer.start()
        out, err = proc.communicate(input=s_input)
        exit_status = proc.returncode
    finally:
        if timeout:
            timer.cancel()

    if isinstance(out, bytes):
        out = out.decode()
    if isinstance(err, bytes):
        err = err.decode()
    return (out.strip(), err.strip(), exit_status)

def remove_out_constraints(all_constraints : str):
    lines = all_constraints.splitlines()
    while 'y' in lines[-1]:
        lines = lines[:-1]
    return "\n".join(lines) + "\n"

def timing_executables(network_path, property_path, debug=False):
    print(property_path)
    # The format of the property is origNUM_blabla we extract NUM
    max_output =property_path[property_path.find("orig") + 4][:1]
    if not os.path.exists(network_path) or not os.path.exists(property_path):
        print("One of the files does not exists")
        exit(1)
    with open(property_path, "r") as f:
        all_constraints = f.read()
    input_constraints = remove_out_constraints(all_constraints)


    start_adv = timer()
    with tempfile.NamedTemporaryFile(mode='w') as adv_property:
        adv_args = [os.path.join(BUILD_ADVERSARIAL, "Marabou"), network_path, adv_property.name]
        adv_property.write(input_constraints)
        adv_property.write("OutputMaxIndex = {}".format(max_output))
        adv_property.flush()
        out, err, exit_status = run_process(adv_args, os.curdir, DEFAULT_TIMEOUT)
        end_adv = timer()
        if debug:
            print(out)
        adv_result = 'UNSAT' if 'UNSAT' in out else 'SAT'
        print("finished adv, time: {} seconds, result: {}".format(end_adv - start_adv, adv_result))

    total_solve = 0
    results_solve = []


    for i in range(0, 10):
        if i == max_output:
            continue
        with tempfile.NamedTemporaryFile(mode='w') as cur_property:
            cur_property.write(input_constraints)
            cur_property.write("+y{} -y{} <= 0".format(max_output, i))
            cur_property.flush()

            solve_args = [os.path.join(BUILD_ADVERSARIAL, "Marabou"), network_path, cur_property.name]
            start_solve = timer()
            out, err, exit_status = run_process(solve_args, os.curdir, DEFAULT_TIMEOUT)
            end_solve = timer()
            total_solve += end_solve - start_solve
            if debug:
                print(out)
            results_solve.append('UNSAT' if 'UNSAT' in out else 'SAT')
            if results_solve[-1] == 'SAT':
                # Found SAT no need to continue running...
                break
    if adv_result == 'SAT':
        assert 'SAT' in results_solve
    else:
        assert 'SAT' not in results_solve
    print("finished solve, time: {} seconds, result: {}".format(total_solve, results_solve))
    if debug:
        print("finished adv, time: {} seconds, result: {}".format(end_adv - start_adv, adv_result))


def run_multiple_compare(target, orig=9):
    eps = ["1e-05", "0.0001", "0.001", "0.01", "0.1"]
    for ep in eps:
        timing_executables("mnist_10_layer.nnet",
                           "500VaryingEpsilon/orig{}_tar{}_ind0_ep{}.txt".format(orig, target, ep)
                           , False)

if __name__ == "__main__":
    # run_multiple_compare(7)
    # run_multiple_compare(1)
    timing_executables("mnist_10_layer.nnet", "500VaryingEpsilon/orig3_tar1_ind0_ep0.0001.txt", True)
    # q, max_idx, out_idx = define_simple_network()
    # assert solve_adversarial(q, max_idx, out_idx)
    # assert not solve_adversarial(q, out_idx[0], out_idx[1:] + [max_idx])

