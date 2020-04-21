""" This script is used to tune hyperparameters used in Polylidar on SynPeb TRAINING dataset
Note - Most of the hyperparameters that are included/important are actually apart of OrganizedPointFilters (OPF)
OPF is a library I made for fast mesh smoothing form organized point clouds

In total there are three libraries used for the SynPEB Benchmark
1. Organized Point Filters (OPF) - Smoothing Mesh
2. Fast Gaussian Accumulator (FastGA) - Fast Plane Normal Detection
3. Polylidar - Fast Plane and Polygon Extraction

Note - Polylidar generates *planes* and *polygon*
    Planes are a sets of spatially connected triangles of similar normals
    Polygons are the concave hull and possible interior holes of said planes

SynPEB (and other benchmarks) are **point** based for planes, not polygon based.
For this reason we use the planes returned from polylidar for benchmark evaluation.
This cooresponds to the point indices of the triangles in each plane 


Question - Why did you make this a python script and not use `Click` for the CLI as you did for `visualize`
Answer - I was paranoid about python `imports` (especially cuda) during process `fork` for multiprocessing
         This script is self isolated and **nothing** is imported until *inside* the launching subprocess

"""
from multiprocessing import Pool, Value, Process
import signal
import time
import sys
import traceback

from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
import itertools
from polylidar_plane_benchmark.scripts.train_core import evaluate_with_params


def get_permutations():
    mesh_param_grid = dict(
        loops_laplacian=[1, 2, 4, 6, 8],
        kernel_size=[3, 5],
        loops_bilateral=[0, 1, 2, 4, 6, 8],
        sigma_angle=[0.10, 0.20],
    )
    all_else_grid = dict(
        # Polylidar
        norm_thresh_min=[0.95],
        min_triangles=[1000, 2000],
        # Downsampling
        stride=[1]
    )
    mesh_parameters = list(ParameterGrid(mesh_param_grid))
    else_parameters = list(ParameterGrid(all_else_grid))

    all_parameters = [{**l[0], **l[1]} for l in itertools.product(mesh_parameters, else_parameters)]

    return all_parameters


def show_prog(counter: Value, total_iterations):
    prog = tqdm(total=total_iterations, desc="Total")
    while 1:
        try:
            count = counter.value
            prog.n = count
            prog.update(0)
            if prog.n >= total_iterations:
                break
            time.sleep(.1)
        except:
            continue


def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def main():
    params = get_permutations()
    # params = params[:100]

    counter = Value('i', 0)

    param_split = 2 # Split the parameters to increase parallelism
    num_variances = 4 # Separate work by variance to increase parallelism
    num_files = 10
    total_iterations = len(params) * num_files * num_variances

    step_size = int(len(params) / param_split)
    params_subset = [params[i:i + step_size] for i in range(0, len(params), step_size)]
    assert len(params_subset) == param_split

    # Will contain the set of function arguments to pass to each process (parallel)
    function_args = []
    for i in range(num_variances):
        variance = i + 1
        for param_index, param_subset in enumerate(params_subset):
            assert len(param_subset) == int(len(params) / param_split), "Param subset should be an even split"
            function_args.append((param_subset, param_index, variance, counter))

    total_processes = num_variances * param_split
    try:
        procs = [Process(target=evaluate_with_params, args=function_args[i]) for i in range(total_processes)]
        progress = Process(target=show_prog, args=(counter, total_iterations))
        progress.start()
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        print("Finished All Processes")
        for i, p in enumerate(procs):
            print("Process {} returned {}".format(i, p.exitcode))
        progress.join()
    except KeyboardInterrupt:
        print("Received Keyboard interrupt! Stopping all processes!")
        for p in procs:
            p.kill()
        for p in procs:
            p.join()
        progress.kill()
        progress.join()
    except Exception as e:
        print("Received Exception in Main Thread! Stopping all processes!")
        for p in procs:
            p.kill()
        for p in procs:
            p.join()
        progress.kill()
        progress.join()
        traceback.print_exc(file=sys.stdout)


if __name__ == "__main__":
    main()
