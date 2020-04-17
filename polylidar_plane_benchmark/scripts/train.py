from multiprocessing import Pool, Value, Process
import signal
import time
import sys, traceback

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
        # Fast GA Peak Identification
        threshold_abs=[2, 4],
        min_total_weight=[0.01, 0.02],

        # Polylidar
        norm_thresh_min=[0.95, .96],
    )
    mesh_parameters = list(ParameterGrid(mesh_param_grid))
    else_parameters = list(ParameterGrid(all_else_grid))

    all_parameters = [{**l[0], **l[1]} for l in itertools.product(mesh_parameters, else_parameters)]

    # param_grid = dict(
    #     ## Mesh Smoothing
    #     loops_laplacian=[1, 2, 4, 6, 8],
    #     # laplacian_lambda=[1.0],
    #     kernel_size=[3,5],
    #     loops_bilateral=[1, 2, 4, 6, 8],
    #     sigma_angle=[0.10, 0.20],

    #     ## Fast GA Peak Identification
    #     threshold_abs=[2, 4],
    #     min_total_weight=[0.01, 0.02],

    #     ## Polylidar
    #     # lmax=[0.1],
    #     # alpha=[0.0],
    #     # min_triangles=[200],
    #     # z_thresh=[0.1],
    #     # min_hole_vertices=[50],
    #     # task_threads=[4],
    #     norm_thresh_min=[0.95, .96],

    #     ## Evaluate Args
    #     # tcomp=[0.7, 0.8],
    #     # variance=[1, 2, 3, 4]
    # )
    # all_parameters = list(ParameterGrid(param_grid))

    # print(all_parameters)
    # print(len(all_parameters))
    return all_parameters


def show_prog(counter:Value, total_iterations):
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

    counter = Value('i', 0)

    # params = params[:10]
    num_variances = 4
    num_files = 10

    total_iterations = len(params) * num_files * num_variances

    iterable_params = []
    for variance in range(1, 5):
        iterable_params.append((params, variance, counter))
    
    try:
        procs = [Process(target=evaluate_with_params, args=(params, i, counter)) for i in range(1, 5)]
        progress = Process(target=show_prog, args=(counter, total_iterations))
        progress.start()
        for p in procs: p.start()
        for p in procs: p.join()
        progress.join()
    except KeyboardInterrupt:
        print("Received Keyboard interrupt! Stopping all processes!")
        for p in procs: p.kill()
        for p in procs: p.join()
        progress.kill()
        progress.join()
    except Exception as e:
        print("Received Exception in Main Thread! Stopping all processes!")
        for p in procs: p.kill()
        for p in procs: p.join()
        progress.kill()
        progress.join()
        traceback.print_exc(file=sys.stdout)



if __name__ == "__main__":
    main()
