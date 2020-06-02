from typing import List
from pathlib import Path

import click
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


from polylidar_plane_benchmark import TRAIN_RESULTS_DIR, SYNPEB_ALL_FNAMES, SYNPEB_DIR_TRAIN_ALL

from polylidar_plane_benchmark.utility.helper import estimate_pc_noise, load_pcd_file

sns.set()


@click.group()
def analyze():
    """Analyze Data"""
    pass


def plot_graph(df, title='', directory=Path('/tmp')):
    g = sns.relplot(data=df, x='loops_laplacian', y='f_corr_seg', row='loops_bilateral',
                    col='min_triangles', style='kernel_size')

    fig = g.fig
    fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(top=0.95, wspace=0.2, hspace=0.2)
    fpath = directory / "images/{}.pdf".format(title)
    fig.savefig(str(fpath), bbox_inches='tight')

    df_sorted = df.sort_values(by=['f_corr_seg'], ascending=False)
    print(df_sorted.head(n=5))

    plt.show()

    best_f_corr_seg = df_sorted.iloc[0, :].values[-1]

    return best_f_corr_seg

def estimate_pc_noise_files(fpaths, stride=1, variance=1, samples=100, sample_size=2,**kwargs):
    noise_records = []
    for fpath in fpaths:
        pc_raw, pc_image = load_pcd_file(fpath, stride)
        noise = estimate_pc_noise(pc_image, samples=samples, sample_size=sample_size)
        fname = Path(fpath).name
        noise_records.append(dict(noise=noise, samples=samples, stride=stride, sample_size=sample_size, variance=variance, fname=fname))
    return noise_records

def create_dataframe_from_file_list(all_files: List[Path]):
    df: pd.DataFrame = pd.read_csv(all_files[0], delimiter=',')
    for fpath in all_files[1:]:
        df_temp = pd.read_csv(fpath, delimiter=',')
        df = df.append(df_temp)
    df = df.reset_index()
    return df


def get_df_by_variance(all_files: List[Path],
                       columns=['fname', 'f_corr_seg', 'kernel_size', 'loops_bilateral',
                                'loops_laplacian', 'sigma_angle', 'min_triangles', 'norm_thresh_min'],
                       groupby=True):
    df = create_dataframe_from_file_list(all_files)

    all_dfs = []
    df_new = df[(df.tcomp == 0.8)]
    for i in range(1, 5):
        df_var1 = df_new[(df_new.variance == i)]
        df_reduced = df_var1[columns]
        if groupby:
            df_mean = df_reduced.groupby(columns[2:]).mean()
            df_mean = df_mean.reset_index()
        else:
            df_mean = df_reduced.reset_index()
        all_dfs.append(df_mean)
    return all_dfs


def get_all_training_fpaths(variance:int=1):
    all_fnames = SYNPEB_ALL_FNAMES
    all_fnames = all_fnames[0:10]
    all_fpaths = []

    base_dir = SYNPEB_DIR_TRAIN_ALL[int(variance) - 1]

    for fname in all_fnames:
        fpath = str(base_dir / fname)
        all_fpaths.append(fpath)
    return all_fpaths


@analyze.command()
def pcd_noise():
    """Visualize noise level estimation in point cloud files"""
    all_fnames = SYNPEB_ALL_FNAMES
    all_fnames = all_fnames[0:10]

    all_records = []
    for variance in range(1, 5):
        base_dir = SYNPEB_DIR_TRAIN_ALL[int(variance) - 1]
        all_fpaths = []
        for fname in all_fnames:
            fpath = str(base_dir / fname)
            all_fpaths.append(fpath)
        for samples in [25, 100, 1000]:
            for sample_size in [2, 3]:
                for stride in [1, 2]:
                    records = estimate_pc_noise_files(all_fpaths, stride=stride,
                                                      variance=variance, samples=samples, sample_size=sample_size)
                    all_records.extend(records)

    df = pd.DataFrame.from_records(all_records)

    g = sns.catplot(x="variance", y="noise", col='samples', row='stride', hue='sample_size', data=df, kind='swarm')
    plt.show()


@analyze.command()
@click.option('-d', '--directory', type=click.Path(exists=True), default=TRAIN_RESULTS_DIR)
def fit_laplacian(directory):
    """ Analyze noise in point clouds and determine a *sufficient* number of laplacian iterations to smooth
    """
    # Create Noise Dataframe
    all_noise_records = []
    for variance in range(1, 5):
        all_fpaths = get_all_training_fpaths(variance)
        pc_noise_records = estimate_pc_noise_files(all_fpaths, stride=2, variance=variance, samples=100, sample_size=2)
        all_noise_records.extend(pc_noise_records)
    df_noise = pd.DataFrame.from_records(all_noise_records)[['fname', 'variance', 'noise']]

    # Create dataframe of all training data, reduce data to whats of interest
    files: List[Path] = [e for e in directory.iterdir() if e.is_file() and '.csv' in e.suffix and not 'test' in e.name]
    all_dfs = create_dataframe_from_file_list(files)
    columns = ['fname', 'variance', 'f_corr_seg', 'kernel_size', 'loops_bilateral',
               'loops_laplacian', 'sigma_angle', 'min_triangles', 'norm_thresh_min']
    all_dfs = all_dfs[columns]
    bp = dict(loops_bilateral=1, kernel_size=3, sigma_angle=0.1, min_triangles=250, norm_thresh_min=0.95)
    df = all_dfs[(all_dfs.loops_bilateral == bp['loops_bilateral']) &
                 (all_dfs.kernel_size == bp['kernel_size']) &
                 (all_dfs.sigma_angle == bp['sigma_angle']) &
                 (all_dfs.min_triangles == bp['min_triangles']) &
                 (all_dfs.norm_thresh_min == bp['norm_thresh_min'])]
    df = df.reset_index()

    # merge the dataframe, now we know the predicted point cloud noise value with all the training data
    combined_df = pd.merge(df, df_noise,  how='left', left_on=['fname','variance'], right_on = ['fname','variance'])
    combined_df = combined_df.sort_values(by=['variance', 'fname']) 

    # plot the data, see what would be good partition points
    g = sns.relplot(x="noise", y="f_corr_seg", hue="variance", col='loops_laplacian', data=combined_df)
    plt.show()

    # Simple split
    print("Splitting at .0002, .0004, .0006, and greater to 2, 4, 6, and 8 Laplacian iterations")
    splits = np.array([.0002, .0003, .00045])
    loops_laplacian = [2, 4, 6, 8]
    correct_records = []
    for index, row in combined_df.iterrows():
        idx = np.searchsorted(splits, row.noise)
        predicted_loops = loops_laplacian[idx]
        data_loops = row.loops_laplacian
        # check if this is the row we want to keep
        if predicted_loops == data_loops:
            correct_records.append(row)

    # Show results of predcition
    df_predicted = pd.DataFrame(correct_records)
    print(df_predicted.mean())

    g = sns.relplot(x="noise", y="f_corr_seg", hue="variance", col='loops_laplacian', data=df_predicted)
    plt.show()


@analyze.command()
@click.option('-d', '--directory', type=click.Path(exists=True), default=TRAIN_RESULTS_DIR)
def test(directory: Path):
    """ Show results of Polylidar on dataset """
    files: List[Path] = [e for e in directory.iterdir() if e.is_file() and '.csv' in e.suffix and 'test' in e.name]
    df = create_dataframe_from_file_list(files)
    columns_metrics = ['n_gt', 'n_ms_all', 'f_weighted_corr_seg', 'rmse',
                       'f_corr_seg', 'n_corr_seg', 'n_over_seg', 'n_under_seg', 'n_missed_seg',
                       'n_noise_seg', 'laplacian', 'bilateral', 'mesh', 'fastga_total', 'polylidar']
    columns = ['variance', 'fname'] + columns_metrics
    df = df[columns].reset_index()
    df_mean = df.groupby('variance').mean()
    print(df_mean)
    print(df[columns_metrics].mean())
    # TODO print out latex table?




@analyze.command()
@click.option('-d', '--directory', type=click.Path(exists=True), default=TRAIN_RESULTS_DIR)
def training(directory: Path):
    """ Show results of hyperparameters for polylidar """
    files: List[Path] = [e for e in directory.iterdir() if e.is_file() and '.csv' in e.suffix and not 'test' in e.name]
    all_dfs = get_df_by_variance(files)

    all_best_values = []
    all_best_values.append(plot_graph(all_dfs[0], title='Variance = 1', directory=directory))
    all_best_values.append(plot_graph(all_dfs[1], title='Variance = 2', directory=directory))
    all_best_values.append(plot_graph(all_dfs[2], title='Variance = 3', directory=directory))
    all_best_values.append(plot_graph(all_dfs[3], title='Variance = 4', directory=directory))

    mean_value = np.mean(all_best_values)
    print("Mean of all variances of best hyperparameters: {:.2f}".format(mean_value))

# Stride = 2
# Var 1 - norm_thresh_min=0.95, threshold_abs=2, min_total_weight=0.1, loops_laplacian=2, loops_bilateral=1, sigma_angle=0.10, kernel_size=3
# Var 2 - norm_thresh_min=0.95, threshold_abs=2, min_total_weight=0.1, loops_laplacian=4, loops_bilateral=1, sigma_angle=0.10, kernel_size=3
# Var 3 - norm_thresh_min=0.95, threshold_abs=2, min_total_weight=0.1, loops_laplacian=4, loops_bilateral=1, sigma_angle=0.10, kernel_size=3
# Var 4 - norm_thresh_min=0.95, threshold_abs=2, min_total_weight=0.1, loops_laplacian=4, loops_bilateral=1, sigma_angle=0.10, kernel_size=5
# Mean of all variances of best hyperparameters: 0.44


# Stride = 1
#      kernel_size  loops_bilateral  loops_laplacian  sigma_angle  min_triangles  norm_thresh_min  f_corr_seg
# 52             3                2                6          0.1           1000             0.95    0.567556
# 14             3                0                6          0.2           1000             0.95    0.566386
# 12             3                0                6          0.1           1000             0.95    0.566386
# 164            5                2                2          0.1           1000             0.95    0.563290
# 68             3                4                4          0.1           1000             0.95    0.560023
#      kernel_size  loops_bilateral  loops_laplacian  sigma_angle  min_triangles  norm_thresh_min  f_corr_seg
# 148            5                1                4          0.1           1000             0.95    0.548350
# 128            5                0                4          0.1           1000             0.95    0.543558
# 130            5                0                4          0.2           1000             0.95    0.543558
# 150            5                1                4          0.2           1000             0.95    0.540325
# 168            5                2                4          0.1           1000             0.95    0.523590
#      kernel_size  loops_bilateral  loops_laplacian  sigma_angle  min_triangles  norm_thresh_min  f_corr_seg
# 172            5                2                6          0.1           1000             0.95    0.504128
# 152            5                1                6          0.1           1000             0.95    0.500836
# 174            5                2                6          0.2           1000             0.95    0.499780
# 138            5                0                8          0.2           1000             0.95    0.493954
# 154            5                1                6          0.2           1000             0.95    0.493563
#      kernel_size  loops_bilateral  loops_laplacian  sigma_angle  min_triangles  norm_thresh_min  f_corr_seg
# 156            5                1                8          0.1           1000             0.95    0.452940
# 176            5                2                8          0.1           1000             0.95    0.447471
# 192            5                4                6          0.1           1000             0.95    0.446807
# 194            5                4                6          0.2           1000             0.95    0.442676
# 158            5                1                8          0.2           1000             0.95    0.439245
# Mean of all variances of best hyperparameters: 0.52
# I just do 2,4,6,8 laplacian loops based upon variance I get 0.51. This is what I chose to do first
# afterward a fit a model of point cloud noise to amount of smoothing needed


# Test Results, stride = 1
#           index       n_gt   n_ms_all  f_weighted_corr_seg  f_corr_seg  n_corr_seg  n_over_seg  n_under_seg  n_missed_seg  n_noise_seg  laplacian  bilateral      mesh  fastga_total  polylidar
# variance                                                                                                                                                                                       
# 1          74.5  42.433333  24.700000             0.798318    0.528498   20.200000    0.166667     0.433333     21.133333     3.733333   1.101684   3.274795  9.003584      6.450586  13.558273
# 2          44.5  42.600000  24.200000             0.732586    0.453687   18.333333    0.166667     0.366667     23.300000     5.166667   1.102073   3.319611  9.156282      6.606149  14.205313
# 3         104.5  42.333333  25.500000             0.757537    0.457345   17.733333    0.266667     0.333333     23.600000     6.833333   1.223209   3.251596  9.320403      6.555388  13.788202
# 4          14.5  42.600000  26.866667             0.751720    0.442893   17.033333    0.466667     0.466667     24.033333     8.433333   1.435756   3.276963  8.796821      6.503823  13.206990
# n_gt                   42.491667
# n_ms_all               25.316667
# f_weighted_corr_seg     0.760040
# f_corr_seg              0.470606
# n_corr_seg             18.325000
# n_over_seg              0.266667
# n_under_seg             0.400000
# n_missed_seg           23.016667
# n_noise_seg             6.041667
# laplacian               1.215680
# bilateral               3.280741
# mesh                    9.069273
# fastga_total            6.528987
# polylidar              13.689695
# dtype: float64

# Test Results, stride = 2
#           index       n_gt   n_ms_all  f_weighted_corr_seg  f_corr_seg  n_corr_seg  n_over_seg  n_under_seg  n_missed_seg  n_noise_seg  laplacian  bilateral      mesh  fastga_total  polylidar
# variance                                                                                                                                                                                       
# 1          74.5  42.433333  23.833333             0.749470    0.420263   15.233333    0.833333     0.333333     25.633333     6.100000   0.519065   0.754139  1.574141      2.469488   3.474274
# 2          44.5  42.600000  24.800000             0.732004    0.419430   15.500000    0.733333     0.333333     25.700000     7.333333   0.519379   0.712587  0.936739      2.514377   3.812655
# 3         104.5  42.333333  26.566667             0.723124    0.400742   15.000000    0.866667     0.366667     25.700000     8.600000   0.483545   0.710547  0.891122      2.558723   3.858203
# 4          14.5  42.600000  30.166667             0.593601    0.333484   13.133333    1.333333     0.466667     27.100000    13.100000   0.481643   0.711259  1.091690      2.846986   4.247679
# n_gt                   42.491667
# n_ms_all               26.341667
# f_weighted_corr_seg     0.699550
# f_corr_seg              0.393480
# n_corr_seg             14.716667
# n_over_seg              0.941667
# n_under_seg             0.375000
# n_missed_seg           26.033333
# n_noise_seg             8.783333
# laplacian               0.500908
# bilateral               0.722133
# mesh                    1.123423
# fastga_total            2.597393
# polylidar               3.848203
# dtype: float64



