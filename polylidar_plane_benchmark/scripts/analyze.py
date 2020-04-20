from typing import List
from pathlib import Path

import click
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


from polylidar_plane_benchmark import TRAIN_RESULTS_DIR

sns.set()

@click.group()
def analyze():
    """Analyze Data"""
    pass


def create_dataframe_from_file_list(all_files: List[Path]):
    df: pd.DataFrame = pd.read_csv(all_files[0], delimiter=',')
    for fpath in all_files[1:]:
        df_temp = pd.read_csv(fpath, delimiter=',')
        df = df.append(df_temp)
    df = df.reset_index()
    return df


def get_df_by_variance(all_files: List[Path]):
    df = create_dataframe_from_file_list(all_files)
    columns = ['fname', 'f_corr_seg', 'kernel_size', 'loops_bilateral',
               'loops_laplacian', 'sigma_angle', 'min_triangles', 'norm_thresh_min']
    all_dfs = []
    df_new = df[(df.tcomp == 0.8)]
    for i in range(1, 5):
        df_var1 = df_new[(df_new.variance == i)]
        df_reduced = df_var1[columns]
        df_mean = df_reduced.groupby(columns[2:]).mean()
        # import ipdb; ipdb.set_trace()
        df_mean = df_mean.reset_index()
        # print(df_mean)
        all_dfs.append(df_mean)
    return all_dfs

# Rows - loops_bilateral
# Cols - min_total_weight
# X Axis - loops_laplacian
# Y Axis - f_corr_seg
# Z Axis - BLANK
# Hue - sigma_angle
# Style - kernel_size
# Size - N/A
# missing norm_thresh_min
# missing min_total_weight

# Notes, norm_thresh_min = 0.95 is always better on mean


def plot_graph(df, title='', directory=Path('/tmp')):
    g = sns.relplot(data=df, x='loops_laplacian', y='f_corr_seg', row='loops_bilateral', col='min_triangles',
                    hue='sigma_angle', style='kernel_size')

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


@analyze.command()
@click.option('-d', '--directory', type=click.Path(exists=True), default=TRAIN_RESULTS_DIR)
def training(directory: Path):
    files: List[Path] = [e for e in directory.iterdir() if e.is_file() and '.csv' in e.suffix]
    all_dfs = get_df_by_variance(files)

    # import ipdb; ipdb.set_trace()
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


