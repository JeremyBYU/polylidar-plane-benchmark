import random
import logging
import pickle
from functools import partial
import json
from os import path
from pathlib import Path
import multiprocessing as mp

import click
# Monkey patch Click to show default values for all commands
orig_init = click.core.Option.__init__


def new_init(self, *args, **kwargs):
    orig_init(self, *args, **kwargs)
    self.show_default = True


click.core.Option.__init__ = new_init


import numpy as np
from polylidar_plane_benchmark.scripts.visualize import visualize
from polylidar_plane_benchmark.scripts.analyze import analyze


@click.group()
def cli():
    """Generates data and run benchmarks for concave algorithms"""
    pass



cli.add_command(visualize)
cli.add_command(analyze)

