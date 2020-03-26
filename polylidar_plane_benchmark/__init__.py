import os
import sys
import logging
from pathlib import Path
from os.path import join, abspath, dirname



# Set up Logger
logger = logging.getLogger("PPB")
logger.setLevel(logging.INFO)
# ch = logging.StreamHandler()
# ch.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# ch.setFormatter(formatter)
# logger.addHandler(ch)



THIS_DIR = Path(__file__).parent
MAIN_DIR = THIS_DIR / "../"
TEST_FIXTURES_DIR = (MAIN_DIR / 'data').resolve()
TEST_FIXTURES_DIR

SYNPEB_DIR = TEST_FIXTURES_DIR / "synpeb"
SYNPEB_MESHES_DIR = TEST_FIXTURES_DIR / "synpeb_meshes"
SYNPEB_DIR_TEST = SYNPEB_DIR / "test"
SYNPEB_DIR_TRAIN = SYNPEB_DIR / "train"

SYNPEB_DIR_TEST_GT = SYNPEB_DIR_TEST / "gt"
SYNPEB_DIR_TRAIN_GT = SYNPEB_DIR_TRAIN / "gt"

SYNPEB_DIR_TEST_ALL = [SYNPEB_DIR_TEST / "var{}".format(i) for i in range(1, 5)]
SYNPEB_DIR_TRAIN_ALL = [SYNPEB_DIR_TRAIN / "var{}".format(i) for i in range(1, 5)]

DEFAULT_PPB_FILE = SYNPEB_DIR_TRAIN_ALL[0] / "pc_01.pcd"
DEFAULT_PPB_FILE_SECONDARY = SYNPEB_DIR_TRAIN_ALL[0] / "pc_02.pcd"


SYNPEB_ALL_FNAMES = ["pc_{:02}.pcd".format(i) for i in range(1, 31)]
