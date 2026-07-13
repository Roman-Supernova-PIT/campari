# Standard Libary
import os
import time
import pytest

# Common Library
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from snappl.config import Config
from snappl.logger import SNLogger


cfg = Config.get()
output_dir = cfg.value("photometry.campari_io.output_dir")
debug_dir = cfg.value("photometry.campari_io.debug_dir")


@pytest.fixture(scope="module")
def campari_test_data(cfg):
    return cfg.value("photometry.campari_io.test_data")


def test_memory(cfg):
    time_start = time.time()

    nprocs = 20
    output = os.system(
        f"python ../RomanASP.py --diaobject-name 20172782 -f Y106 -t 10 -n 10 "
        "--photometry-campari-psf-galaxy_class ou24PSF "
        "--no-photometry-campari-fetch_SED "
        "--photometry-campari-grid_options-type contour "
        "--photometry-campari-cutout_size 19 "
        "--photometry-campari-weighting "
        "--photometry-campari-subtract_background_method SKY_MEAN "
        "--photometry-campari-psf-transient_class ou24PSF_slow "
        "--save_model --image-collection ou2024 "
        " --no-save-to-db"
        " --diaobject-collection ou2024"
        f" --nprocs {nprocs}"
        " --photometry-campari-grid_options-gaussian_var 1000 "
        "--photometry-campari-print_memory_usage "
        "--photometry-campari-save_memory_file_name test_regression"
    )
    assert output == 0, "The test run on a SN failed. Check the logs"

    time_end = time.time()
    total_time = time_end - time_start

    np.testing.assert_array_less(total_time, 150), "The test run on a SN took longer than 150 seconds," + \
                                                   " typical time is 110 seconds."

    SNLogger.debug(f"Test run on a SN took {time_end - time_start} seconds")

    # Typical time on a NERSC interactive (NOT LOGIN) node is 115 +/- 5 seconds for 20 procs.
    # Before making the memory improvements, the time was closer to 100 seconds, but the memory usage
    # was out of control. This is a tradeoff between memory and speed, and I think a 10% speed reduction is
    # worth worth it for a factor of >6 memory reduction.

    debug_dir = cfg.value("photometry.campari_io.debug_dir")
    mem_df = pd.read_csv(f"{debug_dir}/test_regression.csv")

    SNLogger.debug(f"The peak memory usage was {mem_df['memory_gb'].max()} GB")

    try:
        np.testing.assert_array_less(mem_df["memory_gb"].values, 1.5), "Memory usage exceeded 1.5 GB"
    except AssertionError:
        plt.plot(mem_df["elapsed_seconds"].values, mem_df["memory_gb"].values)
        plot_path = f"{debug_dir}/memory_usage_plot.png"
        plt.savefig(plot_path)
        raise ValueError(f"Memory usage exceeded 1.5 GB. See plot at {plot_path} for details.")
