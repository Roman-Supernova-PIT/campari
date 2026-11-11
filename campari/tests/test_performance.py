# Standard Library
import os
import pathlib
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


@pytest.mark.run_last
def test_memory(cfg, capsys):
    SNLogger.debug("Running test_memory")
    SNLogger.debug(os.getenv("GITHUB_ACTIONS"))
    if os.getenv("GITHUB_ACTIONS") == "true":
        print("Running inside GitHub Actions!")
        location = "github_actions"
    else:
        if "perlmutter" not in os.getenv("SNPIT_CONFIG"):
            raise ValueError("This test is only meant to be run on NERSC, or on GitHub Actions.")
        location = "NERSC"

    time_start = time.time()

    nprocs = 20
    # Use an explicit list of the 8 images stored in photometry_test_data, so this test can run on GitHub Actions
    # without needing the full OU2024 image set.
    img_list = pathlib.Path(__file__).parent / "testdata/test_imagelists/test_memory_image_list.csv"
    output = os.system(
        f"python ../pipeline.py --diaobject-name 20172782 -f Y106 -i {img_list} "
        "--photometry-campari-psf-galaxy_class ou24PSF "
        "--no-photometry-campari-fetch_SED "
        "--photometry-campari-grid_options-type regular "
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

    debug_dir = cfg.value("photometry.campari_io.debug_dir")
    mem_df = pd.read_csv(f"{debug_dir}/test_regression.csv")
    peak_memory = mem_df["memory_gb"].max()

    # Disable output capture so these numbers appear in the log even when the test passes.
    with capsys.disabled():
        print(f"\ntest_memory on {location}: run time {total_time:.1f} s, peak memory {peak_memory:.3f} GB")

    time_thresh = {
        "NERSC": 150,
        "github_actions": 150
    }

    assert total_time < time_thresh[location], (
        f"The test run on a SN took longer than {time_thresh[location]} seconds, typical time is ~ 110 seconds."
    )

    # Typical time on a NERSC interactive (NOT LOGIN) node is 115 +/- 5 seconds for 20 procs.
    # Before making the memory improvements, the time was closer to 100 seconds, but the memory usage
    # was out of control. This is a tradeoff between memory and speed, and I think a 10% speed reduction is
    # worth worth it for a factor of >6 memory reduction.

    memory_thresh = {
        "NERSC": 1.5,
        "github_actions": 1.5
    }

    try:
        np.testing.assert_array_less(mem_df["memory_gb"].values, memory_thresh[location]),
        f"Memory usage exceeded the threshold for {location}, which is {memory_thresh[location]} GB"
    except AssertionError:
        plt.plot(mem_df["elapsed_seconds"].values, mem_df["memory_gb"].values)
        plot_path = f"{debug_dir}/memory_usage_plot.png"
        plt.savefig(plot_path)
        raise ValueError(f"Memory usage exceeded {memory_thresh[location]} GB. "
        f"See plot at {plot_path} for details.")
