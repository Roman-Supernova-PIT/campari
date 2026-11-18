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
@pytest.mark.parametrize("psftype", ["galsim", "gaussian"])
def test_memory(cfg, capsys, psftype):

    if psftype == "gaussian":
        galaxy_psf_class = "gaussian"
        transient_psf_class = "gaussian"
    elif psftype == "galsim":
        galaxy_psf_class = "ou24PSF"
        transient_psf_class = "ou24PSF_slow"

    SNLogger.debug("Running test_memory")
    SNLogger.debug(os.getenv("GITHUB_ACTIONS"))
    if os.getenv("GITHUB_ACTIONS") == "true":
        print("Running inside GitHub Actions!")
        location = "github_actions"
    else:
        if "perlmutter" not in os.getenv("SNPIT_CONFIG"):
            raise ValueError("This test is only meant to be run on NERSC, or on GitHub Actions.")
        location = "NERSC"

    # Remove any lightcurve left over from an earlier run (e.g. test_regression writes the same file).
    # Older snappl versions (< 0.48.0) ignore overwrite=True, so an existing file would make this test fail.

    psf = "ou24psf_slow" if psftype == "galsim" else "gaussian"

    curfile = pathlib.Path(output_dir) / f"20172782_Y106_{psf}_lc.ecsv"
    curfile.unlink(missing_ok=True)
    assert not curfile.exists()

    time_start = time.time()

    nprocs = 1
    img_list = pathlib.Path(__file__).parent / "testdata/test_imagelists/test_memory_image_list.csv"
    output = os.system(
        f"python ../pipeline.py --diaobject-name 20172782 -f Y106 -i {img_list} "
        f"--photometry-campari-psf-galaxy_class {galaxy_psf_class} "
        "--no-photometry-campari-fetch_SED "
        "--photometry-campari-grid_options-type regular "
        "--photometry-campari-cutout_size 19 "
        "--photometry-campari-weighting "
        "--photometry-campari-subtract_background_method SKY_MEAN "
        f"--photometry-campari-psf-transient_class {transient_psf_class} "
        "--save_model --image-collection ou2024 "
        " --no-save-to-db"
        " --diaobject-collection ou2024"
        f" --nprocs {nprocs}"
        " --photometry-campari-grid_options-gaussian_var 1000 "
        "--photometry-campari-print_memory_usage "
        f"--photometry-campari-save_memory_file_name test_regression_{psftype}"
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

    time_thresh = {}
    time_thresh["gaussian"] = {"NERSC": 300, "github_actions": 300}
    time_thresh["galsim"] = {"NERSC": 250, "github_actions": 250}

    time_comp = time_thresh[psftype][location]

    assert total_time < time_comp, (
        f"The test run on a SN took {total_time:.1f} seconds, typical time is ~ {time_comp} seconds."
    )

    memory_thresh = {}
    memory_thresh["gaussian"] = {"NERSC": 0.6, "github_actions": 0.6}
    memory_thresh["galsim"] = {"NERSC": 1.3, "github_actions": 1.3}

    mem_compare = memory_thresh[psftype][location]

    try:
        np.testing.assert_array_less(mem_df["memory_gb"].values, mem_compare),
        f"Memory usage exceeded the threshold for {location}, which is {mem_compare} GB"
    except AssertionError:
        plt.plot(mem_df["elapsed_seconds"].values, mem_df["memory_gb"].values)
        plot_path = f"{debug_dir}/memory_usage_plot.png"
        plt.savefig(plot_path)
        raise ValueError(f"Memory usage exceeded {mem_compare} GB. "
        f"See plot at {plot_path} for details.")
