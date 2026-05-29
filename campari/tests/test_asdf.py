# bash /global/cfs/cdirs/m4385/env/interactive-podman-rknop-dev.sh

# Standard Libary
import numpy as np
import os
import pandas as pd
import warnings

# Common Library
import pathlib
import pytest
import subprocess
import tempfile

# Astronomy Library
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.utils.exceptions import AstropyWarning
from erfa import ErfaWarning

# SNPIT
from campari.tests.test_campari import compare_lightcurves
# from snappl.dbclient import SNPITDBClient
from snappl.diaobject import DiaObject
from snappl.config import Config
from snappl.logger import SNLogger
# from snappl.imagecollection import ImageCollection

warnings.simplefilter("ignore", category=AstropyWarning)
warnings.filterwarnings("ignore", category=ErfaWarning)

cfg = Config.get()

output_dir = cfg.value("photometry.campari_io.output_dir")
debug_dir = cfg.value("photometry.campari_io.debug_dir")

os.environ["IN_ASDF_POD"] = "true"

SNLogger.debug(f"IN_ASDF_POD: {os.getenv('IN_ASDF_POD')}")

in_asdf_pod = os.getenv("IN_ASDF_POD") if os.getenv("IN_ASDF_POD") is not None else False
# ASDF tests will only run if using the rob_dev podman environment.


@pytest.mark.skipif( not in_asdf_pod, reason="IN_ASDF_POD is not set" )
def test_asdf(overwrite_meta):

    provenance_tag = "asdf_functional_test"
    diaobj_process = "load_objects_for_49"
    image_process = "load_rdm_image"

    diaobj = DiaObject.find_objects(name="182.8445_+32.2046", #collection="snpitdb",
                                    provenance_tag=provenance_tag, process=diaobj_process)[0]

    np.testing.assert_equal(diaobj.name, "182.8445_+32.2046")
    np.testing.assert_equal(str(diaobj.id), "9b71851c-5de7-4ff1-a8cc-a08443040f46")

    curfile = pathlib.Path(output_dir) / "182.8445_+32.2046_F062_gaussian_lc.ecsv"
    curfile.unlink(missing_ok=True)
    # Make sure the output file we're going to write doesn't exist so
    # we know we're really running this test!
    assert not curfile.exists()

    imsize = 19
    base_cmd = [
            "python", "../RomanASP.py",
            "--diaobject-name", diaobj.name,
            "-f", "F062",
            "--photometry-campari-psf-transient_class", "gaussian",
            "--photometry-campari-psf-galaxy_class", "gaussian", # this can't stay, this needs to be
            # updated to whatever PSF I am supposed to use for ASDF images.
            "--photometry-campari-use_real_images",
            "--no-photometry-campari-fetch_SED",
            "--photometry-campari-grid_options-type", "regular",
            "--photometry-campari-grid_options-spacing", "1",
            "--photometry-campari-grid_options-subsize", "3", # Change this back when off im fixed
            "--photometry-campari-cutout_size", str(imsize),
            "--photometry-campari-weighting",
            "--photometry-campari-subtract_background", "calculate",
            "--image-collection", "snpitdb",
            "--no-save-to-db",
            "--image-process", image_process,
            "--diaobject-process", diaobj_process,
            "--image-provenance-tag", provenance_tag,
            "--diaobject-provenance-tag", provenance_tag,
            "--nprocs", "10",

        ]

    import subprocess
    result = subprocess.run(base_cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    compare_lightcurves(curfile, pathlib.Path(__file__).parent / "testdata/asdf_regression_lightcurve.ecsv",
                        overwrite_meta=overwrite_meta)
