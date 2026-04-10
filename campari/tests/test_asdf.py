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


def test_roman_imsim_images(overwrite_meta):
    imsize = 19
    base_cmd = [
            "python", "../RomanASP.py",
            "--photometry-campari-psf-transient_class", "STPSF",
            "--photometry-campari-psf-galaxy_class", "STPSF", # this can't stay, this needs to be
            # updated to whatever PSF I am supposed to use for ASDF images.
            "--photometry-campari-use_real_images",
            "--no-photometry-campari-fetch_SED",
            "--photometry-campari-grid_options-type", "regular",
            "--photometry-campari-grid_options-spacing", "0.75",
            "--photometry-campari-grid_options-subsize", "4",
            "--photometry-campari-grid_options-error_floor", "0.00001",
            "--photometry-campari-grid_options-gaussian_var", "100000",
            "--photometry-campari-grid_options-cutoff", "2",
            "--photometry-campari-cutout_size", str(imsize),
            "--photometry-campari-weighting",
            "--photometry-campari-subtract_background", "calculate",
            "--image-collection", "manual_rdm",
            "--no-save-to-db",
            "--diaobject-collection", "manual",
            "--nprocs", "25",

        ]


    # Get all of the roman imsim images and put them in a list file
    #isim_path = "/romanimsim_sims/2026-03-24_Nexus"  # Note this needs to be in the rob_dev podman environment
    isim_path = "/romanimsim_sims/Nexus"
    files = sorted(pathlib.Path(isim_path).glob("*.asdf"))
    SNLogger.debug(f"Found {len(files)} ASDF files in {isim_path} for ASDF test.")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp_file:
        for i, file in enumerate(files):
            temp_file.write(str(file) + "\n")
            if i > 3:
                break
        temp_file_path = temp_file.name

    cmd = base_cmd.copy()
    cmd.extend(["--img_list", temp_file_path])

    cmd.extend(["--image-collection-basepath", isim_path])

    # Now we'll get the RA/DECs

    truth_file = f"{isim_path}/TRUTH_HLTDS_CORE_SCA1.SNANA.TEXT"
    #truth_file = f"{isim_path}/TRUTH_HLTDS_CORE_SCA1.LCPLOT.TEXT"
    truth_df = pd.read_csv(truth_file, comment="#", sep="\s+")

    SNLogger.debug(f"Truth df {truth_df.head()}")

    # bands = ["F129"]  # Z Y J  #, "F087 F106", "F129"

    # stars = "/romanimsim_sims/2026-03-24_Nexus/GAIA.csv"
    # #stars = f"{isim_path}/GAIA.csv"
    # stars_df = pd.read_csv(stars, comment ="#", sep=",")


    # import astropy.units as u
    # from astropy.coordinates import SkyCoord
    # from matplotlib import pyplot as plt
    # rick_image_approx_center = SkyCoord(ra=9.42*u.degree, dec=-44*u.degree)
    # star_skycoords = SkyCoord(ra=stars_df.ra.values*u.degree, dec=stars_df.dec.values*u.degree)
    # separations = rick_image_approx_center.separation(star_skycoords)
    # closest_stars = stars_df[separations < 0.075*u.degree]
    # plt.scatter(closest_stars.ra, closest_stars.dec, s=1, color="blue", label="Simulated Stars")
    # plt.xlabel("RA")
    # plt.ylabel("DEC")
    # plt.scatter(truth_df.RA, truth_df.DEC, s=1, color="red", label="Simulated Transients")
    # plt.legend()
    # plt.savefig("stars.png")

    # SNLogger.debug(f"closest stars {closest_stars.head()}")

    # successful = 0
    # for i in range(len(closest_stars)):
    #     for band in bands:
    #         cid = "teststar" + str(i)
    #         ra = closest_stars.ra.values[i]
    #         dec = closest_stars.dec.values[i]
    #         cmd.extend(["--ra", str(ra)])
    #         cmd.extend(["--dec", str(dec)])
    #         cmd.extend(["--transient_start", f"{0}"])
    #         cmd.extend(["-f", band])
    #         cmd.extend(["--diaobject-name", f"{cid}"])
    #         SNLogger.debug(f"Running Campari on CID {cid} and band {band} with RA {ra}, DEC {dec}.")

    #         result = subprocess.run(cmd, capture_output=False, text=True)

    #         try:
    #             if result.returncode != 0:
    #                 raise RuntimeError(
    #                     f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    #                 )
    #         except RuntimeError as e:
    #             SNLogger.error(f"Error processing CID {cid} and band {band}: {e}")
    #             continue

    #         # successful += 1
    #         # if successful >= 3:
    #         #     SNLogger.debug("Successfully processed 3 CIDs, stopping test.")
    #         #     raise ValueError("Successfully processed 3 CIDs, stopping test.")

    # # #CIDs = [20005, 20193, 20206, 20226, 20252] # Only these CIDs have truth information.

    # CIDs = [20252]
    # bands = ["F129"]
    # ras = [9.362747229]
    # decs = [-43.974529427]

    # # # CIDs = [20226]
    # # # bands = ["F106"]

    # # # CIDs = [20252]
    # # # bands = ["F129"]

    CIDs = truth_df.CID.values
    CIDs = [1371]
    bands = ["F129"]
    ras = truth_df.RA.values
    decs = truth_df.DEC.values


    failed_cids = []

    for i, cid in enumerate(CIDs):
        for band in bands:

            #try:
            #ra = ras[i]
            #dec = decs[i]
            ra = truth_df[truth_df.CID == cid].RA.values[0]
            dec = truth_df[truth_df.CID == cid].DEC.values[0]
            pkmjd = truth_df[truth_df.CID == cid].SIM_PEAKMJD.values[0]
            approx_start_date = pkmjd - 20
            cmd.extend(["--ra", str(ra)])
            cmd.extend(["--dec", str(dec)])
            cmd.extend(["--transient_start", str(approx_start_date)])
            cmd.extend(["-f", band])
            cmd.extend(["--diaobject-name", f"{cid}"])
            SNLogger.debug(f"Running Campari on CID {cid} and band {band} with RA {ra}, DEC {dec}, and transient start {approx_start_date}.")


            result = subprocess.run(cmd, capture_output=False, text=True)

            if result.returncode != 0:
                raise RuntimeError(
                    f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
                )
            # except RuntimeError as e:
            #     SNLogger.error(f"Error processing CID {cid} and band {band}: {e}")
            #     failed_cids.append((cid, band))
            #     continue