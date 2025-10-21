# Standard Libary
import pathlib
import tempfile

# Common Library
import numpy as np
import pandas as pd
import pytest

# Astronomy Library
from astropy.table import Table
import astropy.units as u

# SNPIT
from campari.access_truth import (
    add_truth_to_lc,
    extract_id_using_ra_dec,
    extract_object_from_healpix,
    extract_sn_from_parquet_file_and_write_to_csv,
    extract_star_from_parquet_file_and_write_to_csv,
    find_parquet,
)
from campari.io import (
    build_lightcurve,
    open_parquet,
)

from campari.run_one_object import campari_lightcurve_model
from snappl.diaobject import DiaObject
from snappl.image import FITSImageStdHeaders
from snappl.imagecollection import ImageCollection
from snpit_utils.config import Config
from snpit_utils.logger import SNLogger


@pytest.fixture(scope="module")
def sn_path(cfg):
    return cfg.value("ou24.sn_truth_dir")


def test_find_parquet(sn_path):
    parq_file_ID = find_parquet(50134575, sn_path)
    assert parq_file_ID == 10430


def test_extract_sn_from_parquet_file_and_write_to_csv(sn_path):
    cfg = Config.get()
    new_snid_file = (
        pathlib.Path(cfg.value("photometry.campari.paths.debug_dir"))
        / "test_extract_sn_from_parquet_file_and_write_to_csv_snids.csv"
    )
    new_snid_file.unlink(missing_ok=True)
    # Make sure we're really writing a new file so that this
    #   test is really meaningful
    assert not new_snid_file.exists()

    # TODO don't write to testdata
    extract_sn_from_parquet_file_and_write_to_csv(10430, sn_path, new_snid_file, mag_limits=[20, 21])
    sn_ids = pd.read_csv(new_snid_file, header=None).values.flatten()
    test_sn_ids = pd.read_csv(pathlib.Path(__file__).parent / "testdata/test_snids.csv", header=None).values.flatten()
    np.testing.assert_array_equal(sn_ids, test_sn_ids), "The SNIDs do not match the test example"


def test_extract_star_from_parquet_file_and_write_to_csv(sn_path):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as temp_file:
        output_path = temp_file.name
        extract_star_from_parquet_file_and_write_to_csv(10430, sn_path, output_path, ra=7.1, dec=-44.1, radius=0.25)
        star_ids = pd.read_csv(output_path, header=None).values.flatten()
        test_star_ids = pd.read_csv(
            pathlib.Path(__file__).parent / "testdata/test_star_ids.csv", header=None
        ).values.flatten()
    np.testing.assert_array_equal(star_ids, test_star_ids), "The star IDs do not match the test example"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as temp_file:
        output_path = temp_file.name
        extract_star_from_parquet_file_and_write_to_csv(10430, sn_path, output_path)
        star_ids = pd.read_csv(output_path, header=None).values.flatten()
        parq = open_parquet(10430, sn_path, obj_type="star")
        assert len(star_ids) == parq["id"].size, (
            "extract_star_from_parquet_file_and_write_to_csv did not return" + "all stars when no radius was passed"
        )


def test_build_lc_and_add_truth(sn_path):
    exposures = pd.DataFrame(
        {
            "pointing": [5934, 35198],
            "sca": [3, 2],
            "date": [62000.40235, 62495.605],
            "detected": [False, True],
            "filter": ["Y106", "Y106"],
            "x": [2044, 2044],
            "y": [2044, 2044],
            "x_cutout": [5, 5],
            "y_cutout": [5, 5],
        }
    )

    explist = Table.from_pandas(exposures)
    explist.sort(["detected", "sca"])

    # Getting a WCS to use
    pointing = 5934
    sca = 3
    band = "Y106"
    img_collection = ImageCollection()
    img_collection = img_collection.get_collection("ou2024")
    snappl_image = img_collection.get_image(pointing=pointing, sca=sca, band=band)

    wcs = snappl_image.get_wcs()

    image_list = []
    cutout_image_list = []

    for i in range(len(explist["date"])):
        img = FITSImageStdHeaders(
            header=None,
            data=np.zeros((4085, 4085)),
            noise=np.zeros((4085, 4085)),
            flags=np.zeros((4085, 4085)),
            path="/dev/null",
        )
        img.mjd = explist["date"][i]
        img.filter = explist["filter"][i]
        img.pointing = explist["pointing"][i]
        img.sca = explist["sca"][i]
        img._wcs = wcs
        img.band = "Y106"
        image_list.append(img)
        cutout_image_list.append(img)

    lc_model = campari_lightcurve_model(
        flux=100, sigma_flux=10, image_list=image_list, cutout_image_list=cutout_image_list,
        LSB=25.0, pre_transient_images=1, post_transient_images=0
    )

    diaobj = DiaObject.find_objects(id=20172782, ra=7, dec=-41, collection="manual")[0]
    diaobj.mjd_start = 62001.0
    diaobj.mjd_end = np.inf

    # The data values are arbitary, just to check that the lc is constructed properly.
    lc = build_lightcurve(diaobj, lc_model)



    saved_lc = Table.read(pathlib.Path(__file__).parent / "testdata/saved_lc_file.ecsv", format="ascii.ecsv")

    for i in lc.columns:
        SNLogger.debug(f"Checking column {i}, lc: {lc[i].value}, saved_lc: {saved_lc[i]}")
        if not isinstance(saved_lc[i][0], str):

            np.testing.assert_allclose(lc[i].value, saved_lc[i])
        else:
            np.testing.assert_array_equal(lc[i].value, saved_lc[i])
    for key in list(lc.meta.keys()):
        SNLogger.debug(f"Checking column {key}, lc: {lc.meta[key]}")
        SNLogger.debug(f"saved_lc: {saved_lc.meta[key]}")
        if not isinstance(saved_lc.meta[key], str):
            np.testing.assert_allclose(lc.meta[key], saved_lc.meta[key])
        else:
            np.testing.assert_array_equal(lc.meta[key], saved_lc.meta[key])

    # Now add the truth to the lightcurve
    # NOTE: The truth_path thing is a hacky fix, but since I have another issue raised to remove this from
    # campari entirely, I'm leaving it for now. It will be gone soon anyway.
    lc = add_truth_to_lc(lc, sn_path, "SN")
    saved_lc = Table.read(pathlib.Path(__file__).parent / "testdata/saved_lc_file_with_truth.ecsv", format="ascii.ecsv")

    for i in lc.columns:
        if not isinstance(saved_lc[i][0], str):
            np.testing.assert_allclose(lc[i].value, saved_lc[i])
        else:
            np.testing.assert_array_equal(lc[i].value, saved_lc[i])
    for key in list(lc.meta.keys()):
        if not isinstance(saved_lc.meta[key], str):
            np.testing.assert_allclose(lc.meta[key], saved_lc.meta[key])
        else:
            np.testing.assert_array_equal(lc.meta[key], saved_lc.meta[key])


def test_extract_id_using_ra_dec(sn_path):
    ra = 7.3447740
    dec = -44.919229
    ID, dist = extract_id_using_ra_dec(sn_path, ra, dec, radius=5 * u.arcsec, object_type="SN")
    np.testing.assert_equal(ID, 40120913), "The ID extracted from the RA/Dec does not match the expected value."
    (
        np.testing.assert_allclose(dist, 0.003364, rtol=1e-3),
        "The distance from the RA/Dec to the SN does not match the expected value of 0.003364 arcsec.",
    )


def test_extract_object_from_healpix():
    healpix = 42924408
    nside = 2**11
    object_type = "SN"
    source = "OpenUniverse2024"
    id_array = extract_object_from_healpix(healpix, nside, object_type, source=source)
    test_id_array = np.load(pathlib.Path(__file__).parent / "testdata/test_healpix_id_array.npy")
    (
        np.testing.assert_array_equal(id_array, test_id_array),
        "The IDs extracted from the healpix do not match the expected values.",
    )

    object_type = "star"
    id_array = extract_object_from_healpix(healpix, nside, object_type, source=source)
    test_id_array = np.load(pathlib.Path(__file__).parent / "testdata/test_healpix_star_id_array.npy")
    (
        np.testing.assert_array_equal(id_array, test_id_array),
        "The IDs extracted from the healpix do not match the expected values.",
    )

