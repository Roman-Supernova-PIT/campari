# Standard Libary
import uuid
import warnings

# Common Library
import numpy as np
import pathlib
import pytest

# Astronomy Library
from astropy.utils.exceptions import AstropyWarning
from erfa import ErfaWarning


# SNPIT
from campari.data_construction import find_all_exposures

from snappl.dbclient import SNPITDBClient
from snappl.diaobject import DiaObject
from snappl.imagecollection import ImageCollection

warnings.simplefilter("ignore", category=AstropyWarning)
warnings.filterwarnings("ignore", category=ErfaWarning)


@pytest.fixture(scope="module")
def campari_test_data(cfg):
    return cfg.value("system.paths.campari_test_data")


def test_find_diaobject():
    provenance_tag = "ou2024"
    process = "load_ou2024_diaobject"
    diaobj = DiaObject.find_objects(name=20172782, collection="snpitdb",
                                    provenance_tag=provenance_tag, process=process)[0]
    regression_data = {
        "iauname": None,
        "id": uuid.UUID("f00e4a1e-d546-46c3-98f9-e854e2fd8e70"),
        "mjd_discovery": 62450.0,
        "mjd_end": 62881.0,
        "mjd_peak": 62476.5078125,
        "mjd_start": 62450.0,
    }

    np.testing.assert_equal(diaobj.iauname, regression_data["iauname"])
    np.testing.assert_equal(diaobj.id, regression_data["id"])
    np.testing.assert_equal(diaobj.mjd_discovery, regression_data["mjd_discovery"])
    np.testing.assert_equal(diaobj.mjd_end, regression_data["mjd_end"])
    np.testing.assert_equal(diaobj.mjd_peak, regression_data["mjd_peak"])
    np.testing.assert_equal(diaobj.mjd_start, regression_data["mjd_start"])


def test_get_image_collection():
    image_collection = "snpitdb"
    provenance_tag = "ou2024"
    process = "load_ou2024_image"
    dbclient = SNPITDBClient()

    img_collection = ImageCollection().get_collection(
        collection=image_collection, provenance_tag=provenance_tag, process=process, dbclient=dbclient
    )

    np.testing.assert_equal(img_collection.provenance.id, uuid.UUID("305dbfc3-bbb4-8dde-f008-e616e3625e51"))


def test_get_image_collection_missing_provenance():
    image_collection = "snpitdb"
    provenance_tag = "nonexistent_tag"
    process = "load_ou2024_image"
    dbclient = SNPITDBClient()
    with pytest.raises(ValueError):
        ImageCollection().get_collection(
            collection=image_collection, provenance_tag=provenance_tag, process=process, dbclient=dbclient
        )


def test_find_exposures():

    dbclient = SNPITDBClient()

    diaobj_provenance_tag = "ou2024"
    diaobj_process = "load_ou2024_diaobject"
    diaobj = DiaObject.find_objects(name=20172782, collection="snpitdb",
                                    provenance_tag=diaobj_provenance_tag, process=diaobj_process)[0]

    diaobj.mjd_start = 62654.0
    diaobj.mjd_end = 62958.0
    image_list, _ = find_all_exposures(diaobj=diaobj, band="Y106",
                                       truth="simple_model", image_collection="ou2024",
                                       dbclient=dbclient)

    np.testing.assert_equal(len(image_list), 135)

    observation_ids = [int(i.observation_id) for i in image_list]
    observation_ids = np.sort(np.array(observation_ids))

    regression_observation_ids = np.load(pathlib.Path(__file__).parent / "testdata/test_find_exposures_pointings.npy")
    np.testing.assert_array_equal(observation_ids, regression_observation_ids)

# This worked once but I am going to wait to re-enable until I have my own testing database set up.


# def test_write_lc_to_db():
#     dbclient = SNPITDBClient()

#     diaobj_provenance_tag = "ou2024_truth"
#     diaobj_process = "load_ou2024_diaobject"
#     diaobj = DiaObject.find_objects(name=20172782, collection="snpitdb",
#                                     provenance_tag=diaobj_provenance_tag, process=diaobj_process)[0]

#     provenance_tag = "ou2024"
#     process = "load_ou2024_image"

# #   For some reason, fetching the dates is very slow right now. In the interest of time, I am saying that all images
# #   are detections to make this test run faster. I should restore the original date later.
#     #    diaobj.mjd_start = 62654.0
#     diaobj.mjd_start = 0
#     diaobj.mjd_end = 62958.0
#     image_list_all, _ = find_all_exposures(diaobj=diaobj, band="Y106",
#                                            truth="simple_model", image_collection="snpitdb",
#                                            provenance_tag=provenance_tag, process=process, dbclient=dbclient)


#     # This is what is slowing things down a lot right now.
#     # Select only a few images for the test lightcurve.
#     image_list = []
#     for i in image_list_all:
#         if i.mjd > diaobj.mjd_start and i.mjd < diaobj.mjd_end:
#             image_list.append(i)
#         if len(image_list) >= 5:
#             break

#     flux = np.array([120.0, 95.0, 80.0, 60.0, 50.0])
#     sigma_flux = np.array([10.0, 12.0, 9.0, 11.0, 10.0])

#     lc_model = campari_lightcurve_model(
#         flux=flux,
#         sigma_flux=sigma_flux,
#         image_list=image_list,
#         cutout_image_list=image_list,
#         LSB=25.0,
#         diaobj=diaobj,
#         sky_background=[0.0] * len(image_list),
#     )

#     # The data values are arbitary, just to check that the lc is constructed properly.
#     lc = build_lightcurve(diaobj, lc_model)

#     save_lightcurve(lc=lc, identifier="test", psftype="romanpsf", save_to_database=True, dbclient=dbclient,
#                     diaobj_pos=None, new_provenance=False)
