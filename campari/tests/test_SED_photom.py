import pathlib
import subprocess

import numpy as np
import pytest

from astropy.table import Table

from snappl.logger import SNLogger

from campari.tests.test_gausspsfs import generate_diagnostic_plots, perform_gaussianity_checks

imsize = 19
base_cmd = [
        "python", "../RomanASP.py",
        "-t", "1",
        "-n", "0",
        "-f", "R062",
        "--ra", "128.0",
        "--dec", "42.0",
        "--transient_start", "60010",
        "--transient_end", "60060",
        "--photometry-campari-psf-transient_class", "ou24PSF_slow_photonshoot",
        "--photometry-campari-psf-galaxy_class", "ou24PSF",
        "--photometry-campari-use_real_images",
        "--diaobject-collection", "manual",
        "--no-photometry-campari-fetch_SED",
        "--photometry-campari-grid_options-spacing", "1",
        "--photometry-campari-grid_options-subsize", "4",
        "--photometry-campari-cutout_size", str(imsize),
        "--photometry-campari-weighting",
        "--photometry-campari-subtract_background_method", "calculate",
        "--image-collection", "manual_fits",
        "--photometry-campari_simulations-run_name", "gauss_source_no_grid",
        "--image-collection-basepath", "/photometry_test_data/simple_gaussian_test/sig1.0",
        "--image-collection-subset", "threefile",
        "--no-save-to-db"
    ]


def test_nohost_bothnoise_HsiaoSEDsimulated_Hsiaofit():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_Hsiao_sed_redo_seed45.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "none"]

    cmd += ["--save_model"]
    cmd += ["--nprocs", "10"]
    cmd += ["--diaobject-name", "130"]

    # Fitting with a blackbody SED at ~9030 Kelvin
    cmd += ["--SED_file"]
    cmd += [pathlib.Path(__file__).parent / "snflux_1a_peakmjd.csv"]

    cmd.append("--photometry-campari-psf-transient_class")
    cmd.append("ou24PSF_slow_photonshoot")
    cmd.append("--photometry-campari-psf-galaxy_class")
    cmd.append("ou24PSF_slow_photonshoot")
    SNLogger.debug(cmd)

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read("/campari_out_dir/130_R062_ou24psf_slow_photonshoot_lc.ecsv")

    mjd = lc["mjd"]
    peakflux = 10 ** ((24 - 33) / -2.5)
    start_mjd = 60010
    peak_mjd = 60030
    end_mjd = 60060
    flux = np.zeros(len(mjd))
    flux[np.where(mjd < peak_mjd)] = peakflux * (mjd[np.where(mjd < peak_mjd)] - start_mjd) / (peak_mjd - start_mjd)
    flux[np.where(mjd >= peak_mjd)] = peakflux * (mjd[np.where(mjd >= peak_mjd)] - end_mjd) / (peak_mjd - end_mjd)

    try:
        np.testing.assert_allclose(
            lc["flux"],
            flux,
            rtol=5e-6,  # Why does this need to be ~50x higher than the gaussian version?
        )
        SNLogger.debug(lc["flux_err"])
        np.testing.assert_allclose(
            lc["flux_err"], 2.311065590128104, atol=1e-7
        )  # I believe this is smaller because the
        # PSF is a different shape?
    except AssertionError as e:
        plotname = "HsiaoRecoverySED_diagnostic_comparison"
        generate_diagnostic_plots("130_R062_ou24psf_slow_photonshoot", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e

@pytest.mark.skip(reason="This test is superseded by tests with noise.")
def test_nohost_nonoise_HsiaoSEDsimulated_Hsiaofit():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_Hsiao_sed_nonoise_seed45.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "none"]

    cmd += ["--save_model"]
    cmd += ["--nprocs", "10"]
    cmd += ["--diaobject-name", "131"]

    # Fitting with a blackbody SED at ~9030 Kelvin
    cmd += ["--SED_file"]
    cmd += [pathlib.Path(__file__).parent / "snflux_1a_peakmjd.csv"]

    cmd.append("--photometry-campari-psf-transient_class")
    cmd.append("ou24PSF_slow_photonshoot")
    cmd.append("--photometry-campari-psf-galaxy_class")
    cmd.append("ou24PSF_slow_photonshoot")
    SNLogger.debug(cmd)

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read("/campari_out_dir/131_R062_ou24psf_slow_photonshoot_lc.ecsv")

    mjd = lc["mjd"]
    peakflux = 10 ** ((24 - 33) / -2.5)
    start_mjd = 60010
    peak_mjd = 60030
    end_mjd = 60060
    flux = np.zeros(len(mjd))
    flux[np.where(mjd < peak_mjd)] = peakflux * (mjd[np.where(mjd < peak_mjd)] - start_mjd) / (peak_mjd - start_mjd)
    flux[np.where(mjd >= peak_mjd)] = peakflux * (mjd[np.where(mjd >= peak_mjd)] - end_mjd) / (peak_mjd - end_mjd)

    try:
        residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
        perform_gaussianity_checks(residuals_sigma, measuredflux=lc["flux"], trueflux=flux)
    except AssertionError as e:
        plotname = "HsiaoRecoverySED_nonoise_diagnostic_comparison"
        generate_diagnostic_plots("131_R062_ou24psf_slow_photonshoot", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e


def test_nohost_bothnoise_HsiaoSEDsimulated_improvedBBSEDfit():
    cmd = base_cmd + [
        "--img_list",
        pathlib.Path(__file__).parent / "testdata/test_gaussims_Hsiao_sed_redo_seed45.txt",
    ]
    cmd += ["--photometry-campari-grid_options-type", "none"]

    cmd += ["--nprocs", "10"]
    cmd += ["--diaobject-name", "129"]
    cmd += [
        "--prebuilt_static_model",
        "/campari_debug_dir/psf_matrix_ou24PSF_slow_photonshoot_083d7700-0f25-41af-b7f2-661896b36ed8_100_images0_points.npy",
    ]

    # Fitting with a blackbody SED at ~14000 Kelvin
    cmd += ["--SED_file"]
    cmd += [pathlib.Path(__file__).parent / "test_bb_sed_improved.csv"]

    cmd.append("--photometry-campari-psf-transient_class")
    cmd.append("ou24PSF_slow_photonshoot")
    cmd.append("--photometry-campari-psf-galaxy_class")
    cmd.append("ou24PSF_slow_photonshoot")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Check accuracy
    lc = Table.read("/campari_out_dir/129_R062_ou24psf_slow_photonshoot_lc.ecsv")

    mjd = lc["mjd"]
    peakflux = 10 ** ((24 - 33) / -2.5)
    start_mjd = 60010
    peak_mjd = 60030
    end_mjd = 60060
    flux = np.zeros(len(mjd))
    flux[np.where(mjd < peak_mjd)] = peakflux * (mjd[np.where(mjd < peak_mjd)] - start_mjd) / (peak_mjd - start_mjd)
    flux[np.where(mjd >= peak_mjd)] = peakflux * (mjd[np.where(mjd >= peak_mjd)] - end_mjd) / (peak_mjd - end_mjd)

    try:
        residuals_sigma = (lc["flux"] - flux) / lc["flux_err"]
        perform_gaussianity_checks(residuals_sigma, measuredflux=lc["flux"], trueflux=flux)
    except AssertionError as e:
        plotname = "BBSED_diagnostic_comparison_improved"
        generate_diagnostic_plots("129_R062_ou24psf_slow_photonshoot", imsize, plotname, trueflux=flux)
        SNLogger.debug(f"Generated saved diagnostic plots to /campari_debug_dir/{plotname}.png")
        SNLogger.debug(e)
        raise e

# Code used to generate the blackbody SED approximation. Eventually this needs to be implemented
# in campari somehow.


# from roman_imsim.utils import roman_utils
# roman_bandpasses = galsim.roman.getBandpasses()

# sed = galsim.SED("/pscratch/sd/c/cmeldorf/campari/campari/tests/snflux_1a_peakmjd.csv", wave_type="Angstrom", flux_type="flambda", fast=False)

# tot_flux = 0
# xdata = []
# ydata = []

# sed = sed.withFlux(1.0, roman_bandpasses["W146"])
# for b in roman_bandpasses:
#     if "W146" in b:
#         continue

#     flux = sed.calculateFlux(roman_bandpasses[b])
#     xdata.append(roman_bandpasses[b].effective_wavelength * 10)
#     ydata.append(flux)

# xdata = np.array(xdata)
# ydata = np.array(ydata)

# plt.legend()
# sed.calculateFlux(roman_bandpasses["Y106"])

# spectra = pd.read_csv("/pscratch/sd/c/cmeldorf/campari/campari/tests/snflux_1a_peakmjd.csv", delim_whitespace=True, header=None)
# wa = np.array(spectra[0]) * u.AA
# plt.plot(wa, spectra[1] *1e15)

# from scipy.optimize import curve_fit
# import pylab as plt
# import numpy as np
# from astropy import units as u
# import pandas as pd

# import galsim


# def blackbody_lam(lam, T):
#     print("temperature:", T)
#     ''' Blackbody as a function of wavelength (um) and temperature (K).

#     returns units of erg/s/cm^2/cm/Steradian
#     '''
#     # convert angstrom to metres
#     lam = lam * 1e-10
#     from scipy.constants import h,k,c
#     #lam = 1e-6 * lam # convert to metres
#     # lam = lam.to(u.m).value
#     val = 2*h*c**2 / (lam**5 * (np.exp(h*c / (lam*k*T)) - 1))
#     return val

# def func3(xdata, T1):
#     def blackbody_wrapper(lam):
#         return blackbody_lam(lam, T1)
#     sed = galsim.SED(blackbody_wrapper, wave_type="Angstrom", flux_type="flambda")
#     sed = sed.withFlux(1, roman_bandpasses["W146"])
#     fluxes = []
#     for b in roman_bandpasses:
#         #print(b)
#         if "W146" in b:
#             continue
#         fluxes.append(sed.calculateFlux(roman_bandpasses[b]))
#     #print(fluxes)
#     return fluxes

# func3(xdata, 7000)

# popt, pcov = curve_fit(func3, xdata, ydata, p0=(8000))

# plt.subplot(1,2,1)
# plt.title("Flux measured in Roman Passbands")

# plt.scatter(xdata, ydata, label='Data')
# ybest = func3(xdata, *popt)
# plt.scatter(xdata, ybest, label='Fit')
# plt.legend()

# try:
#     wa = wa.value
# except:
#     pass
# cut = np.where((wa > 4000) & (wa < 24150))
# wa = np.array(wa)
# wa_reduced = wa[cut]
# spectra_reduced = np.array(spectra[1])[cut]
# plt.xlabel("Wavelength (Angstrom)")
# plt.ylabel("Normalized Flux")
# plt.subplot(1,2,2)
# plt.plot(wa_reduced, spectra_reduced / np.max(spectra_reduced), label='True spectrum')
# plt.title("Full Spectrum ")
# lam = np.linspace(4000, 24150, 100)
# bb_spec = blackbody_lam(lam, *popt) / np.max(blackbody_lam(lam, *popt))
# plt.plot(lam, bb_spec, label = "Fit Blackbody")
# #plt.plot(lam, blackbody_lam(lam, 5000) / np.max(blackbody_lam(lam, 5000)))
# #plt.plot(lam, blackbody_lam(lam, 500000) / np.max(blackbody_lam(lam, 500000)))
# plt.legend()

# plt.xlabel("Wavelength (Angstrom)")

# pd.DataFrame({"wavelength": lam, "flux": bb_spec}).to_csv("test_bb_sed_improved.csv", index=False, sep = " ", header = False)
#