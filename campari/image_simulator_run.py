# Run image simulator and create an image file.
import numpy as np
import os
import glob
from multiprocessing import Pool

from snappl.logger import SNLogger

images_aligned = False
poisson_noise = True
sky_noise = True
static_source = "galaxy"  # "galaxy"  #"point"  # False  #
static_source_mag = 22

just_shift = False
just_rotate = False

transient_peak_mag = 24

#sed_spec = "/campari/campari/tests/snflux_1a_peakmjd.csv"
sed_spec = None
#sed_spec = "'2.718 ** ( -0.5 * ( (wave - 5100) / 4 )**2 )'" # Delta function-like Gaussian
#sed_spec = None
#sed_spec = "/campari/campari/tests/test_economical_sed.csv" # Delta function-like Gaussian
#sed_wave_type = "Angstrom"
#sed_flux_type = "flambda" # I am not sure if this is true, I can't find the units for the Hsiao template.

# /pscratch/sd/c/cmeldorf/
im_sim_path = "/scratch/snappl/snappl/image_simulator.py"
testdata_path = "/scratch/campari/campari/tests/testdata"
photom_path = "/scratch/photometry_test_data/simple_gaussian_test"

run_dir = "OU24_psf_tests"
#run_name_base = "nohost_nonoise"
#run_name_base = "nophot_sanity_check"
#run_name_base = "testingpsf"
run_name_base = "bothnoise_unaligned_realistichost_faintsource_ou2024_photshoot"

psf_class = "ou24PSF_slow_photonshoot"  # "gaussian"  #"ou24PSF"
#mjd = np.arange(60010, 60060, 0.25)
#mjd = np.array([60030])
mjd = np.arange(60000, 60075, 0.5)

#mjd = np.arange(60010, 60060, 0.5)


#mjd = np.arange(60010, 60030, 10.0)
#mjd = np.array([60000, 60030])
#mjd = np.arange(60000, 60075, 10.0)

bulge_R = 2
bulge_n = 3
disk_R = 4
disk_n = 1

if just_rotate:
     assert images_aligned == True, "Cannot both just rotate and have images not aligned"
if just_shift:
     assert images_aligned == True, "Cannot both just shift and have images not aligned"

assert not (just_rotate and just_shift), "Cannot both just rotate and just shift"

###############################################


def run_sim(
    seed=None,
    images_aligned=None,
    poisson_noise=None,
    sky_noise=None,
    static_source=None,
    static_source_mag=None,
    transient_peak_mag=None,
    mjd=None,
    psf_class=None,
    run_dir=None,
    photom_path=None,
    run_name_base=None,
):
    run_name = run_name_base + f"seed{seed}"

    np.set_printoptions(linewidth=np.inf, threshold=np.inf)
    mjd_str = np.array2string(mjd, separator=" ")
    mjd_str = mjd_str.replace("[", "").replace("]", "")

    thetas = np.linspace(0, 360, len(mjd))
    thetas_str = np.array2string(thetas, separator=" ")
    thetas_str = thetas_str.replace("[", "").replace("]", "")

    num_ims_per_side = np.ceil(np.sqrt(len(mjd))).astype(int)
    image_x = np.linspace(128 - 0.0002, 128 + 0.0002, num_ims_per_side)
    image_y = np.linspace(42 - 0.0002, 42 + 0.0002, num_ims_per_side)

    xx, yy = np.meshgrid(image_x, image_y)
    xx = list(xx.flatten())
    yy = list(yy.flatten())

    image_centers = [float(item) for pair in zip(xx, yy) for item in pair]
    image_centers = np.array(image_centers)
    image_centers = image_centers[: len(mjd) * 2]
    image_centers_str = np.array2string(image_centers, separator=" ")
    image_centers_str = image_centers_str.replace("[", "").replace("]", "")

    thetas_str = "0." if (images_aligned or just_rotate) else thetas_str
    image_centers_str = "128. 42." if (images_aligned or just_shift) else image_centers_str

    source_noise_key = "--no-static-source-noise" if not poisson_noise else ""
    transient_noise_key = "--no-transient-noise" if not poisson_noise else ""
    sky_noise_key = 30 if sky_noise else 0
    if static_source is not False:
        static_source_key = f"--static-source-ra 128.00003 --static-source-dec 42.00003 --static-source-mag {static_source_mag} {source_noise_key}"


    cmd_str = (
        f" python {im_sim_path} --seed {seed} --star-center 128 42 -n 0 --no-star-noise -b {run_name} --width 32"
        + f" --height 32 --pixscale 0.11 -t {mjd_str} --image-centers {image_centers_str} -θ {thetas_str} -r {sky_noise_key} -s 0 "
        + f" --no-star-noise {transient_noise_key} --overwrite {static_source_key if static_source else ''} --psf-class {psf_class} "
    )

    if transient_peak_mag is not None:
        cmd_str += f"--transient-peak-mag {transient_peak_mag} "
        cmd_str += "--transient-ra 128 --transient-dec 42 -n 1 "

    if static_source == "galaxy":
        cmd_str += f"--galaxy-kwargs bulge_R {bulge_R} bulge_n {bulge_n} disk_R {disk_R} disk_n {disk_n} "

    if sed_spec is not None:
        cmd_str += f"--sed-spec {sed_spec} --sed-wave_type {sed_wave_type} --sed-flux_type {sed_flux_type} "

    # if "ou" in psf_class:
    #     cmd_str += " --use-roman-wcs"
    SNLogger.debug(cmd_str)
    os.system(cmd_str)
    SNLogger.debug("Finished image simulation.")
    file_list = glob.glob(f"*{run_name}*")
    SNLogger.debug(f"Expected {np.size(mjd) * 3} files, found {len(file_list)}")
    assert len(file_list) == np.size(mjd) * 3, f"Expected {np.size(mjd) * 3} files, found {len(file_list)}"
    SNLogger.debug("Size Check Passed.")
    if not os.path.exists(f"{photom_path}/{run_dir}/{run_name}"):
        os.makedirs(f"{photom_path}/{run_dir}/{run_name}")
        print(f"Created directory {photom_path}/{run_dir}/{run_name}")
    for item in file_list:
        print(f"Moving {item} to {photom_path}/{run_dir}/{run_name}")
        os.system(f"mv {item} {photom_path}/{run_dir}/{run_name}")
    SNLogger.debug("Down here somehow.")

    base_path = f"{photom_path}/{run_dir}/{run_name}"
    all_images = os.listdir(base_path)

    filename = testdata_path + "/test_gaussims_" + run_name + ".txt"
    print(f"Writing image list to {filename}")
    with open(filename, "w") as f:
        for item in all_images:
            # There is an image, noise, and flags, and we don"t want to read the image thrice.
            if "image" not in item and "flags" not in item and "READ" not in item:
                whole_path = os.path.join(base_path, item)
                newpath = whole_path.split("_noise.fits")[0]
                f.write(f"{newpath.split('cmeldorf')[-1]}\n")


# ###############################################


# seed_list = [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]


#seed_list = [45, 46, 47, 48]
#seed_list = [49, 50, 51, 52]

seed_list = [53, 54, 55, 56, 57, 58, 59, 60]

#seed_list = [51]

nprocs = len(seed_list)
with Pool(nprocs) as pool:
    for seed in seed_list:
        print("Running seed =", seed, "------------------------------------#######################################")
        pool.apply_async(run_sim, kwds={"seed": seed, "images_aligned": images_aligned,
                                        "poisson_noise": poisson_noise, "sky_noise": sky_noise, "static_source": static_source,
                                        "static_source_mag": static_source_mag, "transient_peak_mag": transient_peak_mag, "mjd": mjd,
                                        "psf_class": psf_class, "run_dir": run_dir, "photom_path": photom_path, "run_name_base": run_name_base})
    pool.close()
    pool.join()

        # run_sim(seed=seed, images_aligned=images_aligned, poisson_noise=poisson_noise, sky_noise=sky_noise,
        #         static_source=static_source, static_source_mag=static_source_mag, transient_peak_mag=transient_peak_mag,
        #         mjd=mjd, psf_class=psf_class, run_dir=run_dir, photom_path=photom_path,
        #         run_name_base=run_name_base)


