# Run image simulator and create an image file.
import numpy as np
import os
import glob
from multiprocessing import Pool

from snappl.logger import SNLogger


def write_image_list(output_path, run_dir, run_name, test_data_path):
    base_path = f"{output_path}/{run_dir}/{run_name}"
    all_images = os.listdir(base_path)

    filename = test_data_path / f"image_list_{run_name}.txt"
    print(f"Writing image list to {filename}")
    with open(filename, "w") as f:
        for item in all_images:
            SNLogger.debug(item)
            # There is an image, noise, and flags, and we don"t want to read the image thrice.
            if "image.fits" not in item and "flags.fits" not in item and "READ" not in item:
                whole_path = os.path.join(base_path, item)
                newpath = whole_path.split("_noise.fits")[0]
                SNLogger.debug(f"Writing {newpath} to image list")
                f.write(f"{newpath.split('cmeldorf')[-1]}\n")


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
    output_path=None,
    run_name_base=None,
    sed_spec=None,
    sed_wave_type=None,
    sed_flux_type=None,
    bulge_R=None,
    bulge_n=None,
    disk_R=None,
    disk_n=None,
    just_rotate=False,
    just_shift=False,
    im_sim_path=None,
    test_data_path=None,
    band=None,
    observation_id=1000,
):
    SNLogger.debug(f"USING OBS ID {observation_id}")

    if run_dir is None:
        run_dir = "OU24_psf_tests"
        SNLogger.debug(f"No run_dir provided, using default {run_dir}")

    if output_path is None:
        # come up with a better default path for this! This is just a temporary placeholder.
        output_path = "/scratch/photometry_test_data/simple_gaussian_test"
        SNLogger.debug(f"No output_path provided, using default {output_path}")

    if im_sim_path is None:
        im_sim_path = "/scratch/snappl/snappl/image_simulator.py"
    if test_data_path is None:
        test_data_path = "/scratch/campari/campari/tests/testdata"
        SNLogger.debug(f"No test_data_path provided, using default {test_data_path}")
    if just_rotate:
        assert images_aligned == True, "Cannot both just rotate and have images not aligned"
    if just_shift:
        assert images_aligned == True, "Cannot both just shift and have images not aligned"
    assert not (just_rotate and just_shift), "Cannot both just rotate and just shift"

    if run_name_base is None:
        raise ValueError("run_name_base must be provided")  # This is important to avoid accidentally overwriting data.
        # I want to make sure the user consciously chooses a name for the run.
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
    # This should probably be customizable at some point

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
        static_source_key = "--static-source-ra 128.00003 --static-source-dec 42.00003"\
            f" --static-source-mag {static_source_mag} {source_noise_key}"

    cmd_str = (
        f" python {im_sim_path} --seed {seed} --star-center 128 42 -n 0 --no-star-noise -b {run_name} --width 32"
        + f" --height 32 --pixscale 0.11 -t {mjd_str} --image-centers {image_centers_str} -θ {thetas_str} -r"
        + f" {sky_noise_key} -s 0 "
        + f" --no-star-noise {transient_noise_key} --overwrite {static_source_key if static_source else ''}"
        + f" --psf-class {psf_class} "
    )

    if transient_peak_mag is not None:
        cmd_str += f"--transient-peak-mag {transient_peak_mag} "
        cmd_str += "--transient-ra 128 --transient-dec 42 -n 1 "

    if static_source == "galaxy":
        cmd_str += f"--galaxy-kwargs bulge_R {bulge_R} bulge_n {bulge_n} disk_R {disk_R} disk_n {disk_n} "
    else:
        if any(param is not None for param in [bulge_R, bulge_n, disk_R, disk_n]):
            raise ValueError("Galaxy parameters provided but static_source is not set to 'galaxy'")

    if sed_spec is not None:
        cmd_str += f"--sed-spec {sed_spec} --sed-wave_type {sed_wave_type} --sed-flux_type {sed_flux_type} "

    if band is not None:
        cmd_str += f"--band {band} "

    cmd_str += f"--observation-id {observation_id} "

    SNLogger.debug(cmd_str)
    os.system(cmd_str)
    SNLogger.debug("Finished image simulation.")
    file_list = glob.glob(f"*{run_name}*")
    if not os.path.exists(f"{output_path}/{run_dir}/{run_name}"):
        os.makedirs(f"{output_path}/{run_dir}/{run_name}")
        print(f"Created directory {output_path}/{run_dir}/{run_name}")
    for item in file_list:
        print(f"Moving {item} to {output_path}/{run_dir}/{run_name}")
        os.system(f"mv {item} {output_path}/{run_dir}/{run_name}")

    write_image_list(output_path, run_dir, run_name, test_data_path)


def run_sims_in_parallel(
    seed_list=None,
    images_aligned=None,
    poisson_noise=None,
    sky_noise=None,
    static_source=None,
    static_source_mag=None,
    transient_peak_mag=None,
    mjd=None,
    psf_class=None,
    run_dir=None,
    output_path=None,
    run_name_base=None,
    sed_spec=None,
    sed_wave_type=None,
    sed_flux_type=None,
    bulge_R=None,
    bulge_n=None,
    disk_R=None,
    disk_n=None,
    just_rotate=False,
    just_shift=False,
    im_sim_path=None,
    test_data_path=None,
    band=None,
    observation_id=1000,
):
    SNLogger.debug(f"USING OBS ID {observation_id}")
    nprocs = len(seed_list)

    # Capture only the kwargs relevant to run_sim, before creating pool/nprocs
    sim_kwargs = dict(
        images_aligned=images_aligned,
        poisson_noise=poisson_noise,
        sky_noise=sky_noise,
        static_source=static_source,
        static_source_mag=static_source_mag,
        transient_peak_mag=transient_peak_mag,
        mjd=mjd,
        psf_class=psf_class,
        run_dir=run_dir,
        output_path=output_path,
        run_name_base=run_name_base,
        sed_spec=sed_spec,
        sed_wave_type=sed_wave_type,
        sed_flux_type=sed_flux_type,
        bulge_R=bulge_R,
        bulge_n=bulge_n,
        disk_R=disk_R,
        disk_n=disk_n,
        just_rotate=just_rotate,
        just_shift=just_shift,
        im_sim_path=im_sim_path,
        test_data_path=test_data_path,
        band=band,
        observation_id=observation_id,
    )

    with Pool(nprocs) as pool:
        results = []
        for seed in seed_list:
            print(f"Running seed = {seed} ----")
            r = pool.apply_async(run_sim, kwds={**sim_kwargs, "seed": seed})
            results.append(r)
        pool.close()
        # Call .get() on each result so exceptions are re-raised here
        for r in results:
            r.get()
        pool.join()
