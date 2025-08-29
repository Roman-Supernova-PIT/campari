
def get_psf_image(self, stamp_size, x=None, y=None, x_center=None,
                  y_center=None, pupil_bin=8, sed=None, oversampling_factor=1,
                  include_photonOps=False, n_phot=1e6, pixel=False, flux=1):

    if pixel:
        point = galsim.Pixel(1)*sed
        SNLogger.debug("Building a Pixel shaped PSF source")
    else:
        point = galsim.DeltaFunction()*sed

    # Note the +1s in galsim.PositionD below; galsim uses 1-indexed pixel positions,
    # whereas snappl uses 0-indexed pixel positions
    x_center += 1
    y_center += 1
    x += 1
    y += 1

    point = point.withFlux(flux, self.bpass)
    local_wcs = self.getLocalWCS(x, y)
    wcs = galsim.JacobianWCS(dudx=local_wcs.dudx/oversampling_factor,
                             dudy=local_wcs.dudy/oversampling_factor,
                             dvdx=local_wcs.dvdx/oversampling_factor,
                             dvdy=local_wcs.dvdy/oversampling_factor)
    stamp = galsim.Image(stamp_size*oversampling_factor,
                         stamp_size*oversampling_factor, wcs=wcs)

    if not include_photonOps:
        SNLogger.debug(f"in get_psf_image: {self.bpass}, {x_center}, {y_center}")

        psf = galsim.Convolve(point, self.getPSF(x, y, pupil_bin))
        return psf.drawImage(self.bpass, image=stamp, wcs=wcs,
                             method="no_pixel",
                             center=galsim.PositionD(x_center, y_center),
                             use_true_center=True)

    photon_ops = [self.getPSF(x, y, pupil_bin)] + self.photon_ops
    SNLogger.debug(f"Using {n_phot:e} photons in get_psf_image")
    result = point.drawImage(self.bpass, wcs=wcs, method="phot",
                             photon_ops=photon_ops, rng=self.rng,
                             n_photons=int(n_phot), maxN=int(n_phot),
                             poisson_flux=False,
                             center=galsim.PositionD(x_center, y_center),
                             use_true_center=True, image=stamp)
    return result


def get_galsim_SED_list(ID, dates, fetch_SED, object_type, sn_path, sed_out_dir=None):
    """Return the appropriate SED for the object for each observation.
    If you are getting truth SEDs, this function calls get_SED on each exposure
    of the object. Then, get_SED calls get_SN_SED or get_star_SED depending on
    the object type.
    If you are not getting truth SEDs, this function returns a flat SED for
    each exposure.

    Inputs:
    ID: the ID of the object
    exposures: the exposure table returned by fetch_images.
    fetch_SED: If true, get the SED from truth tables.
               If false, return a flat SED for each expsoure.
    object_type: the type of object (SN or star)
    sn_path: the path to the supernova data

    Returns:
    sedlist: list of galsim SED objects, length equal to the number of
             detection images.
    """
    sed_list = []
    if isinstance(dates, float):
        dates = [dates]  # If only one date is given, make it a list.
    for date in dates:
        sed = get_galsim_SED(ID, date, sn_path, obj_type=object_type, fetch_SED=fetch_SED)
        sed_list.append(sed)
        if sed_out_dir is not None:
            sed_df = pd.DataFrame({"lambda": sed._spec.x, "flux": sed._spec.f})
            sed_df.to_csv(f"{sed_out_dir}/sed_{ID}_{date}.csv", index=False)

    return sed_list


def get_galsim_SED(SNID, date, sn_path, fetch_SED, obj_type="SN"):
    """Return the appropriate SED for the object on the day. Since SN SEDs
    are time dependent but stars are not, we need to handle them differently.

    Inputs:
    SNID: the ID of the object
    date: the date of the observation
    sn_path: the path to the supernova data
    fetch_SED: If true, fetch true SED from the database, otherwise return a
                flat SED.
    obj_type: the type of object (SN or star)

    Internal Variables:
    lam: the wavelength of the SED in Angstrom
    flambda: the flux of the SED units in erg/s/cm^2/Angstrom

    Returns:
    sed: galsim.SED object
    """
    if fetch_SED:
        if obj_type == "SN":
            lam, flambda = get_SN_SED(SNID, date, sn_path)
        if obj_type == "star":
            lam, flambda = get_star_SED(SNID, sn_path)
    else:
        lam, flambda = [1000, 26000], [1, 1]

    sed = galsim.SED(galsim.LookupTable(lam, flambda, interpolant="linear"), wave_type="Angstrom", flux_type="fphotons")

    return sed


def get_star_SED(SNID, sn_path):
    """Return the appropriate SED for the star.
    Inputs:
    SNID: the ID of the object
    sn_path: the path to the supernova data

    Returns:
    lam: the wavelength of the SED in Angstrom (numpy  array of floats)
    flambda: the flux of the SED units in erg/s/cm^2/Angstrom
             (numpy array of floats)
    """
    filenum = find_parquet(SNID, sn_path, obj_type="star")
    pqfile = open_parquet(filenum, sn_path, obj_type="star")
    file_name = pqfile[pqfile["id"] == str(SNID)]["sed_filepath"].values[0]
    # SED needs to move out to snappl
    fullpath = pathlib.Path(Config.get().value("photometry.campari." + "paths.sims_sed_library")) / file_name
    sed_table = pd.read_csv(fullpath, compression="gzip", sep=r"\s+", comment="#")
    lam = sed_table.iloc[:, 0]
    flambda = sed_table.iloc[:, 1]
    return np.array(lam), np.array(flambda)


def get_SN_SED(SNID, date, sn_path, max_days_cutoff=10):
    """Return the appropriate SED for the supernova on the given day.

    Inputs:
    SNID: the ID of the object
    date: the date of the observation
    sn_path: the path to the supernova data

    Returns:
    lam: the wavelength of the SED in Angstrom
    flambda: the flux of the SED units in erg/s/cm^2/Angstrom
    """
    filenum = find_parquet(SNID, sn_path, obj_type="SN")
    file_name = "snana" + "_" + str(filenum) + ".hdf5"

    fullpath = os.path.join(sn_path, file_name)
    # Setting locking=False on the next line becasue it seems that you can't
    #   open an h5py file unless you have write access to... something.
    #   Not sure what.  The directory where it exists?  We won't
    #   always have that.  It's scary to set locking to false, because it
    #   subverts all kinds of safety stuff that hdf5 does.  However,
    #   because these files were created once in this case, it's not actually
    #   scary, and we expect them to be static.  Locking only matters if you
    #   think somebody else might change the file
    #   while you're in the middle of reading bits of it.
    sed_table = h5py.File(fullpath, "r", locking=False)
    sed_table = sed_table[str(SNID)]
    flambda = sed_table["flambda"]
    lam = sed_table["lambda"]
    mjd = sed_table["mjd"]
    SNLogger.debug(f"MJD values in SED: {np.array(mjd)}")
    bestindex = np.argmin(np.abs(np.array(mjd) - date))
    closest_days_away = np.min(np.abs(np.array(mjd) - date))

    if np.abs(closest_days_away) > max_days_cutoff:
        SNLogger.warning(
            f"WARNING: No SED data within {max_days_cutoff} days of "
            f"date. \n The closest SED is {closest_days_away} days away."
        )
    return np.array(lam), np.array(flambda[bestindex])


def load_SED_from_directory(sed_directory, wave_type="Angstrom", flux_type="fphotons"):
    """This function loads SEDs from a directory of SED files. The files must be in CSV format with
    two columns: "lambda" and "flux". The "lambda" column should contain the
    wavelengths in Angstroms, and the "flux" column should contain the fluxes in
    the appropriate units for the specified wave_type and flux_type.
    Inputs:
    sed_directory: str, the path to the directory containing the SED files.

    Returns:
    sed_list: list of galsim SED objects. (Temporary until we remove galsim)
    """
    SNLogger.debug(f"Loading SEDs from {sed_directory}")
    sed_list = []
    for file in pathlib.Path(sed_directory).glob("*.csv"):
        sed_table = pd.read_csv(file)

        flambda = sed_table["flux"]
        lam = sed_table["lambda"]
        # Assuming units are Angstroms how can I check this?
        sed = galsim.SED(galsim.LookupTable(lam, flambda, interpolant="linear"),
                         wave_type=wave_type, flux_type=flux_type)
        sed_list.append(sed)
    return sed_list

