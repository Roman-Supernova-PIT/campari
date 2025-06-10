=====================================================================
Example running campari on Perlmutter in a docker image with podman
======================================================================

.. toctree::

Getting set up
==============

Secure the podman (docker) image
################################

You will need to do this once to get started.  Thereafter, you *shouldn't* need to do it again very often; usually, you will only need to pull it when the environment has been updated and you need the updated version.

This example uses the ``cpu`` docker image defined by the `Roman SNPIT environment <https://github.com/Roman-Supernova-PIT/environment>`_.  As of this writing, it works with version 0.0.7 of that environment, but hopefully it will continue to work with the latest version of that environment.  If things don't work, try adding ``-0.0.7`` to the name of docker images below.

On perlmutter, run::

  podman-hpc login registry.nersc.gov

Give it your usual nersc username and password (*without* the OTP).  You may be able to skip this step, as once you've logged in, it will often (but not always) remember that.

Pull the image with::

  podman-hpc pull registry.nersc.gov/m4385/rknop/roman-snpit-env:cpu

If this doesn't work, or if you aren't able to log into the NERSC image registry, you can also try pulling the image from dockerhub by pulling the image ``rknop/roman-snpit-env:cpu``.

Verify that you got it with the command ``podman-hpc images``, which should show (at least)::

  REPOSITORY                                      TAG                 IMAGE ID      CREATED      SIZE        R/O
  registry.nersc.gov/m4385/rknop/roman-snpit-env  cpu                 2b4bd71145cd  4 days ago   2.88 GB     false
  registry.nersc.gov/m4385/rknop/roman-snpit-env  cpu                 2b4bd71145cd  4 days ago   2.88 GB     true

(The Image ID, created time, and (probably) size are likely to be different from what's in this example.)  The ``R/O false`` version is the image you pulled directly, and will only be available on the login node where you ran the pull.  More important is the ``R/O true`` (the readonly, or "squashed", image).  This one should be available on all login nodes and all compute nodes.  (You can verify this by running ``podman-hpc images`` on another login node; you should see only the ``R/O true`` image there.)

If you've pulled images before, and you're now working on a new login node, you will only see the ``R/O=true`` image.  That's the only image you really need, so in that case there's no need to pull the image again.

Pick a directory to work in
###########################

From now on, I will call this directory your *parent* directory.  A good place to work is a directory you make underneath your ``$SCRATCH`` directory, but if you know what you're doing, you might also put this somewhere on CFS.  You probably don't want to do this on your home directory, because your disk quota there is very limited.

Create output directories
#########################

Under your parent directory, use ``mkdir`` to create each of the following directories::

  campari_out_dir
  campari_debug_dir


Get campari
###########

In your parent directory, run::

   git clone https://github.com/Roman-Supernova-PIT/campari.git

(If you know what you're doing, you might clone ``git@github.com:...`` instead of ``https:...``.)


Running interactively
=====================

Make sure you are in your parent directory.

Copy the file ``interactive_podman.sh`` from the ``examples/perlmutter`` directory (which, under your parent directory, will be at ``campari/examples/perlmutter/interactive_podman.sh``) to your parent directory.

Make sure you're on the machine where you want to run.  If you're going to do something short, it's probably OK to do it on the login node, but it's antisocial to do big things on login nodes.  For running the tests in this example, the login node is *probably* fine, but you might want to get yourself an interactive node to work on.  (Be careful not to let the interactive node sit around after you're done with it, as you will be eating up our nersc allocation.)

Run::

  bash interactive_podman.sh

You are now inside the container; you can tell that this has happened because your prompt will have changed to something like ``root@0a168a1f80df:/#`` (the actual hex barf will be different).


Installing campari inside the container
#######################################

This step will eventually be unnecessary, as campari will be in the snpit environment.  For now, though, things are under heavy developoment, so every time time you start a container, you have to install campari in it.  Fortunately, it's pretty fast.  Run::

  cd /campari
  pip install -e .

If all is well, there will be a whole bunch of stuff on the screen you can ignore, ending in something like::

  Successfully built campari
  Installing collected packages: campari
  Successfully installed campari-0.1.dev330+gfe05108

(where the version number at the end will almost certainly be different for you).


Running the tests
#################

Assuming everything has worked, you should be able to run all the campari tests in this environment.  Those tests include a regression test, so you will run a (small, not-terribly-meaningful) full run of the scene modelling in so doing.

Get to the test directory with::

  cd /campari/campari/tests

Run the tests with::

  pytest -v

If all is well, at the end you should be told that lots of tests passed, that no tests failed, that there were no errors, and that there were a whole bunch of warnings (which you will just ignore).

Inside the container, do::

  ls -l /campari_out_dir
  ls -l /campari_debug_dir

You should see files that were just written as a result of your test run.  You can find these same directories outside the container as subdirectories of your parent driectory.


Running an example lightcurve
#############################

TODO -- we should make this a bigger one that takes longer, since a quick-and-dirty one happens when running the tests above.


Submitting a job with sbatch
============================

TODO
