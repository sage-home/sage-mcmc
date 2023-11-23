# sage-mcmc
MCMC with sage


## Steps to run sage-mcmc on OzSTAR supercomputer:
Steps to run sage-mcmc (only emcee right now) on OzSTAR supercomputer:
- Load the required modules. On NT, I use the following on the terminal:
    - `ml gcc/12.2.0 openmpi/4.1.4 foss/2022b gsl/2.7`
    - `ml python/3.10.8 mpi4py/3.1.4 scipy-bundle/2023.02 numpy/1.24.2-scipy-bundle-2023.02`
    - `ml h5py/3.8.0 astropy/5.2.2 tqdm/4.64.1`

The final list of loaded modules should look like (the order is not, or at least should not, be important):
```=
   1) nvidia/.latest    10) libpciaccess/0.17    19) openblas/0.3.21       28) tcl/8.6.12              37) hdf5/1.14.0
   2) slurm/.latest     11) hwloc/2.8.0          20) flexiblas/3.2.1       29) sqlite/3.39.4           38) h5py/3.8.0
   3) gcccore/12.2.0    12) openssl/1.1          21) fftw/3.3.10           30) gmp/6.2.1               39) libyaml/0.2.5
   4) zlib/1.2.12       13) libevent/2.1.12      22) fftw.mpi/3.3.10       31) libffi/3.4.4            40) pyyaml/6.0
   5) binutils/2.39     14) ucx/1.13.1           23) scalapack/2.2.0-fb    32) python/3.10.8           41) astropy/5.2.2
   6) gcc/12.2.0        15) libfabric/1.16.1     24) foss/2022b            33) mpi4py/3.1.4            42) numpy/1.24.2-scipy-bundle-2023.02
   7) numactl/2.0.16    16) pmix/4.2.2           25) bzip2/1.0.8           34) pybind11/2.10.3         43) gsl/2.7
   8) xz/5.2.7          17) ucc/1.1.0            26) ncurses/6.3           35) scipy-bundle/2023.02    44) tqdm/4.64.1
   9) libxml2/2.10.3    18) openmpi/4.1.4        27) libreadline/8.2       36) szip/2.1.1
```

- Save these modules under the name ``sage-emcee`` using `ml save sage-emcee`
- Create a sage-home directory and from this directory, run the following commands to clone the two repos
    - `git clone git@github.com:sage-home/sage-model`
    - `git clone git@github.com:sage-home/sage-mcmc`
   
- Install emcee, schwimbadd (python -m pip install pkgname)
    - You might need to set `export PIP_REQUIRE_VIRTUALENV=FALSE` to install the packages
- Change the values at the bottom of ``sage-emcee.py`` to suit your needs. For example, set the target redshift, and the observational GSMF data used to constrain the target redshift. Currently, `sage-MCMC` only works with `GSMF` constraints, but further constraints need to be added to better rule parameter space
- Adapt the ``sage-emcee.slurm`` script to suit your needs (this should be placed in the `sage-mcmc` directory)
```
#!/bin/bash -l
#SBATCH --job-name=<JOBNAME>
#SBATCH --mail-user=<YOUR EMAIL>
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00
#SBATCH -N <number of nodes>
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --export==NONE

ml purge
# assuming you have saved the modules under 
# the name `sage-emcee` (using ml save)
ml restore sage-emcee
ml
srun python ./sage_emcee.py
```

- Do a test run with `mpirun -np 2 python ./sage_emcee.py` - be sure to confirm that `sage-model` is being compiled with `mpicc` (otherwise, the parallel run will be corrupted). It should happen automatically (regardless of the contents of the `Makefile` within the `sage-model` directory) - however, if you have issues, then set `USE-MPI=yes`  within `sage-model/Makefile`
- Change directory into `sage-mcmc`, and submit the parallel job as `sbatch sage-emcee.slurm`
- Look at the output directory to confirm that the output hdf5 file is generated. This output will be in an hdf5 file called: `sage_emcee_{VARYING_SAGE_PARAMS}_z_<target_redshift>_nwalkers_<nwalkers>.hdf5`. The chain is saved under a special name (think hdf5 dataset) that without the prefix `sage_emcee` or the suffix `.hdf5`. 
- Analyse data with `plot_chains.ipynb` - set the chain name and hte  directory containing the chain results. The chain filename will be automatically reconstructed from the chain name. If you change the filename generation within `sage_emcee`, then you will have to change that within `plot_chains.ipynb`

