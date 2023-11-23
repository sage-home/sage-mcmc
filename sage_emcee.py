from os import path
import shutil
import sys
import logging
import functools
from tempfile import mkdtemp

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger.setLevel(logging.DEBUG)
sys.path.insert(0, '/home/msinha/research/codes/sage-home/sage-model/')

def add_local_sage_path(sage_libpath=None):
    import os
    import sys
    sage_pylib_fname = "sage.py"
    if not sage_libpath:
        sage_libpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../sage-model'))

    print(f"{sage_libpath = }")
    # Check that `sage.py` is in sage_libpath
    try:
        f = open(f"{sage_libpath}/{sage_pylib_fname}", 'r')
    except FileNotFoundError:
        msg = f"{sage_pylib_fname} not found in directory: {sage_libpath}. Please pass the directory "
        msg += "where the source code for 'sage_model' in located."
        raise ValueError(msg)
    else:
        f.close()

    logger.info(f"Adding sage library path = {sage_libpath}")
    sys.path.insert(0, sage_libpath)

def compile_sage_pythonext(sage_libpath=None, verbose=True):
    ## assumes that the root of the sage repo is in PATH
    ## **and** sage.py is in the root of the sage repo
    add_local_sage_path(sage_libpath)
    from sage import build_sage_pyext
    logger.info("Building SAGE python extension")
    build_sage_pyext(use_from_mcmc=True, verbose=verbose)
    logger.info("Finished building SAGE python extension")
    import _sage_cffi
    logger.info("Finished importing SAGE python extension")
    return


def set_all_simulations():
    allsims_dict = {
                    'Millennium': {'Boxsize':500, #Mpc/h
                                   'Hubble_h':0.73,
                                   'Omega':0.25,
                                   'OmegaLambda':0.75,
                                   'OmegaBaryon':0.045,
                                   'Partmass':8.6e8, #Msun/h
                                   'Npart': 2160**3,
                                   'SimulationDir':"/fred/oz004/msinha/simulations/Millennium",
                                   'TreeType':"lhalo_binary",
                                   'TreeName':"trees_063",
                                   'FirstFile': 0,
                                   'LastFile': 511,
                                   'NumSimulationTreeFiles':512,
                                   'FileWithSnapList':'/fred/oz004/msinha/simulations/Millennium/millennium.alist',
                                   'LastSnapShotNr': 63,
                                   'Softening':5.0*1e-3, #kpc/h->Mpc/h
                                   'scale_factors':[0.0078125,0.012346,0.019608,0.032258,0.047811,0.051965,0.056419,0.061188,0.066287,0.071732,0.077540,0.083725,0.090306,0.097296,0.104713,0.112572,0.120887,0.129675,0.138950,0.148724,0.159012,0.169824,0.181174,0.193070,0.205521,0.218536,0.232121,0.246280,0.261016,0.276330,0.292223,0.308691,0.325730,0.343332,0.361489,0.380189,0.399419,0.419161,0.439397,0.460105,0.481261,0.502839,0.524807,0.547136,0.569789,0.592730,0.615919,0.639314,0.662870,0.686541,0.710278,0.734031,0.757746,0.781371,0.804849,0.828124,0.851138,0.873833,0.896151,0.918031,0.939414,0.960243,0.980457,1.000000],
                        },
                    'Mini-Millennium': {'Boxsize': 62.5, #Mpc/h
                                        'Hubble_h':0.73,
                                        'Omega':0.25,
                                        'OmegaLambda':0.75,
                                        'OmegaBaryon':0.045,
                                        'Partmass':8.6e8, #Msun/h
                                        'Npart': 270**3,
                                        'Softening':5.0*1e-3, #kpc/h->Mpc/h
                                        'SimulationDir':"/fred/oz004/msinha/simulations/mini-Millennium/",
                                        'TreeType':"lhalo_binary",
                                        'TreeName':"trees_063",
                                        'FirstFile': 0,
                                        'LastFile': 7,
                                        'NumSimulationTreeFiles':8,
                                        'FileWithSnapList':'/fred/oz004/msinha/simulations/mini-Millennium/millennium.a_list',
                                        'LastSnapShotNr': 63,
                                        'scale_factors':[0.0078125,0.012346,0.019608,0.032258,0.047811,0.051965,0.056419,0.061188,0.066287,0.071732,0.077540,0.083725,0.090306,0.097296,0.104713,0.112572,0.120887,0.129675,0.138950,0.148724,0.159012,0.169824,0.181174,0.193070,0.205521,0.218536,0.232121,0.246280,0.261016,0.276330,0.292223,0.308691,0.325730,0.343332,0.361489,0.380189,0.399419,0.419161,0.439397,0.460105,0.481261,0.502839,0.524807,0.547136,0.569789,0.592730,0.615919,0.639314,0.662870,0.686541,0.710278,0.734031,0.757746,0.781371,0.804849,0.828124,0.851138,0.873833,0.896151,0.918031,0.939414,0.960243,0.980457,1.000000],
                        },
                    'Uchuu': {'Boxsize': 2000, #Mpc/h
                              'Hubble_h':0.6774,
                              'Omega':0.3089,
                              'OmegaLambda':0.6911,
                              'OmegaBaryon':0.0486,
                              'Partmass':3.27e8, #Msun/h
                              'Npart': 12800**3, #particles per side
                              'Softening':4.27*1e-3, #kpc/h->Mpc/h
                              'mergertree_dir':"/fred/oz214/simulations/uchuu/U2000/mergertree/",
                              'mergertree_format':"consistent-trees-hdf5",
                              'mergertree_fname':"mergertree_info.h5",
                              'FirstFile': 0,
                              'LastFile': 1999,
                              'NumSimulationTreeFiles':2000,
                              'scale_factors': [0.06686,0.07306,0.07994,0.08740,0.09549,0.10430,0.11410,0.12460,0.13610,0.14869,0.16247,0.17760,0.18980,0.20281,0.21694,0.23174,0.24218,0.25311,0.26458,0.27658,0.28916,0.30216,0.31590,0.33016,0.34519,0.36084,0.37723,0.39435,0.41225,0.43083,0.45040,0.47079,0.49215,0.51433,0.53776,0.56206,0.58752,0.61420,0.64189,0.67117,0.70146,0.73310,0.76647,0.80122,0.83735,0.87529,0.91501,0.95645,0.97782,0.99998],
                        },
                    'Micro-Uchuu': {'Boxsize': 100.0, #Mpc/h
                                  'Hubble_h':0.6774,
                                  'Omega':0.3089,
                                  'OmegaLambda':0.6911,
                                  'OmegaBaryon':0.0486,
                                  'Partmass':3.27e8, #Msun/h
                                  'Npart': 640**3, #particles per side
                                  'Softening':4.27*1e-3, #kpc/h->Mpc/h
                                  'mergertree_dir':"/fred/oz214/simulations/uchuu/microuchuu/mergertree/",
                                  'mergertree_format':"consistent-trees-hdf5",
                                  'mergertree_fname':"mergertree_info.h5",
                                  'FirstFile': 0,
                                  'LastFile': 0,
                                  'NumSimulationTreeFiles':1,
                                  'scale_factors': [6.696400e-02,7.313100e-02,7.994900e-02,8.736300e-02,9.541300e-02,1.043250e-01,1.139670e-01,1.244960e-01,1.361580e-01,1.487720e-01,1.625190e-01,1.776490e-01,1.898560e-01,2.028780e-01,2.168980e-01,2.317990e-01,2.422230e-01,2.530860e-01,2.646840e-01,2.765360e-01,2.891750e-01,3.021400e-01,3.158640e-01,3.302550e-01,3.452800e-01,3.608860e-01,3.771820e-01,3.943330e-01,4.121550e-01,4.307270e-01,4.503890e-01,4.707730e-01,4.921300e-01,5.143560e-01,5.377130e-01,5.620990e-01,5.875370e-01,6.141530e-01,6.420350e-01,6.711510e-01,7.015750e-01,7.331350e-01,7.664560e-01,8.010340e-01,8.375390e-01,8.752300e-01,9.149410e-01,9.564120e-01,9.778480e-01,9.998870e-01],
                        },
                    'Mini-Uchuu': {'Boxsize': 400.0, #Mpc/h
                                  'Hubble_h':0.6774,
                                  'Omega':0.3089,
                                  'OmegaLambda':0.6911,
                                  'OmegaBaryon':0.0486,
                                  'FirstFile': 0,
                                  'LastFile': 0,
                                  'Partmass':3.27e8, #Msun/h
                                  'Npart': 2560**3, #particles per side
                                  'Softening':4.27*1e-3, #kpc/h->Mpc/h
                                  'mergertree_dir':"/fred/oz214/simulations/uchuu/miniuchuu/mergertree/",
                                  'mergertree_format':"consistent-trees-ascii",
                                  'mergertree_fname':"MiniUchuu_0_0_0.trees",
                                  'scale_factors': [6.696400e-02,7.313100e-02,7.994900e-02,8.736300e-02,9.541300e-02,1.043250e-01,1.139670e-01,1.244960e-01,1.361580e-01,1.487720e-01,1.625190e-01,1.776490e-01,1.898560e-01,2.028780e-01,2.168980e-01,2.317990e-01,2.422230e-01,2.530860e-01,2.646840e-01,2.765360e-01,2.891750e-01,3.021400e-01,3.158640e-01,3.302550e-01,3.452800e-01,3.608860e-01,3.771820e-01,3.943330e-01,4.121550e-01,4.307270e-01,4.503890e-01,4.707730e-01,4.921300e-01,5.143560e-01,5.377130e-01,5.620990e-01,5.875370e-01,6.141530e-01,6.420350e-01,6.711510e-01,7.015750e-01,7.331350e-01,7.664560e-01,8.010340e-01,8.375390e-01,8.752300e-01,9.149410e-01,9.564120e-01,9.778480e-01,9.998870e-0],
                        },
        }

    for k,v in allsims_dict.items():
        a = v['scale_factors']
        v['redshifts'] = [(1.0/aa - 1.0) for aa in a]
        allsims_dict[k] = v

    return allsims_dict


# def _return_unique_redshifts(redshifts):
#     redshifts = np.atleast_1d(redshifts)
#     assert redshifts.ndim == 1, f"Target redshifts must be a 1D array. Instead found shape = {redshifts.shape}"
#     uniq_redshifts = np.unique(redshifts) # unique redshifts is also sorted now
#     if uniq_redshifts.shape != redshifts.shape:
#         logger.warn("There are repeated redshifts in the list of target redshifts. Only using unique redshifts")
#     return uniq_redshifts

def _get_obsdata_from_name(obs, catalogname, return_errors=True):
    idx = [i for (i, name) in enumerate(obs.target_observation.index) if catalogname==name]
    if len(idx) != 1:
        msg = f"Error: Catalog name = {catalogname} does not exist at the requested redshift\n"
        msg += f"Available catalogs are: {obs.target_observation.index}"
        raise ValueError(msg)

    idx = idx[0]
    data = obs.target_observation['Data'][idx]
    datatype = obs.target_observation['DataType'][idx]
    if datatype != 'data':
        msg = f"Error: catalog = {catalogname} does not seem to contain observational data (catalog type = {datatype}))"
        raise ValueError(msg)

    with np.errstate(divide='ignore'):
        data[:,1:] = np.log10(data[:,1:])

    # astrodatapy returns number density data in the following format:
    # 'mass  nden  nden+err nden-err' (where the nden+/err columns might be swapped)
    # Errors as returned as ULLimit, i.e., upper and lower 1-sigma limits
    mass = data[:, 0]
    y = data[:, 1]
    yhi = data[:, 2]
    ylow = data[:, 3]
    if return_errors:
        obs_numdenerr = np.abs(yhi - ylow)*0.5
        obs_numden = (yhi + ylow)*0.5
        return mass, obs_numden, obs_numdenerr, datatype

    # xlabel  = r"$\log_{10}[M_*/{\rm M_{\odot}}]$"
    # ylabel  = r"$\log_{10}[\rm \phi/Mpc^{-3} dex^{-1}]$"

    return mass, y, yhi, ylow, datatype


@functools.cache
def get_observational_data_and_errors(redshift, h, feature, data_identifier='Baldry+2012', IMF_out='Salpeter',
                                      return_errors=True, quiet=True):
    from astrodatapy.number_density import number_density
    # logger.info(f"Getting number density for z = {redshift}")
    obs = number_density(feature=feature, z_target=redshift, h=h, IMF_out=IMF_out, quiet=quiet)
    return _get_obsdata_from_name(obs, data_identifier, return_errors=return_errors)


@functools.cache
def read_sage_parameterfile(param_file):
    import numpy as np

    # This will break for the standard template because of the
    # '->' used in the parameter file after NumOutputs to denote
    # the snapshots at which the output is written
    # In the template file, that line needs to be fixed by
    # using a "%" instead of "->". However, it does not make
    # sense to output unnecessary snapshots, therefore, we
    # first write out the modified parameter file based on the
    # parsed dictionary + modified parameter values. After this
    # modified parameter file is written out, we can open that
    # text file, and add the specific snapshot numbers to be output
    # into the parameter file. - MS: 18th Oct, 2023
    try:
        sage_params = np.genfromtxt(param_file, dtype=(str, str),
                                       comments='%', autostrip=True)
    except ValueError:
        print(f"ValueError encountered while reading {param_file} - probably from an uncomment snapshot number line. Fixing this now...")
        with open(param_file, 'r') as f:
            alltext = f.read()

        print("Replacing '->' with '%->' in the parameter file")
        # prepend '->' with '%'
        alltext = alltext.replace('->', '%->')
        with open(param_file, 'w') as f:
            f.write(alltext)

        sage_params = np.genfromtxt(param_file, dtype=(str, str),
                                       comments='%', autostrip=True)

    sage_params = dict(sage_params)
    return sage_params


def write_sage_parameter_file(input_dict, outputfile, extra_params_dict):

    # Determine what the maximum length of a key is
    max_len = max(map(len, input_dict))

    # Create a new input_data list
    input_data = [' '.join([key.ljust(max_len, ' '),
                            str(value)])
                    for key, value in input_dict.items()]

    # Convert input_data to a single string
    input_data = '\n'.join(input_data)

    # Create the output parameter file
    with open(outputfile, 'w') as f:
        # Write input_data to this file
        f.write(input_data)

    snapshot = extra_params_dict['snapshot']
    snapshot_string = str(snapshot)

    import fileinput
    import os
    for line in fileinput.FileInput(outputfile, inplace=True):
        if line.startswith('NumOutputs'):
            line += "-> " + snapshot_string + os.linesep
        print(line, end="")

    return




def set_sage_params(**kwargs):
    wanted_redshift = np.float64(kwargs['wanted_redshift'])

    which_sim = kwargs['which_sim']
    allsims = set_all_simulations()
    if which_sim not in allsims.keys():
        raise ValueError(f"which_sim = {which_sim} not in {allsims.keys()}")

    sim_dict = allsims[which_sim]
    firstfile = kwargs.get('firstfile', sim_dict['FirstFile'])
    lastfile  = kwargs.get('lastfile', sim_dict['LastFile'])
    if firstfile < sim_dict['FirstFile'] or lastfile > sim_dict['LastFile']:
        msg = f"For sim = {which_sim}, the parameters firstfile = {firstfile} or lastfile = {lastfile} must be in range of [{sim_dict['FirstFile']}, {sim_dict['LastFile']}]"
        raise ValueError(msg)

    # print(f"sim_dict.keys = {sim_dict.keys()}")
    # print(f"sim_dict['redshifts'] = {sim_dict['redshifts']}")
    redshifts = np.array(sim_dict['redshifts'])

    ## find closest redshift to the wanted redshifts
    # print(f"redshifts = {redshifts}  wanted_redshift = {wanted_redshift}")
    # print(f"dtypes = {redshifts.dtype} {wanted_redshift.dtype} diff = np.abs(redshifts - wanted_redshift) = {np.abs(redshifts - wanted_redshift) }")
    snapnum = np.argmin(np.abs(redshifts - wanted_redshift))
    # logger.debug(f"Target redshift = {wanted_redshift} corresponding snapshot = {snapnum}")

    sage_params_dict = read_sage_parameterfile(kwargs['sage_template_param_fname'])
    sage_params_dict['FirstFile'] = firstfile
    sage_params_dict['LastFile'] = lastfile
    sage_params_dict['NumOutputs'] = 1
    extra_params_dict = {'snapshot':snapnum}
    for k, v in sim_dict.items():
        if k in sage_params_dict.keys():
            sage_params_dict[k] = v

    # print(f"sage_params[treetype] = '{sage_params_dict['TreeType']}'")
    # logger.info("Finished setting SAGE parameters")
    return sage_params_dict, extra_params_dict


def read_and_bin_sage_output(sage_output_file, log_bins, sage_params_dict, *, column=None, snapshot=63):
    import h5py
    import numpy as np

    if not column:
        column = 'StellarMass'

    massunits = float(sage_params_dict['UnitMass_in_g'])
    solar_mass = 1.989e33
    massunits_in_msun = massunits/solar_mass
    h = float(sage_params_dict['Hubble_h'])
    # print(f"Mass conversion factor -> Msun = {massunits_in_msun}")

    fullhist = np.zeros((len(log_bins)-1), dtype=np.int64)
    with h5py.File(sage_output_file, 'r') as hf:
        numcores = hf['Header/Misc'].attrs['num_cores']
        for icore in range(numcores):
            field = f"Core_{icore}/Snap_{snapshot}/{column}"
            shape = hf[field].shape
            sm = np.empty(shape, dtype=hf[field].dtype)
            hf[field].read_direct(sm)

            sm *= massunits_in_msun # convert arbitrary mass units to Msun/h
            sm /= h # convert to Msun

            with np.errstate(divide='ignore'):
                log_sm = np.log10(sm)

            hist, xx = np.histogram(log_sm, bins=log_bins)
            np.testing.assert_array_equal(xx, log_bins)
            fullhist += hist

    return fullhist


import emcee
import numpy as np
from schwimmbad import MPIPool
import time
import sys

def get_rank_from_mpx():
    import multiprocessing
    print("HERE: multiprocessing.current_process()._identity = ", multiprocessing.current_process()._identity)
    return multiprocessing.current_process()._identity[0] - 1

def log_priors(theta, **kwargs):
    sage_params_to_vary = kwargs['sage_params_to_vary']
    prior_limits = {'SfrEfficiency':[0.01, 0.2],
                    'ReIncorporationFactor': [0.0, 1.0],
                    'FeedbackReheatingEpsilon': [0.0, 10.0],
                    'RadioModeEfficiency': [0.0, 0.5],
    }

    for i, param in enumerate(sage_params_to_vary):
        if param not in prior_limits.keys():
            msg = f"Error: parameter = {param} is not in the list of parameters for setting the priors"
            raise ValueError(msg)

        limits = prior_limits[param]
        if theta[i] < limits[0] or theta[i] > limits[1]:
            return -np.inf

    return 0.0

@functools.cache # Caching is important -> we keep reusing the same directory
def create_workdir(outputdir):
    return mkdtemp(dir=outputdir)

def log_probability(theta, *args, **kwargs):
    lp = log_priors(theta, **kwargs)
    if not np.isfinite(lp):
        return -np.inf #, *kwargs['default_blobs']
    ll = log_likelihood(theta, *args, **kwargs)
    return lp + ll #, *blobs


def log_likelihood(theta, *args, **kwargs):
    from _sage_cffi import ffi, lib
    # start = time.time()

    sage_rank = 0
    sage_ntasks = 1

    sage_params_dict, extra_params_dict = set_sage_params(**kwargs)
    sage_params_to_vary = kwargs['sage_params_to_vary']
    for p, v in zip(sage_params_to_vary, theta):
        # print(f"For param = {p}, replacing {sage_params_dict[p]} with {v}")
        sage_params_dict[p] = v

    outputdir = kwargs['outputdir']
    workdir = create_workdir(outputdir)
    sage_params_dict['OutputDir'] = workdir
    paramfile = f"{workdir}/sage_params_{sage_rank}.par"
    write_sage_parameter_file(sage_params_dict, paramfile, extra_params_dict)

    # Get params ready for cffi to run sage
    params_struct = ffi.new("void **")
    fname = ffi.new("char []", paramfile.encode())

    # logger.info(f"Running sage ...")
    # sage_start = time.time()
    lib.run_sage(sage_rank, sage_ntasks, fname, params_struct)
    lib.finalize_sage(params_struct[0])
    # sage_end = time.time()
    # logger.info(f"Running sage ...done. Time taken = {sage_end-sage_start:0.1f} sec")

    logmass = kwargs['logmass']
    obs_numden = kwargs['obs_numden']
    obs_numdenerr = kwargs['obs_numdenerr']
    bin_edges = kwargs['bin_edges']
    bin_widths = kwargs['bin_widths']

    sage_output_file = f"{workdir}/{sage_params_dict['FileNameGalaxies']}.hdf5"
    sim_counts = read_and_bin_sage_output(sage_output_file, bin_edges, sage_params_dict, snapshot=extra_params_dict['snapshot'])
    vol = float(sage_params_dict['BoxSize'])**3
    sim_numden = sim_counts*sage_params_dict['Hubble_h']**3/(vol*bin_widths)
    with np.errstate(divide='ignore'):
        log_sim_numden = np.log10(sim_numden)

    chi = (obs_numden - log_sim_numden)/obs_numdenerr
    stellarmass_limits = kwargs['catalog_xlimits']['StellarMass']
    xx = np.where((logmass >= stellarmass_limits[0]) & (logmass <= stellarmass_limits[1]))
    if not xx:
        logger.info("Warning: masses do not overlap with the observational data. Returning -np.inf")
        return -np.inf #, kwargs['default_blobs']

    chisqr = np.sum(chi[xx]*chi[xx])

    # xx = [(a, b, c, ch) for a, b, c, ch in zip(sim_numden, obs_numden, obs_numdenerr, chi)]
    # for x in zip(logmass, log_sim_numden, obs_numden, obs_numdenerr, chi):
    #     print(f"{x[0]:0.3e} {x[1]:0.2e} {x[2]:0.2e} {x[3]:0.2e} {x[4]:0.2e}")

    # shutil.rmtree(dirname)

    # blobs = kwargs['default_blobs']

    # end = time.time()
    # blobs[0] = end - start
    # blobs[1] = chisqr
    # blobs[2:] = sim_numden
    # blobs = [end-start, chisqr, sim_numden]
    # print(f"[On rank = {rank}] chisqr = {chisqr} chi = {chi} theta = {theta}.\n"\
    #       f"Time taken = {end - start:0.3e} sage runtime = {sage_end - sage_start:0.3e} seconds")
    # end = time.time()
    # logger.info(f"Done with likelihood. sage time = {sage_end - sage_start:0.3e} sec. Total time = {end - start:0.3e} sec")
    return -0.5*chisqr #, blobs


def run_emcee(sage_template_param_fname, sage_libpath=None, **kwargs):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    kwargs['sage_template_param_fname'] = sage_template_param_fname

    sage_params_to_vary = kwargs.get('sage_params_to_vary', None)
    if not sage_params_to_vary:
        sage_params_to_vary = ['SfrEfficiency', 'ReIncorporationFactor']

    seed = kwargs.get('seed', 213589749325792)

    rank = kwargs['rank']
    ntasks = kwargs['ntasks']

    if rank == 0:
        logger.info(f"[On {rank =}]: Compiling SAGE python extension and importing it")
        compile_sage_pythonext(sage_libpath=sage_libpath, verbose=verbose)

    kwargs.pop('sage_libpath', None)
    if ntasks > 1:
        logger.info(f"[On {rank = }]: Waiting at barrier for rank=0 to be done with compiling")
        comm.Barrier() # Make sure all MPI ranks execute this Barrier statement (otherwise code will hang)

    outputdir = kwargs.get('outputdir', '.')
    nwalkers = kwargs.get('nwalkers', 1000)
    ndim = len(sage_params_to_vary)

    logger.info(f"Starting run on rank = {rank} (out of ntasks = {ntasks})")
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            workdir = create_workdir(outputdir)
            shutil.rmtree(workdir)
            sys.exit(0)

        logger.info(f"On main {rank = }: kwargs = {kwargs}")
        template_params_dict = read_sage_parameterfile(sage_template_param_fname)

        filename_suffix = '_'.join(sage_params_to_vary)
        filename_suffix += f"_z_{kwargs['wanted_redshift']}_nwalkers_{nwalkers}"
        emcee_output_filename = f"{outputdir}/sage_emcee_{filename_suffix}.hdf5"

        logmass, obs_numden, obs_numdenerr, datatype = get_observational_data_and_errors(float(kwargs['wanted_redshift']),
                                                                               float(template_params_dict['Hubble_h']),
                                                                               data_identifier=kwargs['catalog'],
                                                                               IMF_out=kwargs['IMF'],
                                                                               feature=kwargs['catalogtype'])

        # Masses are derived from fluxes which have a h^2 dependence
        # however, only one h has been accounted for by astrodatapy
        # therefore, we need to divide by h to get the correct units
        logmass -= np.log10(float(template_params_dict['Hubble_h']))

        xx = np.argsort(logmass)
        logmass = logmass[xx]
        obs_numden = obs_numden[xx]
        obs_numdenerr = obs_numdenerr[xx]

        kwargs['logmass'] = logmass
        kwargs['obs_numden'] = obs_numden
        kwargs['obs_numdenerr'] = obs_numdenerr

        bin_widths = np.empty_like(logmass)
        if kwargs['catalog'] == 'Baldry+2012':
            bin_widths[2:] = 0.2
            bin_widths[0:2] = 0.5
        else:
            diff = np.diff(logmass)
            bin_widths[1:] = diff
            bin_widths[0] = diff[0]

        bin_edges = np.empty(len(logmass) + 1, dtype=np.float64)
        bin_edges[0] = logmass[0] - 0.5*bin_widths[0]
        for i in range(len(logmass)):
            bin_edges[i+1] = logmass[i] + 0.5*bin_widths[i]
        kwargs['bin_edges'] = bin_edges
        kwargs['bin_widths'] = bin_widths

        # blobs_dtype = np.dtype([('time', np.float64), ('chisqr', np.float64), ('sim_gsmf', np.float64, (len(bin_widths)))])
        # default_blobs = [-np.inf]*(len(bin_widths) + 2)
        # default_blobs[:] = -np.inf
        # print(f"len(default_blobs) = {len(default_blobs)}")
        # kwargs['default_blobs'] = default_blobs

        backend = emcee.backends.HDFBackend(emcee_output_filename, name=filename_suffix)

        rng = np.random.default_rng(seed)
        loc = np.array([template_params_dict[k] for k in sage_params_to_vary], dtype=np.float64)
        scale = 0.1*loc
        initial = rng.normal(loc=loc, scale=scale, size=(nwalkers, ndim))

        resume_iter = False
        try:
            if backend.iteration > 0:
                resume_iter = True
        except FileNotFoundError:
            pass

        nburn_in = kwargs.get('nburn_in', 200)
        nsteps = 1000*nburn_in

        # We'll track how the average autocorrelation time estimate changes
        index = 0
        autocorr = np.empty(nsteps)

        # This will be useful to testing convergence
        old_tau = np.inf

        # create the sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool, kwargs=kwargs, backend=backend)

        # state = initial
        # if do_burnin:
        #     logger.info(f"[On rank = {rank}]: Starting burn-in phase")
        #     burn_start = time.time()
        #     state = sampler.run_mcmc(initial, nburn_in, progress=True)
        #     burn_end = time.time()
        #     sampler.reset()
        #     logger.info(f"Done with burn-in phase with {nburn_in} steps. Time taken = {burn_end-burn_start:0.3e}")

        prod_start = time.time()

        if resume_iter:
            initial = sampler.get_last_sample()
            logger.info(f"Got last sample from reader. {type(initial) = }")
            logger.info(f"Resuming production phase. {initial = }")
        else:
            logger.info(f"Starting production phase. {initial = }")

        for sample in sampler.sample(initial, iterations=nsteps, store=True, progress=True):
            print(f"[{rank = }]: iteration = {sampler.iteration}", file=sys.stderr)
            if sampler.iteration % 100:
                continue

            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            tau = sampler.get_autocorr_time(tol=0)
            logger.info(f"iteration = {sampler.iteration} tau = {tau} steps")
            autocorr[index] = np.mean(tau)
            index += 1

            # Check convergence
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau

        prod_end = time.time()
        logger.info(f"Done with production phase with {sampler.iteration} steps. Time taken = {prod_end - prod_start:0.2e} sec")
        logger.info(f"Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
        logger.info(f"Mean autocorrelation time: {0:.3f} steps".format(np.mean(sampler.get_autocorr_time())))


if __name__ == "__main__":
    sage_params_to_vary = ['SfrEfficiency', 'ReIncorporationFactor', 'FeedbackReheatingEpsilon', 'RadioModeEfficiency']
    nwalkers = 1000
    seed = 2783946238
    verbose = True

    sage_template_param_fname = "./mini-millennium.par"  # comes with the repo and contains the default params (the ones not being varied)
    sage_libpath = None # set to the root directory containing source code and sage.py (usually '../sage-model')

    # the hdf5 file containing the emcee ouput will be here. Temp workdirs are created *per* MPI task
    # within this outputdir and (should be?) are deleted once the job completes. If the job errors out
    # then you will have to manually delete the tmp* directories within the outputdir. (Just be careful
    # that you might accidentally delete tmp directories another running MCMC job outputting to this same
    # outputdir - should be obvious how I figured that one out: MS 23rd Nov, 2023)
    outputdir = "/fred/oz004/msinha/sage_mcmc_output/"


    ## simulation specific details
    ## all files are used by default (if firstfile and lastfile are set to None)
    which_sim = "Mini-Millennium"
    firstfile = 0
    lastfile = 7

    ## observational data specifications (passed to astrodatapy)
    catalog, target_redshift = 'Baldry+2012', 0.0
    # catalog, target_redshift = 'Stefanon+2021', 6.0
    # catalog, target_redshift = 'Perez-Gonzalez+2008', 1.0
    # catalog, target_redshift = 'Huertas-Company+2016', 2.0
    # catalog, target_redshift = 'Qin+2017_Tiamat125_HR', 2.0
    # catalog, target_redshift = 'Grazian+2015', 4.0

    catalogtype = 'GSMF'
    IMF = 'Chabrier'

    # What are the limits for meaningful range of X-axis (i.e., stellar mass)
    # considering both the target redshift and the simulation being used
    catalog_xlimits = {'StellarMass': [7.0, 12.0]}

    ## You should not need to modify anything below
    rank = 0
    ntasks = 1
    try:
        from mpi4py import MPI
        pool_type = 'mpi4py'
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        ntasks = comm.Get_size()
        logger.info(f"Running on with MPI on {rank = } (out of {ntasks} tasks)")
    except ImportError:
        import multiprocessing
        ntasks = multiprocessing.cpu_count()
        pool_type = 'multiprocessing'

    # logger.info(f"After try-mpi: {rank = }. Created {workdir = }")
    logger.info(f"[On rank={rank}]: (out of {ntasks} tasks)]: Parallelism type = {pool_type}.")
    run_emcee(sage_template_param_fname, sage_libpath=sage_libpath, verbpse=verbose, sage_params_to_vary=sage_params_to_vary,
              wanted_redshift=target_redshift, catalog=catalog, catalogtype=catalogtype, IMF=IMF, catalog_xlimits=catalog_xlimits,
              seed=seed, outputdir=outputdir,
              which_sim=which_sim, firstfile=firstfile, lastfile=lastfile,
              rank=rank, ntasks=ntasks, pool_type=pool_type)