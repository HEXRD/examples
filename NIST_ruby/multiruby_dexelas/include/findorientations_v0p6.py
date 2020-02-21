#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 19:04:10 2017

@author: bernier2
"""
from __future__ import print_function

import os
# =============================================================================
# To disable numba when running, set env var to 0
# os.environ['HEXRD_USE_NUMBA'] = '0'
# =============================================================================

import glob

import multiprocessing

import numpy as np

from scipy import ndimage

import timeit

try:
    import dill as cpl
except(ImportError):
    import cPickle as cpl

import yaml

from hexrd import constants as const
from hexrd import config
from hexrd import imageseries
from hexrd.imageseries.omega import OmegaImageSeries
from hexrd import instrument
from hexrd import matrixutil as mutil
from hexrd.findorientations import \
    generate_orientation_fibers, \
    run_cluster
from hexrd.xrd import transforms_CAPI as xfcapi
from hexrd.xrd import indexer
from matplotlib import pyplot as plt
from hexrd.xrd.xrdutil import EtaOmeMaps


# plane data
def load_pdata(cpkl, key):
    with file(cpkl, "r") as matf:
        mat_list = cpl.load(matf)
    return dict(zip([i.name for i in mat_list], mat_list))[key].planeData


# images
def load_images(yml):
    return imageseries.open(yml, format="frame-cache", style="npz")


# instrument
def load_instrument(yml):
    with file(yml, 'r') as f:
        icfg = yaml.load(f, Loader=yaml.SafeLoader)
    return instrument.HEDMInstrument(instrument_config=icfg)


# %%
# =============================================================================
# START USER INPUT
# =============================================================================

cfg_filename = 'test_config_v0p6.yml'
samp_name = 'mruby-0129'
scan_number = 4

image_dir = '../imageseries'
data_dir = os.getcwd()

fc_stem = "%s_%06d_%%s_*-cachefile.npz" % (samp_name, scan_number)

make_max_frames = False
use_direct_search = False

# maps options
clobber_maps = True
show_maps = False

# =============================================================================
# END USER INPUT
# =============================================================================

# %%
cfg = config.open(cfg_filename)[0]

analysis_id = '%s_%s' % (
    cfg.analysis_name.strip().replace(' ', '-'),
    cfg.material.active.strip().replace(' ', '-'),
    )

active_hkls = cfg.find_orientations.orientation_maps.active_hkls
if active_hkls == 'all':
    active_hkls = None

max_tth = cfg.fit_grains.tth_max
if max_tth:
    if type(cfg.fit_grains.tth_max) != bool:
        max_tth = np.degrees(float(max_tth))
else:
    max_tth = None

# load plane data
plane_data = load_pdata(cfg.material.definitions, cfg.material.active)
plane_data.tThMax = max_tth

# load instrument
instr = load_instrument(cfg.instrument.parameters)
det_keys = instr.detectors.keys()

# !!! panel buffer setting is global and assumes same type of panel!
for det_key in det_keys:
    instr.detectors[det_key].panel_buffer = \
        np.array(cfg.fit_grains.panel_buffer)


# grab eta ranges
eta_ranges = cfg.find_orientations.eta.range

# for indexing
build_map_threshold = cfg.find_orientations.orientation_maps.threshold

on_map_threshold = cfg.find_orientations.threshold
fiber_ndiv = cfg.find_orientations.seed_search.fiber_ndiv
fiber_seeds = cfg.find_orientations.seed_search.hkl_seeds

tth_tol = np.degrees(plane_data.tThWidth)
eta_tol = cfg.find_orientations.eta.tolerance
ome_tol = cfg.find_orientations.omega.tolerance
# omega period...
# ???: necessary?
ome_period = np.radians(cfg.find_orientations.omega.period)

npdiv = cfg.fit_grains.npdiv

compl_thresh = cfg.find_orientations.clustering.completeness
cl_radius = cfg.find_orientations.clustering.radius

# %%

imsd = dict.fromkeys(det_keys)
for det_key in det_keys:
    fc_file = sorted(
        glob.glob(
            os.path.join(
                image_dir,
                fc_stem % det_key
            )
        )
    )
    if len(fc_file) != 1:
        raise RuntimeError('cache file not found, or multiple found')
    else:
        ims = load_images(fc_file[0])
        oims = OmegaImageSeries(ims)
        imsd[det_key] = oims
        # !!! we assume all detector ims have the same ome ranges, so any will do!
        ome_ranges = [
                (np.radians([i['ostart'], i['ostop']]))
                for i in oims.omegawedges.wedges
            ]

if make_max_frames:
    print("Making requested max frame...")
    max_frames_output_name = os.path.join(
        data_dir,
        "%s_%d-maxframes.hdf5" % (samp_name, scan_number)
    )

    if os.path.exists(max_frames_output_name):
        os.remove(max_frames_output_name)

    max_frames = dict.fromkeys(det_keys)
    for det_key in det_keys:
        max_frames[det_key] = imageseries.stats.max(imsd[det_key])

    ims_out = imageseries.open(
            None, 'array',
            data=np.array([max_frames[i] for i in max_frames]),
            meta={'panels': max_frames.keys()}
        )
    imageseries.write(
            ims_out, max_frames_output_name,
            'hdf5', path='/imageseries'
        )
# %%

maps_fname = analysis_id + "_maps.npz"
if os.path.exists(maps_fname) and not clobber_maps:
    eta_ome = EtaOmeMaps(maps_fname)
else:
    map_hkls = plane_data.hkls.T[active_hkls]
    hklseedstr = ', '.join(
        [str(map_hkls[i]) for i in active_hkls]
        )
    print("INFO:\tbuilding eta_ome maps using hkls: %s" % hklseedstr)

    start = timeit.default_timer()

    # make eta_ome maps
    eta_ome = instrument.GenerateEtaOmeMaps(
        imsd, instr, plane_data,
        active_hkls=active_hkls, threshold=build_map_threshold,
        ome_period=cfg.find_orientations.omega.period)

    print("INFO:\t\t...took %f seconds" % (timeit.default_timer() - start))

    # save them
    eta_ome.save(maps_fname)


# %%
# =============================================================================
# OPTINAL PLOTTING
# =============================================================================
if show_maps:
    cmap = plt.cm.hot
    cmap.set_under('b')

    for ihkl, this_map in enumerate(eta_ome.dataStore):
        fig, ax = plt.subplots()
        this_map_f = -ndimage.filters.gaussian_laplace(
            this_map, 1.0,
        )
        ax.imshow(this_map_f, interpolation='nearest',
                  vmin=on_map_threshold, vmax=None, cmap=cmap)
        labels, num_spots = ndimage.label(
            this_map_f > on_map_threshold,
            ndimage.generate_binary_structure(2, 1)
        )
        coms = np.atleast_2d(
            ndimage.center_of_mass(
                this_map,
                labels=labels,
                index=range(1, num_spots+1)
            )
        )
        if len(coms) > 1:
            ax.plot(coms[:, 1], coms[:, 0], 'm+', ms=12)
        ax.axis('tight')
        hklseedstr = str(eta_ome.planeData.hkls.T[eta_ome.iHKLList[ihkl]])
        fig.suptitle(r'\{%s\}' % hklseedstr)

# %%
# =============================================================================
# SEARCH SPACE GENERATION
# =============================================================================
ncpus = cfg.multiprocessing

print("INFO:\tgenerating search quaternion list using %d processes" % ncpus)
start = timeit.default_timer()

qfib = generate_orientation_fibers(
    eta_ome, instr.chi, on_map_threshold,
    fiber_seeds, fiber_ndiv,
    ncpus=ncpus)
print("INFO:\t\t...took %f seconds" % (timeit.default_timer() - start))
print("INFO: will test %d quaternions using %d processes"
      % (qfib.shape[1], ncpus))

# %%
# =============================================================================
# ORIENTATION SCORING
# =============================================================================

if use_direct_search:
    def test_orientation_FF_init(params):
        global paramMP
        paramMP = params

    def test_orientation_FF_reduced(quat):
        """
        input parameters are [
        plane_data, instrument, imgser_dict,
        tth_tol, eta_tol, ome_tol, npdiv, threshold
        ]
        """
        plane_data = paramMP['plane_data']
        instrument = paramMP['instrument']
        imgser_dict = paramMP['imgser_dict']
        tth_tol = paramMP['tth_tol']
        eta_tol = paramMP['eta_tol']
        ome_tol = paramMP['ome_tol']
        npdiv = paramMP['npdiv']
        threshold = paramMP['threshold']

        phi = 2*np.arccos(quat[0])
        n = xfcapi.unitRowVector(quat[1:])
        grain_params = np.hstack([
            phi*n, const.zeros_3, const.identity_6x1,
        ])

        compl, scrap = instrument.pull_spots(
            plane_data, grain_params, imgser_dict,
            tth_tol=tth_tol, eta_tol=eta_tol, ome_tol=ome_tol,
            npdiv=npdiv, threshold=threshold,
            eta_ranges=np.radians(cfg.find_orientations.eta.range),
            ome_period=np.radians(cfg.find_orientations.omega.period),
            check_only=True)

        return sum(compl)/float(len(compl))

    params = dict(
            plane_data=plane_data,
            instrument=instr,
            imgser_dict=imsd,
            tth_tol=tth_tol,
            eta_tol=eta_tol,
            ome_tol=ome_tol,
            npdiv=npdiv,
            threshold=cfg.fit_grains.threshold)

    print("INFO:\tusing direct seach")
    pool = multiprocessing.Pool(ncpus, test_orientation_FF_init, (params, ))
    completeness = pool.map(test_orientation_FF_reduced, qfib.T)
    pool.close()
else:
    print("INFO:\tusing map search with paintGrid on %d processes"
          % ncpus)
    start = timeit.default_timer()

    completeness = indexer.paintGrid(
        qfib,
        eta_ome,
        etaRange=np.radians(cfg.find_orientations.eta.range),
        omeTol=np.radians(cfg.find_orientations.omega.tolerance),
        etaTol=np.radians(cfg.find_orientations.eta.tolerance),
        omePeriod=np.radians(cfg.find_orientations.omega.period),
        threshold=on_map_threshold,
        doMultiProc=ncpus > 1,
        nCPUs=ncpus
        )
    print("INFO:\t\t...took %f seconds" % (timeit.default_timer() - start))
completeness = np.array(completeness)


# %%
# =============================================================================
# CLUSTERING AND GRAINS OUTPUT
# =============================================================================
if not os.path.exists(cfg.analysis_dir):
    os.makedirs(cfg.analysis_dir)
qbar_filename = 'accepted_orientations_' + analysis_id + '.dat'

print("INFO:\trunning clustering using '%s'"
      % cfg.find_orientations.clustering.algorithm)
start = timeit.default_timer()

# Simulate N random grains to get neighborhood size
seed_hkl_ids = [
    plane_data.hklDataList[active_hkls[i]]['hklID'] for i in fiber_seeds
]

if seed_hkl_ids is not None:
    ngrains = 100
    rand_q = mutil.unitVector(np.random.randn(4, ngrains))
    rand_e = np.tile(2.*np.arccos(rand_q[0, :]), (3, 1)) \
        * mutil.unitVector(rand_q[1:, :])
    refl_per_grain = np.zeros(ngrains)
    num_seed_refls = np.zeros(ngrains)
    grain_param_list = np.vstack(
            [rand_e,
             np.zeros((3, ngrains)),
             np.tile(const.identity_6x1, (ngrains, 1)).T]
        ).T
    sim_results = instr.simulate_rotation_series(
            plane_data, grain_param_list,
            eta_ranges=np.radians(cfg.find_orientations.eta.range),
            ome_ranges=ome_ranges,
            ome_period=np.radians(cfg.find_orientations.omega.period)
    )

    refl_per_grain = np.zeros(ngrains)
    seed_refl_per_grain = np.zeros(ngrains)
    for sim_result in sim_results.itervalues():
        for i, refl_ids in enumerate(sim_result[0]):
            refl_per_grain[i] += len(refl_ids)
            seed_refl_per_grain[i] += np.sum(
                    [sum(refl_ids == hkl_id) for hkl_id in seed_hkl_ids]
                )

    min_samples = max(
        int(np.floor(0.5*compl_thresh*min(seed_refl_per_grain))),
        2
        )
    mean_rpg = int(np.round(np.average(refl_per_grain)))
else:
    min_samples = 1
    mean_rpg = 1

print("INFO:\tmean reflections per grain: %d" % mean_rpg)
print("INFO:\tneighborhood size: %d" % min_samples)
print("INFO:\tFeeding %d orientations above %.1f%% to clustering"
      % (sum(completeness > compl_thresh), compl_thresh))

qbar, cl = run_cluster(
    completeness, qfib, plane_data.getQSym(), cfg,
    min_samples=min_samples,
    compl_thresh=compl_thresh,
    radius=cl_radius)

print("INFO:\t\t...took %f seconds" % (timeit.default_timer() - start))
print("INFO:\tfound %d grains; saved to file: '%s'"
      % (qbar.shape[1], qbar_filename))

np.savetxt(qbar_filename, qbar.T,
           fmt='%.18e', delimiter='\t')

gw = instrument.GrainDataWriter(os.path.join(cfg.analysis_dir, 'grains.out'))
grain_params_list = []
for gid, q in enumerate(qbar.T):
    phi = 2*np.arccos(q[0])
    n = xfcapi.unitRowVector(q[1:])
    grain_params = np.hstack([phi*n, const.zeros_3, const.identity_6x1])
    gw.dump_grain(gid, 1., 0., grain_params)
    grain_params_list.append(grain_params)
gw.close()
