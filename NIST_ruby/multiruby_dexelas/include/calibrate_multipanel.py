
from __future__ import print_function

import os

import numpy as np

import yaml

from hexrd import config
from hexrd import constants as cnst
from hexrd import matrixutil as mutil
from hexrd.xrd import fitting
from hexrd.xrd import transforms_CAPI as xfcapi

from scipy.optimize import leastsq, least_squares

from matplotlib import pyplot as plt

from hedmutil import load_pdata, load_instrument

# =============================================================================
# %% *USER INPUT*
# =============================================================================

# hexrd yaml config file
cfg_filename = 'mruby_config_hexrd06_py27.yml'
block_id = 0    # only change this if you know what you are doing!
overwrite_parfile = False    # if you want to overwrite the instr par
cfg = config.open(cfg_filename)[block_id]
root_cfg_dict = yaml.load(open(cfg_filename, 'r'))

# select which orientaion to use (in case of more than one...)
grain_ids = [0, 1, 2]

# if you know this is a strain-free standard, then set the following to True
clobber_strain = True

'''
# load previously saved exclusions
use_saved_exclusions = False
excl_filename = 'exclusion_index_%05d.npy' % grain_id
'''

# refit tolerances
delta_ome = 0.25
if cfg.fit_grains.refit is not None:
    n_pixels_tol = cfg.fit_grains.refit[0]
    ome_tol = cfg.fit_grains.refit[1]*delta_ome
else:
    n_pixels_tol = 2.
    ome_tol = 2.*delta_ome

# Distrotion handling
# FIXME: still hard-coded for GE!!!
refine_distortion = False
dparam_flags_DFLT = np.zeros(6, dtype=bool)
if refine_distortion:
    dparam_flags_DFLT = np.array([1, 1, 1, 0, 0, 0], dtype=bool)

# grains
grain_flags_DFLT = np.array(
    [1, 1, 1,
     1, 1, 1,
     0, 0, 0, 0, 0, 0],
    dtype=bool)


# =============================================================================
# %% Local function definitions
# =============================================================================


def calibrate_instrument_from_sx(
        instr, grain_params, bmat, xyo_det, hkls_idx,
        param_flags=None, dparam_flags=None, grain_flags=None,
        ome_period=None,
        xtol=cnst.sqrt_epsf, ftol=cnst.sqrt_epsf,
        factor=10., sim_only=False, use_robust_lsq=False):
    """
    arguments xyo_det, hkls_idx are DICTs over panels

    !!!
        distortion is still hosed...
        Currently a dict of detector keys with
        distortion[key] = [d_func, d_params, d_flags]
    """
    pnames = [
        '{:>24s}'.format('beam energy'),
        '{:>24s}'.format('beam azimuth'),
        '{:>24s}'.format('beam polar'),
        '{:>24s}'.format('chi'),
        '{:>24s}'.format('tvec_s[0]'),
        '{:>24s}'.format('tvec_s[1]'),
        '{:>24s}'.format('tvec_s[2]'),
    ]

    for det_key, panel in instr.detectors.iteritems():
        pnames += [
            '{:>24s}'.format('%s tilt[0]' % det_key),
            '{:>24s}'.format('%s tilt[1]' % det_key),
            '{:>24s}'.format('%s tilt[2]' % det_key),
            '{:>24s}'.format('%s tvec[0]' % det_key),
            '{:>24s}'.format('%s tvec[1]' % det_key),
            '{:>24s}'.format('%s tvec[2]' % det_key),
        ]

    # now add distortion if
    for det_key, panel in instr.detectors.iteritems():
        if panel.distortion is not None:
            for j in range(len(panel.distortion[1])):
                pnames.append(
                    '{:>24s}'.format('%s dparam[%d]' % (det_key, j))
                )

    grain_params = np.atleast_2d(grain_params)
    ngrains = len(grain_params)
    for ig, grain in enumerate(grain_params):
        pnames += [
            '{:>24s}'.format('grain %d expmap_c[0]' % ig),
            '{:>24s}'.format('grain %d expmap_c[0]' % ig),
            '{:>24s}'.format('grain %d expmap_c[0]' % ig),
            '{:>24s}'.format('grain %d tvec_c[0]' % ig),
            '{:>24s}'.format('grain %d tvec_c[1]' % ig),
            '{:>24s}'.format('grain %d tvec_c[2]' % ig),
            '{:>24s}'.format('grain %d vinv_s[0]' % ig),
            '{:>24s}'.format('grain %d vinv_s[1]' % ig),
            '{:>24s}'.format('grain %d vinv_s[2]' % ig),
            '{:>24s}'.format('grain %d vinv_s[3]' % ig),
            '{:>24s}'.format('grain %d vinv_s[4]' % ig),
            '{:>24s}'.format('grain %d vinv_s[5]' % ig)
        ]

    # reset parameter flags for instrument as specified
    if param_flags is None:
        param_flags = instr.param_flags
    else:
        # will throw an AssertionError if wrong length
        instr.param_flags = param_flags

    # re-map omegas if need be
    if ome_period is not None:
        for det_key in instr.detectors.iterkeys():
            for ig in range(ngrains):
                xyo_det[det_key][ig][:, 2] = xfcapi.mapAngle(
                        xyo_det[det_key][ig][:, 2],
                        ome_period
                )

    # first grab the instrument parameters
    # 7 global
    # 6*num_panels for the detectors
    # num_panels*ndp in case of distortion
    plist_full = instr.calibration_params()

    dfuncs = {}
    ndparams = {}
    dparam_flags = []
    for det_key, panel in instr.detectors.iteritems():
        if panel.distortion is not None:
            dfuncs[det_key] = panel.distortion[0]
            # ititialize to all False...
            ndparams[det_key] = len(panel.distortion[1])
        else:
            dfuncs[det_key] = None
            ndparams[det_key] = 0
        # FIXME: need to fix distrotion handling instead of hard-coded GE
        # dparam_flags.append(np.zeros(ndparams[det_key], dtype=bool))
        if refine_distortion:
            assert ndparams[det_key] == 6, "wrong length distortion params!"
            dparam_flags.append(dparam_flags_DFLT)
        else:
            dparam_flags.append(np.zeros(ndparams[det_key], dtype=bool))
    dparam_flags = np.hstack(dparam_flags)

    # now handle grains
    # reset parameter flags for grains as specified
    if grain_flags is None:
        grain_flags = np.tile(grain_flags_DFLT, ngrains)

    plist_full = np.concatenate(
        [plist_full, np.hstack(grain_params)]
    )

    # concatenate refinement flags
    refine_flags = np.hstack([param_flags, dparam_flags, grain_flags])
    plist_fit = plist_full[refine_flags]
    fit_args = (plist_full, param_flags,
                dfuncs, dparam_flags, ndparams,
                grain_flags,
                instr, xyo_det, hkls_idx,
                bmat, ome_period)
    if sim_only:
        return sxcal_obj_func(
            plist_fit, plist_full, param_flags,
            dfuncs, dparam_flags, ndparams,
            grain_flags,
            instr, xyo_det, hkls_idx,
            bmat, ome_period,
            sim_only=True)
    else:
        print("Set up to refine:")
        for i in np.where(refine_flags)[0]:
            print("\t%s = %1.7e" % (pnames[i], plist_full[i]))

        # run optimization
        if use_robust_lsq:
            result = least_squares(
                sxcal_obj_func, plist_fit, args=fit_args,
                xtol=xtol, ftol=ftol,
                loss='soft_l1', method='trf'
            )
            x = result.x
            resd = result.fun
            mesg = result.message
            ierr = result.status
        else:
            # do least squares problem
            x, cov_x, infodict, mesg, ierr = leastsq(
                sxcal_obj_func, plist_fit, args=fit_args,
                factor=factor, xtol=xtol, ftol=ftol,
                full_output=1
            )
            resd = infodict['fvec']
        if ierr not in [1, 2, 3, 4]:
            raise RuntimeError("solution not found: ierr = %d" % ierr)
        else:
            print("INFO: optimization fininshed successfully with ierr=%d"
                  % ierr)
            print("INFO: %s" % mesg)

        # ??? output message handling?
        fit_params = plist_full
        fit_params[refine_flags] = x

        # run simulation with optimized results
        sim_final = sxcal_obj_func(
            x, plist_full, param_flags,
            dfuncs, dparam_flags, ndparams,
            grain_flags,
            instr, xyo_det, hkls_idx,
            bmat, ome_period,
            sim_only=True)

        # ??? reset instrument here?
        instr.update_from_calibration_params(fit_params)

        return fit_params, resd, sim_final


def sxcal_obj_func(plist_fit, plist_full, param_flags,
                   dfuncs, dparam_flags, ndparams,
                   grain_flags,
                   instr, xyo_det, hkls_idx,
                   bmat, ome_period,
                   sim_only=False, return_value_flag=None):
    """
    """
    NP_INS = 7
    NP_DET = 6
    NP_GRN = 12

    # stack flags and force bool repr
    refine_flags = np.array(
        np.hstack([param_flags, dparam_flags, grain_flags]),
        dtype=bool)

    # fill out full parameter list
    # !!! no scaling for now
    plist_full[refine_flags] = plist_fit

    # instrument update
    instr.update_from_calibration_params(plist_full)

    # assign some useful params
    wavelength = instr.beam_wavelength
    bvec = instr.beam_vector
    chi = instr.chi
    tvec_s = instr.tvec

    # right now just stuck on the end and assumed
    # to all be the same length... FIX THIS
    xy_unwarped = {}
    meas_omes = {}
    calc_omes = {}
    calc_xy = {}

    # advance counter past distortion parameters
    ii = NP_INS + NP_DET*instr.num_panels
    dp_count = 0
    for ndp in ndparams.itervalues():
        dp_count += ndp

    # grain params
    grain_params = plist_full[ii + dp_count:]
    if np.mod(len(grain_params), NP_GRN) != 0:
        raise RuntimeError("parameter list length is not consistent")
    ngrains = len(grain_params) / NP_GRN
    grain_params = grain_params.reshape((ngrains, NP_GRN))

    # loop over panels
    npts_tot = 0
    for det_key, panel in instr.detectors.iteritems():
        rmat_d = panel.rmat
        tvec_d = panel.tvec

        xy_unwarped[det_key] = []
        meas_omes[det_key] = []
        calc_omes[det_key] = []
        calc_xy[det_key] = []

        dfunc = dfuncs[det_key]
        for ig, grain in enumerate(grain_params):
            ghkls = hkls_idx[det_key][ig]
            xyo = xyo_det[det_key][ig]

            npts_tot += len(xyo)

            xy_unwarped[det_key].append(xyo[:, :2])
            meas_omes[det_key].append(xyo[:, 2])
            if dfunc is not None:    # do unwarping
                dparams = panel.distortion[1]
                xy_unwarped[det_key][ig] = dfunc(
                    xy_unwarped[det_key][ig], dparams
                )
                pass

            # transform G-vectors:
            # 1) convert inv. stretch tensor from MV notation in to 3x3
            # 2) take reciprocal lattice vectors from CRYSTAL to SAMPLE frame
            # 3) apply stretch tensor
            # 4) normalize reciprocal lattice vectors in SAMPLE frame
            # 5) transform unit reciprocal lattice vetors back to CRYSAL frame
            rmat_c = xfcapi.makeRotMatOfExpMap(grain[:3])
            tvec_c = grain[3:6]
            vinv_s = grain[6:]
            gvec_c = np.dot(bmat, ghkls.T)
            vmat_s = mutil.vecMVToSymm(vinv_s)
            ghat_s = mutil.unitVector(np.dot(vmat_s, np.dot(rmat_c, gvec_c)))
            ghat_c = np.dot(rmat_c.T, ghat_s)

            match_omes, calc_omes_tmp = fitting.matchOmegas(
                xyo, ghkls.T,
                chi, rmat_c, bmat, wavelength,
                vInv=vinv_s,
                beamVec=bvec,
                omePeriod=ome_period)

            rmat_s_arr = xfcapi.makeOscillRotMatArray(
                chi, np.ascontiguousarray(calc_omes_tmp)
            )
            calc_xy_tmp = xfcapi.gvecToDetectorXYArray(
                    ghat_c.T, rmat_d, rmat_s_arr, rmat_c,
                    tvec_d, tvec_s, tvec_c
            )
            if np.any(np.isnan(calc_xy_tmp)):
                print("infeasible parameters: "
                      + "may want to scale back finite difference step size")

            calc_omes[det_key].append(calc_omes_tmp)
            calc_xy[det_key].append(calc_xy_tmp)
            pass
        pass

    # return values
    if sim_only:
        retval = {}
        for det_key in calc_xy.keys():
            # ??? calc_xy is always 2-d
            retval[det_key] = []
            for ig in range(ngrains):
                retval[det_key].append(
                    np.vstack(
                        [calc_xy[det_key][ig].T, calc_omes[det_key][ig]]
                    ).T
                )
    else:
        meas_xy_all = []
        calc_xy_all = []
        meas_omes_all = []
        calc_omes_all = []
        for det_key in xy_unwarped.keys():
            meas_xy_all.append(np.vstack(xy_unwarped[det_key]))
            calc_xy_all.append(np.vstack(calc_xy[det_key]))
            meas_omes_all.append(np.hstack(meas_omes[det_key]))
            calc_omes_all.append(np.hstack(calc_omes[det_key]))
            pass
        meas_xy_all = np.vstack(meas_xy_all)
        calc_xy_all = np.vstack(calc_xy_all)
        meas_omes_all = np.hstack(meas_omes_all)
        calc_omes_all = np.hstack(calc_omes_all)

        diff_vecs_xy = calc_xy_all - meas_xy_all
        diff_ome = xfcapi.angularDifference(calc_omes_all, meas_omes_all)
        retval = np.hstack(
            [diff_vecs_xy,
             diff_ome.reshape(npts_tot, 1)]
        ).flatten()
        if return_value_flag == 1:
            retval = sum(abs(retval))
        elif return_value_flag == 2:
            denom = npts_tot - len(plist_fit) - 1.
            if denom != 0:
                nu_fac = 1. / denom
            else:
                nu_fac = 1.
            nu_fac = 1 / (npts_tot - len(plist_fit) - 1.)
            retval = nu_fac * sum(retval**2)
    return retval


def parse_reflection_tables(cfg, instr, grain_ids, refit_idx=None):
    """
    make spot dictionaries
    """
    hkls = {}
    xyo_det = {}
    idx_0 = {}
    for det_key, panel in instr.detectors.iteritems():
        hkls[det_key] = []
        xyo_det[det_key] = []
        idx_0[det_key] = []
        for ig, grain_id in enumerate(grain_ids):
            spots_filename = os.path.join(
                cfg.analysis_dir, os.path.join(
                    det_key, 'spots_%05d.out' % grain_id
                )
            )

            # load pull_spots output table
            gtable = np.loadtxt(spots_filename)

            # apply conditions for accepting valid data
            valid_reflections = gtable[:, 0] >= 0  # is indexed
            not_saturated = gtable[:, 6] < panel.saturation_level
            print("INFO: panel '%s', grain %d" % (det_key, grain_id))
            print("INFO: %d of %d reflections are indexed"
                  % (sum(valid_reflections), len(gtable))
                  )
            print("INFO: %d of %d"
                  % (sum(not_saturated), sum(valid_reflections)) +
                  " valid reflections be are below" +
                  " saturation threshold of %d" % (panel.saturation_level)
                  )

            # valid reflections index
            if refit_idx is None:
                idx = np.logical_and(valid_reflections, not_saturated)
                idx_0[det_key].append(idx)
            else:
                idx = refit_idx[det_key][ig]
                idx_0[det_key].append(idx)
                print("INFO: input reflection specify " +
                      "%d of %d total valid reflections"
                      % (sum(idx), len(gtable))
                      )

            hkls[det_key].append(gtable[idx, 2:5])
            meas_omes = gtable[idx, 12].reshape(sum(idx), 1)
            xyo_det[det_key].append(np.hstack([gtable[idx, -2:], meas_omes]))
    return hkls, xyo_det, idx_0


# %% Initialization...

# read config
cfg = config.open(cfg_filename)[block_id]

# output for eta-ome maps as pickles
working_dir = cfg.working_dir
analysis_name = cfg.analysis_name
analysis_dir = cfg.analysis_dir
analysis_id = "%s_%s" % (analysis_name, cfg.material.active)

# instrument
# 0.6.12 and prior
# instr = load_instrument(cfg.instrument.parameters, saturation_level=65535)
instr = cfg.instrument.hedm
num_panels = instr.num_panels
det_keys = instr.detectors.keys()

# plane data
plane_data = load_pdata(cfg.material.definitions, cfg.material.active)
bmat = plane_data.latVecOps['B']

# the maximum pixel dimension in the instrument for plotting
max_pix_size = 0.
for panel in instr.detectors.itervalues():
    max_pix_size = max(max_pix_size,
                       max(panel.pixel_size_col, panel.pixel_size_col)
                       )
    pass

# grab omega period
# !!! data should be consistent
# !!! this is in degrees
ome_period = cfg.find_orientations.omega.period

# load reflection tables from grain fit
hkls, xyo_det, idx_0 = parse_reflection_tables(cfg, instr, grain_ids)

# load grain parameters
grain_parameters = np.loadtxt(
    os.path.join(cfg.analysis_dir, 'grains.out'),
    ndmin=2)[grain_ids, 3:15]
if clobber_strain:
    for grain in grain_parameters:
        grain[6:] = cnst.identity_6x1
ngrains = len(grain_parameters)

# =============================================================================
# %% plot initial guess
# =============================================================================

xyo_i = calibrate_instrument_from_sx(
    instr, grain_parameters, bmat, xyo_det, hkls,
    ome_period=np.radians(ome_period), sim_only=True
)

for det_key, panel in instr.detectors.iteritems():
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(9, 5))
    fig.suptitle("detector %s" % det_key)
    for ig in range(ngrains):
        ax[0].plot(
            xyo_det[det_key][ig][:, 0],
            xyo_det[det_key][ig][:, 1],
            'k.'
        )
        ax[0].plot(xyo_i[det_key][ig][:, 0], xyo_i[det_key][ig][:, 1], 'rx')
        ax[0].grid(True)
        ax[0].axis('equal')
        ax[0].set_xlim(-0.5*panel.col_dim, 0.5*panel.col_dim)
        ax[0].set_ylim(-0.5*panel.row_dim, 0.5*panel.row_dim)
        ax[0].set_xlabel('detector X [mm]')
        ax[0].set_ylabel('detector Y [mm]')

        ax[1].plot(
            xyo_det[det_key][ig][:, 0],
            np.degrees(xyo_det[det_key][ig][:, 2]), 'k.'
        )
        ax[1].plot(
            xyo_i[det_key][ig][:, 0],
            np.degrees(xyo_i[det_key][ig][:, 2]),
            'rx'
        )
        ax[1].grid(True)
        ax[1].set_xlim(-0.5*panel.col_dim, 0.5*panel.col_dim)
        ax[1].set_ylim(ome_period[0], ome_period[1])
        ax[1].set_xlabel('detector X [mm]')
        ax[1].set_ylabel(r'$\omega$ [deg]')

    fig.show()

# =============================================================================
# %% RUN OPTIMIZATION
# =============================================================================
instr_param_flags = np.array(
    [0,
     0, 0,
     1,
     0, 0, 0], dtype=bool
 )
panel_param_flags = np.tile(
    np.ones(6, dtype=bool), instr.num_panels
)
param_flags = np.hstack([instr_param_flags, panel_param_flags])

grain_flags = np.vstack(
    [grain_flags_DFLT,
     np.array([1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=bool),
     grain_flags_DFLT]
).flatten()

params, resd, xyo_f = calibrate_instrument_from_sx(
    instr, grain_parameters, bmat, xyo_det, hkls,
    ome_period=np.radians(ome_period),
    param_flags=param_flags,
    grain_flags=grain_flags
)

# define difference vectors for spot fits
for det_key, panel in instr.detectors.iteritems():
    for ig in range(ngrains):
        x_diff = abs(xyo_det[det_key][ig][:, 0] - xyo_f[det_key][ig][:, 0])
        y_diff = abs(xyo_det[det_key][ig][:, 1] - xyo_f[det_key][ig][:, 1])
        ome_diff = np.degrees(
            xfcapi.angularDifference(
                xyo_det[det_key][ig][:, 2],
                xyo_f[det_key][ig][:, 2])
        )

        # filter out reflections with centroids more than
        # a pixel and delta omega away from predicted value
        idx_1 = np.logical_and(
            x_diff <= n_pixels_tol*panel.pixel_size_col,
            np.logical_and(
                y_diff <= n_pixels_tol*panel.pixel_size_row,
                ome_diff <= ome_tol
            )
        )

        print("INFO: Will keep %d of %d input reflections "
              % (sum(idx_1), sum(idx_0[det_key][ig]))
              + "on panel %s for re-fit" % det_key)

        idx_new = np.zeros_like(idx_0[det_key][ig], dtype=bool)
        idx_new[np.where(idx_0[det_key][ig])[0][idx_1]] = True
        idx_0[det_key][ig] = idx_new

# =============================================================================
# %% Look ok? Then proceed
# =============================================================================
#
# define difference vectors for spot fits
# for det_key, panel in instr.detectors.iteritems():
#     hkls_refit = hkls[det_key][idx_new[det_key], :]
#     xyo_det_refit = xyo_det[det_key][idx_0[det_key], :]
#     pass

# update calibration crystal params
grain_parameters_fit = params[-grain_parameters.size:].reshape(ngrains, 12)
grain_parameters = grain_parameters_fit

# reparse data
hkls_refit, xyo_det_refit, idx_0 = parse_reflection_tables(
    cfg, instr, grain_ids, refit_idx=idx_0
)

# perform refit
params2, resd2, xyo_f2 = calibrate_instrument_from_sx(
    instr, grain_parameters, bmat, xyo_det_refit, hkls_refit,
    ome_period=np.radians(ome_period),
    param_flags=param_flags,
    grain_flags=grain_flags
)


# =============================================================================
# %% perform refit
# =============================================================================

for det_key, panel in instr.detectors.iteritems():
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(9, 5))
    fig.suptitle("detector %s" % det_key)
    for ig in range(ngrains):
        ax[0].plot(
            xyo_det[det_key][ig][:, 0],
            xyo_det[det_key][ig][:, 1],
            'k.'
        )
        ax[0].plot(xyo_i[det_key][ig][:, 0], xyo_i[det_key][ig][:, 1], 'rx')
        ax[0].plot(xyo_f2[det_key][ig][:, 0], xyo_f2[det_key][ig][:, 1], 'b+')
        ax[0].grid(True)
        ax[0].axis('equal')
        ax[0].set_xlim(-0.5*panel.col_dim, 0.5*panel.col_dim)
        ax[0].set_ylim(-0.5*panel.row_dim, 0.5*panel.row_dim)
        ax[0].set_xlabel('detector X [mm]')
        ax[0].set_ylabel('detector Y [mm]')

        ax[1].plot(
            xyo_det[det_key][ig][:, 0],
            np.degrees(xyo_det[det_key][ig][:, 2]), 'k.'
        )
        ax[1].plot(
            xyo_i[det_key][ig][:, 0],
            np.degrees(xyo_i[det_key][ig][:, 2]),
            'rx'
        )
        ax[1].plot(
            xyo_f2[det_key][ig][:, 0],
            np.degrees(xyo_f2[det_key][ig][:, 2]),
            'b+'
        )
        ax[1].grid(True)
        ax[1].set_xlim(-0.5*panel.col_dim, 0.5*panel.col_dim)
        ax[1].set_ylim(ome_period[0], ome_period[1])
        ax[1].set_xlabel('detector X [mm]')
        ax[1].set_ylabel(r'$\omega$ [deg]')

        ax[0].axis('equal')

    fig.show()


# =============================================================================
# %% output results
# =============================================================================

# update calibration crystal params
grain_parameters_fit = params2[-grain_parameters.size:].reshape(ngrains, 12)
grain_parameters = grain_parameters_fit

calibration_dict = dict.fromkeys(grain_ids)
for grain_id, grain in zip(grain_ids, grain_parameters):
    calibration_dict[grain_id] = {
        'inv_stretch': grain[6:].tolist(),
        'orientation': grain[:3].tolist(),
        'position': grain[3:6].tolist(),
    }

# write out
output_name = 'new_instrument.yml'
if overwrite_parfile:
    output_name = cfg.instrument.parameters
instr.write_config(filename=output_name,
                   calibration_dict=calibration_dict)
