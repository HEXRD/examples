#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 09:50:12 2020

@author: joel
"""

import numpy as np

import os
 
try:
    import dill as cpl
except(ImportError):
    import cPickle as cpl

import yaml

from hexrd import constants as const
from hexrd import imageseries
from hexrd import instrument
from hexrd.xrd.fitting import fitGrain, objFuncFitGrain, gFlag_ref
from hexrd.xrd import transforms_CAPI as xfcapi

# =============================================================================
# I/O methods
# =============================================================================

# plane data
def load_pdata(cpkl, key):
    """
    Extract specified PlaneData object from materials archive (pickle).

    Parameters
    ----------
    cpkl : str
        Filename for the pickle containing the materials list.
    key : str
        The name for the desired material object.

    Returns
    -------
    hexrd.xrd.crystallography.PlaneData
        The specified PlaneData object stored in the materials archive.

    """
    with file(cpkl, "r") as matf:
        mat_list = cpl.load(matf)
    return dict(zip([i.name for i in mat_list], mat_list))[key].planeData


# images
def load_images(fc_filename):
    """
    Instantiate an imageseries from the specified frame-cache archive (npz).

    Parameters
    ----------
    fc_filename : str
        The filename path of the target frame-cache (npz) archive.

    Returns
    -------
    hexrd.imageseries.baseclass.ImageSeries
        The imageseries object stored in the specified npz archive.

    """
    return imageseries.open(fc_filename, format="frame-cache", style="npz")


# instrument
def load_instrument(yml):
    """
    Instantiate the instrument object described by the specified YAML config.

    Parameters
    ----------
    yml : str
        The filename path for the YAML config describing the instrument.

    Returns
    -------
    instr : hexrd.instrument.HEDMInstrument
        The instrument class specifed by the YAML config.

    """
    with file(yml, 'r') as f:
        icfg = yaml.load(f, Loader=yaml.SafeLoader)
    instr = instrument.HEDMInstrument(instrument_config=icfg)
    return instr


def process_max_tth(cfg):
    # process max_tth option for plane_data
    max_tth = cfg.fit_grains.tth_max
    if max_tth:
        if type(cfg.fit_grains.tth_max) != bool:
            max_tth = np.degrees(float(max_tth))
    else:
        max_tth = None
    return max_tth

# =============================================================================
# FINDORIENTATIONS
# =============================================================================

# multiprocessing fit funcs for DIRECT SEARCH only; map-based search is in
# hexrd.xrd.indexer

def test_orientation_FF_init(params):
    """
    Broadcast the indexing parameters as globals for multiprocessing

    Parameters
    ----------
    params : dict
        The dictionary of indexing parameters.

    Returns
    -------
    None.
    
    Notes
    -----
    See test_orientation_FF_reduced for specification.
    """
    global paramMP
    paramMP = params

def test_orientation_FF_reduced(quat):
    """
    

    Parameters
    ----------
    quat : array_like (4,)
        The unit quaternion representation for the orientation to be tested.

    Returns
    -------
    float
        The completeness, i.e., the ratio between the predicted and observed 
        Bragg reflections subject to the specified tolerances.

    
    Notes
    -----
    input parameters are 
    [plane_data, instrument, imgser_dict,
    tth_tol, eta_tol, ome_tol, eta_ranges, ome_period,
    npdiv, threshold]
    """
    plane_data = paramMP['plane_data']
    instrument = paramMP['instrument']
    imgser_dict = paramMP['imgser_dict']
    tth_tol = paramMP['tth_tol']
    eta_tol = paramMP['eta_tol']
    ome_tol = paramMP['ome_tol']
    eta_ranges = paramMP['eta_ranges']
    ome_period = paramMP['ome_period']
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
        eta_ranges=eta_ranges,
        ome_period=ome_period,
        check_only=True)

    return sum(compl)/float(len(compl))


# =============================================================================
# FITGRAINS
# =============================================================================

# multiprocessing fit funcs

def fit_grain_FF_init(params):
    """
    Broadcast the fitting parameters as globals for multiprocessing

    Parameters
    ----------
    params : dict
        The dictionary of fitting parameters.

    Returns
    -------
    None.
    
    Notes
    -----
    See fit_grain_FF_reduced for specification.
    """
    global paramMP
    paramMP = params


def fit_grain_FF_reduced(grain_id):
    """
    Perform non-linear least-square fit for the specified grain.

    Parameters
    ----------
    grain_id : int
        The grain id.

    Returns
    -------
    grain_id : int
        The grain id.
    completeness : float
        The ratio of predicted to measured (observed) Bragg reflections.
    chisq: float
        Figure of merit describing the sum of squared residuals for each Bragg
        reflection in the form (x, y, omega) normalized by the total number of 
        degrees of freedom.
    grain_params : array_like
        The optimized grain parameters
        [<orientation [3]>, <centroid [3]> <inverse stretch [6]>].

    Notes
    -----
    input parameters are 
    [plane_data, instrument, imgser_dict,
    tth_tol, eta_tol, ome_tol, npdiv, threshold]
    """
    grains_table = paramMP['grains_table']
    plane_data = paramMP['plane_data']
    instrument = paramMP['instrument']
    imgser_dict = paramMP['imgser_dict']
    tth_tol = paramMP['tth_tol']
    eta_tol = paramMP['eta_tol']
    ome_tol = paramMP['ome_tol']
    npdiv = paramMP['npdiv']
    refit = paramMP['refit']
    threshold = paramMP['threshold']
    eta_ranges = paramMP['eta_ranges']
    ome_period = paramMP['ome_period']
    analysis_dirname = paramMP['analysis_dirname']
    spots_filename = paramMP['spots_filename']

    grain = grains_table[grain_id]
    grain_params = grain[3:15]

    for tols in zip(tth_tol, eta_tol, ome_tol):
        complvec, results = instrument.pull_spots(
            plane_data, grain_params,
            imgser_dict,
            tth_tol=tols[0],
            eta_tol=tols[1],
            ome_tol=tols[2],
            npdiv=npdiv, threshold=threshold,
            eta_ranges=eta_ranges,
            ome_period=ome_period,
            dirname=analysis_dirname, filename=spots_filename % grain_id,
            save_spot_list=False,
            quiet=True, check_only=False, interp='nearest')

        # ======= DETERMINE VALID REFLECTIONS =======

        culled_results = dict.fromkeys(results)
        num_refl_tot = 0
        num_refl_valid = 0
        for det_key in culled_results:
            panel = instrument.detectors[det_key]

            presults = results[det_key]

            valid_refl_ids = np.array([x[0] for x in presults]) >= 0

            spot_ids = np.array([x[0] for x in presults])

            # find unsaturated spots on this panel
            if panel.saturation_level is None:
                unsat_spots = np.ones(len(valid_refl_ids))
            else:
                unsat_spots = \
                    np.array([x[4] for x in presults]) < panel.saturation_level

            idx = np.logical_and(valid_refl_ids, unsat_spots)

            # if an overlap table has been written, load it and use it
            overlaps = np.zeros_like(idx, dtype=bool)
            try:
                ot = np.load(
                    os.path.join(
                        analysis_dirname, os.path.join(
                            det_key, 'overlap_table.npz'
                        )
                    )
                )
                for key in ot.keys():
                    for this_table in ot[key]:
                        these_overlaps = np.where(
                            this_table[:, 0] == grain_id)[0]
                        if len(these_overlaps) > 0:
                            mark_these = np.array(
                                this_table[these_overlaps, 1], dtype=int
                            )
                            otidx = [
                                np.where(spot_ids == mt)[0]
                                for mt in mark_these
                            ]
                            overlaps[otidx] = True
                idx = np.logical_and(idx, ~overlaps)
                # print("found overlap table for '%s'" % det_key)
            except(IOError, IndexError):
                # print("no overlap table found for '%s'" % det_key)
                pass

            # attach to proper dict entry
            culled_results[det_key] = [presults[i] for i in np.where(idx)[0]]
            num_refl_tot += len(valid_refl_ids)
            num_refl_valid += sum(valid_refl_ids)

            pass  # now we have culled data

        # CAVEAT: completeness from pullspots only; incl saturated and overlaps
        # <JVB 2015-12-15>
        completeness = num_refl_valid / float(num_refl_tot)

        # ======= DO LEASTSQ FIT =======

        if num_refl_valid <= 12:    # not enough reflections to fit... exit
            return grain_id, completeness, np.inf, grain_params
        else:
            grain_params = fitGrain(
                    grain_params, instrument, culled_results,
                    plane_data.latVecOps['B'], plane_data.wavelength
                )
            # get chisq
            # TODO: do this while evaluating fit???
            chisq = objFuncFitGrain(
                    grain_params[gFlag_ref], grain_params, gFlag_ref,
                    instrument,
                    culled_results,
                    plane_data.latVecOps['B'], plane_data.wavelength,
                    ome_period,
                    simOnly=False, return_value_flag=2)
            pass  # end conditional on fit
        pass  # end tolerance looping

    if refit is not None:
        # first get calculated x, y, ome from previous solution
        # NOTE: this result is a dict
        xyo_det_fit_dict = objFuncFitGrain(
            grain_params[gFlag_ref], grain_params, gFlag_ref,
            instrument,
            culled_results,
            plane_data.latVecOps['B'], plane_data.wavelength,
            ome_period,
            simOnly=True, return_value_flag=2)

        # make dict to contain new culled results
        culled_results_r = dict.fromkeys(culled_results)
        num_refl_valid = 0
        for det_key in culled_results_r:
            presults = culled_results[det_key]

            ims = imgser_dict[det_key]
            ome_step = sum(np.r_[-1, 1]*ims.metadata['omega'][0, :])

            xyo_det = np.atleast_2d(
                np.vstack([np.r_[x[7], x[6][-1]] for x in presults])
            )

            xyo_det_fit = xyo_det_fit_dict[det_key]

            xpix_tol = refit[0]*panel.pixel_size_col
            ypix_tol = refit[0]*panel.pixel_size_row
            fome_tol = refit[1]*ome_step

            # define difference vectors for spot fits
            x_diff = abs(xyo_det[:, 0] - xyo_det_fit['calc_xy'][:, 0])
            y_diff = abs(xyo_det[:, 1] - xyo_det_fit['calc_xy'][:, 1])
            ome_diff = np.degrees(
                xfcapi.angularDifference(xyo_det[:, 2],
                                         xyo_det_fit['calc_omes'])
                )

            # filter out reflections with centroids more than
            # a pixel and delta omega away from predicted value
            idx_new = np.logical_and(
                x_diff <= xpix_tol,
                np.logical_and(y_diff <= ypix_tol,
                               ome_diff <= fome_tol)
                               )

            # attach to proper dict entry
            culled_results_r[det_key] = [
                presults[i] for i in np.where(idx_new)[0]
            ]

            num_refl_valid += sum(idx_new)
            pass

        # only execute fit if left with enough reflections
        if num_refl_valid > 12:
            grain_params = fitGrain(
                grain_params, instrument, culled_results_r,
                plane_data.latVecOps['B'], plane_data.wavelength
            )
            # get chisq
            # TODO: do this while evaluating fit???
            chisq = objFuncFitGrain(
                    grain_params[gFlag_ref],
                    grain_params, gFlag_ref,
                    instrument,
                    culled_results_r,
                    plane_data.latVecOps['B'], plane_data.wavelength,
                    ome_period,
                    simOnly=False, return_value_flag=2)
            pass
        pass  # close refit conditional
    return grain_id, completeness, chisq, grain_params
