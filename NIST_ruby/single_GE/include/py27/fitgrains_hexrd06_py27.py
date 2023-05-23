#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 19:04:10 2017

@author: bernier2
"""
import argparse
import os
import multiprocessing
import numpy as np
import timeit
import sys
import yaml

from hexrd import config
from hexrd import constants as cnst
from hexrd import instrument
from hexrd.xrd import transforms_CAPI as xfcapi

from hedmutil import process_max_tth, fit_grain_FF_init, fit_grain_FF_reduced


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fit grain parameters for and indexed ff-HEDM dataset')
    parser.add_argument(
        'cfg', metavar='cfg_filename',
        type=str, help='a YAML ff-HEDM config filename')
    parser.add_argument(
        'opts', metavar='options_filename',
        type=str, help='a YAML filename specifying parameters for indexing')

    args = vars(parser.parse_args(sys.argv[1:]))

    print("loaded options file %s" % args['opts'])
    
    cfg_filename = args['cfg']
    opts_filename = args['opts']

    cfg_list = config.open(cfg_filename)
    opts = yaml.load(open(opts_filename, 'r'), Loader=yaml.SafeLoader)
    
    # grab additional options from YAML file
    fitg_opts = opts['options']['fit_grains']
    block_number = fitg_opts['block_number']
    clobber_grains = fitg_opts['clobber_grains']

    # process block id options
    if block_number is not None:
        if isinstance(block_number, int):
            block_ids = [block_number, ]
        elif isinstance(block_number, list):
            block_ids = block_number
    else:
        block_ids = range(len(cfg_list))

    for block_id in block_ids:
        # =====================================================================
        # INITIALIZATION
        # =====================================================================
        print("fitting block %d" % block_id)

        cfg = cfg_list[block_id]
        
        analysis_id = '%s_%s' % (
            cfg.analysis_name.strip().replace(' ', '-'),
            cfg.material.active.strip().replace(' ', '-'),
            )
        
        grains_filename = os.path.join(
            cfg.analysis_dir, 'grains.out'
        )
        
        # load plane data
        plane_data = cfg.material.plane_data
        max_tth = process_max_tth(cfg)
        if max_tth is not None:
            plane_data.exclusions = None
            plane_data.tThMax = max_tth
        
        # load instrument
        instr = cfg.instrument.hedm
                
        # make output directories
        if not os.path.exists(cfg.analysis_dir):
            os.mkdir(cfg.analysis_dir)
            for det_key in instr.detectors:
                os.mkdir(os.path.join(cfg.analysis_dir, det_key))
        else:
            # make sure panel dirs exist under analysis dir
            for det_key in instr.detectors:
                if not os.path.exists(os.path.join(cfg.analysis_dir, det_key)):
                    os.mkdir(os.path.join(cfg.analysis_dir, det_key))

        # load imageseries
        imsd = cfg.image_series
        
        # grab eta ranges and ome_period
        eta_ranges = np.radians(cfg.find_orientations.eta.range)
        ome_period = np.radians(cfg.find_orientations.omega.period)
        
        ncpus = cfg.multiprocessing
        
        threshold = cfg.fit_grains.threshold
    
        # =====================================================================
        # FITTING
        # =====================================================================
        
        # make sure grains.out is there...
        if not os.path.exists(grains_filename) or clobber_grains:
            try:
                qbar = np.loadtxt(
                    'accepted_orientations_' + analysis_id + '.dat',
                    ndmin=2).T
        
                gw = instrument.GrainDataWriter(grains_filename)
                grain_params_list = []
                for i_g, q in enumerate(qbar.T):
                    phi = 2*np.arccos(q[0])
                    n = xfcapi.unitRowVector(q[1:])
                    grain_params = np.hstack(
                        [phi*n, cnst.zeros_3, cnst.identity_6x1]
                    )
                    gw.dump_grain(int(i_g), 1., 0., grain_params)
                    grain_params_list.append(grain_params)
                gw.close()
            except(IOError):
                if os.path.exists(cfg.fit_grains.estimate):
                    grains_filename = cfg.fit_grains.estimate
                else:
                    raise(RuntimeError, "neither estimate nor %s exist!"
                          % 'accepted_orientations_' + analysis_id + '.dat')
            pass
        
        grains_table = np.loadtxt(grains_filename, ndmin=2)
        spots_filename = "spots_%05d.out"
        params = dict(
                grains_table=grains_table,
                plane_data=plane_data,
                instrument=instr,
                imgser_dict=imsd,
                tth_tol=cfg.fit_grains.tolerance.tth,
                eta_tol=cfg.fit_grains.tolerance.eta,
                ome_tol=cfg.fit_grains.tolerance.omega,
                npdiv=cfg.fit_grains.npdiv,
                refit=cfg.fit_grains.refit,
                threshold=threshold,
                eta_ranges=eta_ranges,
                ome_period=ome_period,
                analysis_dirname=cfg.analysis_dir,
                spots_filename=spots_filename)
    
        # =====================================================================
        # EXECUTE MP FIT
        # =====================================================================
        
        # DO FIT!
        if len(grains_table) == 1 or ncpus == 1:
            print("INFO:\tstarting serial fit")
            start = timeit.default_timer()
            fit_grain_FF_init(params)
            fit_results = map(
                fit_grain_FF_reduced,
                np.array(grains_table[:, 0], dtype=int)
            )
            elapsed = timeit.default_timer() - start
        else:
            print("INFO:\tstarting fit on %d processes"
                  % min(ncpus, len(grains_table)))
            start = timeit.default_timer()
            pool = multiprocessing.Pool(min(ncpus, len(grains_table)),
                                        fit_grain_FF_init,
                                        (params, ))
            fit_results = pool.map(
                fit_grain_FF_reduced,
                np.array(grains_table[:, 0], dtype=int)
            )
            pool.close()
            elapsed = timeit.default_timer() - start
        print("INFO: fitting took %f seconds" % elapsed)
        
        # =====================================================================
        # WRITE OUTPUT
        # =====================================================================
        
        gw = instrument.GrainDataWriter(
            os.path.join(cfg.analysis_dir, 'grains.out')
        )
        for fit_result in fit_results:
            gw.dump_grain(*fit_result)
            pass
        gw.close()
