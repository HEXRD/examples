#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 13:53:30 2020

@author: bernier2
"""

import copy
import numpy as np
import yaml

from hexrd import instrument

# instrument
def load_instrument(yml):
    """
    Instantiate an instrument from YAML spec.

    Parameters
    ----------
    yml : str
        filename for the instrument configuration in YAML format.

    Returns
    -------
    hexrd.instrument.HEDMInstrument
        Instrument instance.

    """
    with open(yml, 'r') as f:
        icfg = yaml.safe_load(f)
    return instrument.HEDMInstrument(instrument_config=icfg)


# %%
comp_instrument_name = 'dexelas_id3a_20200130.yml'

col_gap = 0
row_gap = 0

nrows = 2
ncols = 2


base_dim = (1944, 1536)

# %%
instr = load_instrument(comp_instrument_name)

row_starts = [i*(base_dim[0] + row_gap) for i in range(nrows)]
col_starts = [i*(base_dim[1] + col_gap) for i in range(ncols)]
rr, cc = np.meshgrid(row_starts, col_starts, indexing='ij')

icfg_dict = instr.write_config()
new_icfg_dict = dict(beam=icfg_dict['beam'],
                     oscillation_stage=icfg_dict['oscillation_stage'],
                     detectors={})
output_instrument_name = comp_instrument_name.split('.')[0] + '_comp.yml'

# %%
for panel_id, panel in instr.detectors.items():
    pcfg_dict = panel.config_dict(instr.chi, instr.tvec)['detector']

    pnum = 0
    panel_names = []
    panel_masks = {}
    for i in range(nrows):
        for j in range(ncols):
            panel_name = '%s_%d_%d' % (panel_id, i, j)

            rstr = rr[i, j]
            rstp = rr[i, j] + base_dim[0]
            cstr = cc[i, j]
            cstp = cc[i, j] + base_dim[1]

            ic_pix = 0.5*(rstr + rstp)
            jc_pix = 0.5*(cstr + cstp)
            # ax.plot(jc_pix, ic_pix, 'bo')

            sp_tvec = np.concatenate(
                [panel.pixelToCart(np.atleast_2d([ic_pix, jc_pix])).flatten(),
                 np.zeros(1)]
            )

            tvec = np.dot(panel.rmat, sp_tvec) + panel.tvec

            # new config dict
            tmp_cfg = copy.deepcopy(pcfg_dict)

            # fix sizes
            tmp_cfg['pixels']['rows'] = base_dim[0]
            tmp_cfg['pixels']['columns'] = base_dim[1]

            # update tvec
            tmp_cfg['transform']['translation'] = tvec.tolist()

            new_icfg_dict['detectors'][panel_name] = copy.deepcopy(tmp_cfg)

            # subimg = img[rstr:rstp, cstr:cstp]

            '''
            mask = np.ones_like(subimg, dtype=bool)
            mask[:m, :] = False
            mask[-m:, :] = False
            mask[:, :m] = False
            mask[:, -m:] = False
            imask = binary_fill_holes(
                np.logical_and(subimg >= 0, subimg < 2**20)
            )
            panel_masks[panel_name] = np.logical_and(mask, imask)

            np.save('Pilatus_%s_mask.npy'
                    % panel_name, panel_masks[panel_name])

            io.imsave(output_template % (i, j), subimg)

            img_stack[pnum] = subimg
            '''
            panel_names.append(panel_name)

            pnum += 1
            pass
        pass

# %% instrument
with open(output_instrument_name, 'w') as fid:
    print(yaml.dump(new_icfg_dict, default_flow_style=False), file=fid)
