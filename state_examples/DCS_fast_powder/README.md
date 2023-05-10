DCS Fast Powder
===============

This is a good example of performing fast powder calibration with a different
type of peak shape ("DCS" peak shape, which comes from a "pink beam").

However, the more complicated peak shapes also take more time to compute, and
thus the calibration will take longer.

`dcs_fast_powder_refine_detector.h5` is set up to refine the detector parameters
(all three translation vectors and one of the rotation parameters).

`dcs_fast_powder_refine_beam_energy.h5` is set up to only refine the beam
energy.
