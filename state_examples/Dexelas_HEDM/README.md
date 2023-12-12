Dexelas HEDM
============

The Dexelas HEDM dataset (`dexelas_hedm.h5`) contains 3 grains of ruby.
This is a great example for performing the HEDM workflow.

Due to some limitations in saving frame caches within state files (that will
hopefully soon be fixed), the data files are not saved in the state, and you
must load them in manually after opening the state file.

Afterwards, proceed to "Run" -> "HEDM" -> "Indexing" to begin!

[Example Video](https://drive.google.com/file/d/1WOuwmDsTBN-A3e4Bp_t4oH9J82BVerVQ/view?usp=sharing).


The Dexelas detectors each contain 4 subpanels. Sometimes, treating the
subpanels as separate detectors can produce better results (for example, you
can take into account any misalignment between the subpanels).

The `roi_dexelas_hedm.h5` file is an example where each subpanel is treated
as a separate detector, for a total of 8 detectors. After loading that state
file, the two image files may be loaded as normal. This is an example of using
`roi` and `group` in the instrument config to split images into different
detectors.
