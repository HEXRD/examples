This example uses several scans for a single ruby crystal.
There are two YAML config files available to run.
The first is a completeness study, `cstudy.yml`.
It examines the effect on completeness of changing the selected HKLs in a systematic way.
The second example, `ruby_config.yml`, shows multiple scans on the same grain.
There are six different imageseries files for the scans, and each is used to refit the grain using the previous result as an initial guess.

These examples illustrate the following features of hexrd:
* **HKL selection**. Using the command line interface (CLI), the `dmin` parameter is set in the _material_ section of the config file;
it's value is a cutoff for the lattice spacing, and it determines the list of actual HKLs available in the material.

For the grain fitting, the _fit-grains_ section of the config file determines which HKLs to use by setting the _exclusions_,
which marks HKLs that won't be used.
The exclusions can be set by using various cutoff parameters (`dmin`, `dmax`, `tthmin`, `tthmax`, `sfacmin`, `sfacmax`, `pintmin`, `pintmax`).
In our examples, we use the `sfacmin` parameter to eliminate HKLs with the lowest structure factors (below 5% of maximum value).


* **Multiple Config documents**.
This is a feature of the command line interface. If there are multiple documents in the config file, they are read in sequence, and each subsequent document provides updates to the previous one. In that way, all of the required parameters are in the first document, and for subsequent documents you only have to specify values that are different.
Both examples make use of this feature.

## Completeness Study
In the completeness study, we vary both the `dmin` (minimum lattice spacing) parameter and the `sfac` parameter (structure factor).
We vary the `dmin` parameter for values 0.75 (all HKLs), 0.8 (all rings that are entirely on the detector), 1.0, 1.5 and 1.6 (excluding all rings except the six used for indexing).

The data is for a single ruby crystal.

To run the completeness study:
```
mkdir cstudy
hexrd fit-grains cstudy.yml
```
The completeness results are shown in the table below.
Even including all rings, we get a completeness of over 94%.
Increasing the structure factor cutoff, we get over 99% completeness at 1% cutoff and quickly get to 100%.
Finally, for the highest values of `dmin` (1.6), the fitting essentially fails because we don't enough rings
to fit all the grain data.

```
  |        |   d=0.75 |    d=0.8 |    d=1.0 |   d=1.50 |    d=1.6 |
  |--------+----------+----------+----------+----------+----------|
  | sf=0%  | 0.946759 | 0.959893 | 0.971429 | 1.000000 | 0.906250 |
  | sf=1%  | 0.996552 | 0.996032 | 1.000000 | 1.000000 | 0.906250 |
  | sf=5%  | 1.000000 | 1.000000 | 1.000000 | 1.000000 |0.178571 |
  | sf=10% | 1.000000 | 1.000000 | 1.000000 | 1.000000 |0.178571 |
```
## Multiple Scans
To run this example:
```
mkdir ruby
hexrd fit-grains ruby-config.yml
```
Resulting _grains.out_ files are shown below.
* scan-0
```
# grain ID    completeness  chi^2         exp_map_c[0]             exp_map_c[1]             exp_map_c[2]             t_vec_c[0]               t_vec_c[1]               t_vec_c[2]               inv(V_s)[0,0]            inv(V_s)[1,1]            inv(V_s)[2,2]            inv(V_s)[1,2]*sqrt(2)    inv(V_s)[0,2]*sqrt(2)    inv(V_s)[0,1]*sqrt(2)    ln(V_s)[0,0]             ln(V_s)[1,1]             ln(V_s)[2,2]             ln(V_s)[1,2]             ln(V_s)[0,2]             ln(V_s)[0,1]
0             1.000000      1.174290e-04  6.6909374023432477e-01   -9.8663099205794291e-01  7.3677248443066823e-01   3.7178373363421584e-03   -3.4431142257553514e-04  -6.5611013920972820e-05  1.0001790115667917e+00   1.0001779962708608e+00   1.0001936547581616e+00   -1.0958010675725486e-06  5.2726899612292163e-06   7.4800638218912037e-06   -1.7899552520236116e-04  -1.7798041712088698e-04  -1.9363600225173909e-04  7.7471426212444082e-07   -3.7276622885141345e-06  -5.2882613208036980e-06
```
* scan-1
```
# grain ID    completeness  chi^2         exp_map_c[0]             exp_map_c[1]             exp_map_c[2]             t_vec_c[0]               t_vec_c[1]               t_vec_c[2]               inv(V_s)[0,0]            inv(V_s)[1,1]            inv(V_s)[2,2]            inv(V_s)[1,2]*sqrt(2)    inv(V_s)[0,2]*sqrt(2)    inv(V_s)[0,1]*sqrt(2)    ln(V_s)[0,0]             ln(V_s)[1,1]             ln(V_s)[2,2]             ln(V_s)[1,2]             ln(V_s)[0,2]             ln(V_s)[0,1]
0             1.000000      1.231251e-04  6.6908769186413075e-01   -9.8661218552156082e-01  7.3676961907567451e-01   -9.7093402544126822e-02  5.1963165418904847e-02   -1.0044556006823169e-01  1.0003317256199185e+00   1.0003122948320269e+00   1.0002954874020675e+00   2.0394016696840059e-05   -3.3504518338513817e-05  6.4302951196229646e-06   -3.3167032034982485e-04  -3.1224596389892287e-04  -2.9544336988003502e-04  -1.4416420327864629e-05  2.3683877482590871e-05   -4.5456123226091969e-06
```
* scan-2
```
# grain ID    completeness  chi^2         exp_map_c[0]             exp_map_c[1]             exp_map_c[2]             t_vec_c[0]               t_vec_c[1]               t_vec_c[2]               inv(V_s)[0,0]            inv(V_s)[1,1]            inv(V_s)[2,2]            inv(V_s)[1,2]*sqrt(2)    inv(V_s)[0,2]*sqrt(2)    inv(V_s)[0,1]*sqrt(2)    ln(V_s)[0,0]             ln(V_s)[1,1]             ln(V_s)[2,2]             ln(V_s)[1,2]             ln(V_s)[0,2]             ln(V_s)[0,1]
0             1.000000      2.235621e-04  6.6905000021055194e-01   -9.8666085620059840e-01  7.3679816052237768e-01   -9.8559005263727392e-02  5.2180447882893118e-02   1.0070088941956182e-01   1.0003340182693155e+00   1.0003087917307736e+00   1.0002833033482859e+00   1.1136855662609626e-05   -4.6952822617251431e-05  4.6365416066963762e-06   -3.3396194146805209e-04  -3.0874402805879071e-04  -2.8326264366891282e-04  -7.8726698862318569e-06  3.3190427609560458e-05   -3.2776072614647045e-06
```
* scan-3
```
# grain ID    completeness  chi^2         exp_map_c[0]             exp_map_c[1]             exp_map_c[2]             t_vec_c[0]               t_vec_c[1]               t_vec_c[2]               inv(V_s)[0,0]            inv(V_s)[1,1]            inv(V_s)[2,2]            inv(V_s)[1,2]*sqrt(2)    inv(V_s)[0,2]*sqrt(2)    inv(V_s)[0,1]*sqrt(2)    ln(V_s)[0,0]             ln(V_s)[1,1]             ln(V_s)[2,2]             ln(V_s)[1,2]             ln(V_s)[0,2]             ln(V_s)[0,1]
0             1.000000      1.071730e-04  6.6907967889380837e-01   -9.8667125803142741e-01  7.3678796179872663e-01   -1.4890752697505280e-01  -5.5390962849136388e-02  -1.4326585943747082e-04  1.0003319196889549e+00   1.0003129212915851e+00   1.0003016977291810e+00   8.9741395152150266e-06   -3.8676362238671330e-05  1.2825299433716124e-05   -3.3186420098092002e-04  -3.1287228071096875e-04  -3.0165183371772626e-04  -6.3438493518996067e-06  2.7339685335099753e-05   -9.0660198776052238e-06
```
* scan-4
```
# grain ID    completeness  chi^2         exp_map_c[0]             exp_map_c[1]             exp_map_c[2]             t_vec_c[0]               t_vec_c[1]               t_vec_c[2]               inv(V_s)[0,0]            inv(V_s)[1,1]            inv(V_s)[2,2]            inv(V_s)[1,2]*sqrt(2)    inv(V_s)[0,2]*sqrt(2)    inv(V_s)[0,1]*sqrt(2)    ln(V_s)[0,0]             ln(V_s)[1,1]             ln(V_s)[2,2]             ln(V_s)[1,2]             ln(V_s)[0,2]             ln(V_s)[0,1]
0             1.000000      2.184230e-04  6.6908307690362601e-01   -9.8660921569959248e-01  7.3678364538474306e-01   -4.7008164951465668e-02  1.1799427706793930e-02   -5.1877278583175847e-02  1.0003310867450121e+00   1.0003108249823074e+00   1.0002773693936440e+00   2.9182033177556696e-05   -3.5010718492536793e-05  1.3741894595114956e-05   -3.3103159446223711e-04  -3.1077642627661190e-04  -2.7733041483019808e-04  -2.0628866906735629e-05  2.4748887392629306e-05   -9.7141243998889755e-06
```
* scan-5
```
# grain ID    completeness  chi^2         exp_map_c[0]             exp_map_c[1]             exp_map_c[2]             t_vec_c[0]               t_vec_c[1]               t_vec_c[2]               inv(V_s)[0,0]            inv(V_s)[1,1]            inv(V_s)[2,2]            inv(V_s)[1,2]*sqrt(2)    inv(V_s)[0,2]*sqrt(2)    inv(V_s)[0,1]*sqrt(2)    ln(V_s)[0,0]             ln(V_s)[1,1]             ln(V_s)[2,2]             ln(V_s)[1,2]             ln(V_s)[0,2]             ln(V_s)[0,1]
0             1.000000      1.442392e-04  6.6906630325214522e-01   -9.8657479185228714e-01  7.3678528216049810e-01   -2.0008019416563182e-02  7.0357864841642144e-03   -2.3713331795712750e-02  1.0002881070432232e+00   1.0002731391142208e+00   1.0003200333826343e+00   2.1897682084052492e-05   -5.0614830832335669e-05  8.8035049525911695e-06   -2.8806488890946078e-04  -2.7310167934875535e-04  -3.1998142299921356e-04  -1.5479519856979626e-05  3.5779258906163731e-05   -6.2235485841646080e-06
```
