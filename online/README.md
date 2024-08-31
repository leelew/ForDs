# Tools for CoLM downscaling module
[Lu Li](https://www.researchgate.net/profile/Lu-Li-69?ev=hdr_xprf)

### How to use

**1. Generate topographic factors**

After selecting the simulation regions, CoLM will read high-resolution topographic factors for downscaling process. `topocalc_CoLM` is an automatic tool to generate topographic factors in NetCDF4 types. This tool is based on [topocalc](https://github.com/USDA-ARS-NWRC/topocalc) package developed by USDA ARS Northwest Watershed Research Center with several modifications:
1) Add the calculation of curvature based on methods from ArcGIS.
2) Change the range of aspect (radian) from [-$\pi$, $\pi$] to [0, 2$\pi$].
3) Consider the effect of spatial resolution in the calculation of horizon.

Input your DEM data, and run:
```shell
python3 topocalc_CoLM/main.py
```
then you will get several NetCDF4 files which could be directly used by CoLM.

**2. Offine training precipitation downscaling model**

**3. Prepare environment to coupling machine learning model with CoLM**

**4. Run CoLM**
