# ForDs | Topography-based downscaling of meteorological forcing data
[Lu Li](https://www.researchgate.net/profile/Lu-Li-69?ev=hdr_xprf)

### Introduction

Here, we developed a comprehensive downscaling framework based on topography-adjusted methods and automated machine learning (AutoML). 
With this framework, a 90 m atmospheric forcing dataset is developed from ERA5 data at a 0.25° resolution, and the Common Land Model
(CoLM) is then forced with the developed forcing data over two complex terrain regions (Heihe and Upper Colorado River basins). 
We systematically evaluated the downscaled forcing and the CoLM outputs against both in-situ observations and gridded data. 
The ground-based validation results suggested consistent improvements for all downscaled forcing variables. The downscaled
forcings, which incorporated detailed topographic features, offered improved magnitude estimates, achieving a comparable level 
of performance to that of regional reanalysis forcing data. The downscaled forcing driving the CoLM model show comparable or better 
skills in simulating water and energy fluxes, as verified by in-situ validations. The hyper-resolution simulations offered a detailed 
and more reasonable description of land surface processes and attained similar spatial patterns and magnitudes with high-resolution 
land surface data, especially over highly elevated areas. 

### Citation

This work is under review in Journal of Geophysical Research: Atmospheres.
In case you use ForDs in your research or work, please cite this preprint:

```bibtex
@article{Lu Li,
    author = {SisiChen, Lu Li, Yongjiu Dai et al.},
    title = {Exploring Topography Downscaling Methods for Hyper-Resolution Land Surface Modeling},
    year = {2024},
    journal = {Journal of Geophysical Research: Atmospheres},
    DOI = {10.1029/2024JD041338}
}
```

### [License](https://github.com/leelew/ForDs/LICENSE)

Copyright (c) 2024, Lu Li

