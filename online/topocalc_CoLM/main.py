import numpy as np
import xarray as xr
import netCDF4 as nc

from gradient import gradient_d8
from viewf import viewf
from cur import calc_cur


def main(dem, dx, dy, lat, lon, spacing):
    # aspect (rad), slope (rad)?
    slp, asp = gradient_d8(dem, dx, dy, aspect_rad=True)

    # cur
    cur = calc_cur(dem, dx, dy)

    # sky view factor, terrain configuration factor, terrain elevation angle
    # NOTE: aspect is radian ranging from [-pi, pi]
    svf, tcf, tea, tea_front, tea_back = viewf(dem, spacing, slp, np.pi-asp)
 
    # save nc files
    save2nc(lat, lon, tea, "tea", "terrain_elev_angle", dim_size=3)
    save2nc(lat, lon, tea_front, "tea_front", "terrain_elev_angle_front",dim_size=3)
    save2nc(lat, lon, tea_back, "tea_back", "terrain_elev_angle_back",dim_size=3)
    save2nc(lat, lon, svf, "svf", "sky_view_factor")
    save2nc(lat, lon, tcf, "tcf", "terrain_configuration_factor")
    save2nc(lat, lon, slp, "slope", "slope")
    save2nc(lat, lon, asp, "aspect", "aspect")
    save2nc(lat, lon, cur, "curvature", "curvature")


def save2nc(lat, lon, H, varname, filename, dim_size=2):
    f = nc.Dataset(f"{filename}.nc", "w", format="NETCDF4")
    
    f.createDimension("lat", size=len(lat))
    f.createDimension("lon", size=len(lon))
    lats = f.createVariable("lat", "f4", dimensions="lat")
    lons = f.createVariable("lon", "f4", dimensions="lon")
    lats[:] = lat
    lons[:] = lon

    H[np.isnan(H)] = -9999

    if dim_size == 2:
        vars=f.createVariable(varname, "f8", dimensions=("lat","lon"), fill_value = -9999.)
    elif dim_size == 3:
        f.createDimension("azimuth",size = 36)
        vars=f.createVariable(varname,"f8",dimensions=("lat","lon","azimuth"), fill_value = -9999.)

    vars[:] = H
    f.close()



if __name__ == '__main__':
    DATA_PATH = '/tera05/lilu/students/Chenss/ForDs_bk/data/Colo/DEM/MERITDEM/'
    f = xr.open_dataset(DATA_PATH+'MERITDEM_height.nc')
    dem = f["hgt"].data.astype("double")
    lat = f["latitude"].data
    lon = f["longitude"].data
    dx, dy = 90, 90 # length of a grid (unit:m)
    spacing = 90 # grid spacing of the DEM (unit:m)
    main(dem, dx, dy, lat, lon, spacing)



