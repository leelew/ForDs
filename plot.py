import os
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap



def plot_day(var_name,
             year, 
             month, 
             day_of_month, # Start from 0 to consist with other codes
             region_name,
             latb=23.5, 
             latu=25.5,
             llon=112.5, 
             rlon=114.5,
             DATA_PATH="/tera06/lilu/ForDs/data/GD/",
             OUT_PATH='/tera06/lilu/ForDs/out/'):
    # load coarse elev
    f = nc.Dataset(DATA_PATH+'DEM/SRTM/ERA5Land_height.nc', 'r')
    lat_coarse, lon_coarse = f['latitude'][:], f['longitude'][:]
    lat_coarse_index = np.where((lat_coarse>latb) & (lat_coarse<latu))[0]
    lon_coarse_index = np.where((lon_coarse>llon) & (lon_coarse<rlon))[0]
    lat_coarse, lon_coarse = lat_coarse[lat_coarse_index], lon_coarse[lon_coarse_index]
    elev_coarse = f['z'][0][lat_coarse_index][:,lon_coarse_index]/9.8
    mask_coarse = np.zeros_like(elev_coarse)
    mask_coarse[np.isnan(elev_coarse)] = np.nan
    
    # load fine elev
    f = nc.Dataset(DATA_PATH+'DEM/MERITDEM/MERITDEM_GD_height.nc', 'r')
    lat_fine, lon_fine = f['lat'][:], f['lon'][:]
    lat_fine_index = np.where((lat_fine>lat_coarse[-1]) & (lat_fine<lat_coarse[0]))[0]
    lon_fine_index = np.where((lon_fine>lon_coarse[0]) & (lon_fine<lon_coarse[-1]))[0]
    lat_fine, lon_fine = lat_fine[lat_fine_index], lon_fine[lon_fine_index]
    elev_fine = f['hgt'][lat_fine_index][:,lon_fine_index]
    mask_fine = np.zeros_like(elev_fine)
    mask_fine[np.isnan(elev_fine)] = np.nan
    
    # read and adjust coarse data
    # NOTE: We read coarse data in montly scale, and the `day_of_month` is used to index the data
    #       Thus, the `day_of_month` should start from 0
    if var_name == 'ws':
        f = nc.Dataset(DATA_PATH+"forcing/ERA5LAND_GD_{year:04}_{month:02}_u10.nc".format(year=year, month=month, name=var_name),'r')
        u10_coarse = f['u10'][day_of_month*24:day_of_month*24+24][:,lat_coarse_index][:,:,lon_coarse_index]+mask_coarse[np.newaxis]
        f = nc.Dataset(DATA_PATH+"forcing/ERA5LAND_GD_{year:04}_{month:02}_v10.nc".format(year=year, month=month, name=var_name),'r')
        v10_coarse = f['v10'][day_of_month*24:day_of_month*24+24][:,lat_coarse_index][:,:,lon_coarse_index]+mask_coarse[np.newaxis]
        var_coarse = np.sqrt(u10_coarse**2+v10_coarse**2)
    elif 'tp' in var_name:
        f = nc.Dataset(DATA_PATH+"forcing/ERA5LAND_GD_{year:04}_{month:02}_tp.nc".format(year=year, month=month, name=var_name),'r')
        var_coarse = f['tp'][day_of_month*24:day_of_month*24+24][:,lat_coarse_index][:,:,lon_coarse_index]+mask_coarse[np.newaxis]
        var_coarse = var_coarse*1000
    else:
        f = nc.Dataset(DATA_PATH+"forcing/ERA5LAND_GD_{year:04}_{month:02}_{name}.nc".format(year=year, month=month, name=var_name),'r')
        var_coarse = f[var_name][day_of_month*24:day_of_month*24+24][:,lat_coarse_index][:,:,lon_coarse_index]+mask_coarse[np.newaxis]

    # read fine data
    # NOTE: The fine data is saved in daily scale, and the `day_of_month` should start from 1
    day_of_month = day_of_month+1 
    f = nc.Dataset(OUT_PATH+"ERA5LAND_GD_fine_{year:04}_{month:02}_{day:02}_{name}.nc".format(year=year, month=month, day=day_of_month, name=var_name),'r')
    var_fine = f[var_name][:]+mask_fine[np.newaxis]

    # meshgrid
    grid_lon_coarse, grid_lat_coarse = np.meshgrid(lon_coarse, lat_coarse)
    grid_lon_fine, grid_lat_fine = np.meshgrid(lon_fine, lat_fine)

    # plot 24 hours for coase and fine
    plt.figure(figsize=(15,15))
    for i in range(24):
        ax = plt.subplot(5,5,i+1)
        plt.imshow(var_coarse[i], vmin=np.min(var_coarse[i]), vmax=np.max(var_coarse[i]), cmap='jet')
        plt.colorbar()
    plt.savefig(var_name+"_{name}_hours_coarse.png".format(name=region_name))

    plt.figure(figsize=(15,15))
    for i in range(24):
        ax = plt.subplot(5,5,i+1)
        plt.imshow(var_fine[i], vmin=np.min(var_coarse[i]), vmax=np.max(var_coarse[i]), cmap='jet')
        plt.colorbar()
    plt.savefig(var_name+"_{name}_hours_fine.png".format(name=region_name))

    if 'tp' in var_name:
        var_coarse = np.nansum(var_coarse, axis=0)
        var_fine = np.nansum(var_fine, axis=0)
        
    else:
        var_coarse = np.nanmean(var_coarse, axis=0)
        var_fine = np.nanmean(var_fine, axis=0)

    plt.figure()
    ax = plt.subplot(2,2,1)
    plt.imshow(var_coarse, vmin=np.min(var_coarse), vmax=np.max(var_coarse), cmap='jet')
    ax = plt.subplot(2,2,2)
    plt.imshow(var_fine, vmin=np.min(var_coarse), vmax=np.max(var_coarse), cmap='jet')
    plt.savefig(var_name+"_{name}_day_compare.png".format(name=region_name))



if __name__ == '__main__':
    plot_day('t2m', 2018, 1, 3, 'SG', latb=23.5, latu=25.5, llon=112.5, rlon=114.5)
    plot_day('d2m', 2018, 1, 3, 'SG', latb=23.5, latu=25.5, llon=112.5, rlon=114.5)
    plot_day('Q', 2018, 1, 3, 'SG', latb=23.5, latu=25.5, llon=112.5, rlon=114.5)
    #plot_day('ssrd', 2018, 1, 3, 'SG', latb=23.5, latu=25.5, llon=112.5, rlon=114.5)
    plot_day('strd', 2018, 1, 3, 'SG', latb=23.5, latu=25.5, llon=112.5, rlon=114.5)
    plot_day('sp', 2018, 1, 3, 'SG', latb=23.5, latu=25.5, llon=112.5, rlon=114.5)
    plot_day('ws', 2018, 1, 3, 'SG', latb=23.5, latu=25.5, llon=112.5, rlon=114.5)
    #plot_day('tp_mtl_dnn', 2018, 1, 3, 'SG', latb=23.5, latu=25.5, llon=112.5, rlon=114.5)
    plot_day('tp_rf', 2018, 1, 3, 'SG', latb=23.5, latu=25.5, llon=112.5, rlon=114.5)
    os.system("mv *png figures/")