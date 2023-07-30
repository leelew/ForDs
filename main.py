import os
import numpy as np
import netCDF4 as nc
from multiprocessing import Process
from topography_downscale import downscale_air_temperature 
import time


def save2nc(name, year, month, begin_hour, var):
    out_file = '/tera06/lilu/ForDs/data/DEM/MERITDEM/MERITDEM_GD_Aspect.nc'
    f = nc.Dataset(out_file)
    lat_out, lon_out = f['lat'][:], f['lon'][:]

    f = nc.Dataset('ERA5LAND_GD_fine_{year:04}_{month:02}_{begin_hour:02}_{var}.nc'.format(
        year=year, month=month, begin_hour=begin_hour+1, var=name), 'w', format='NETCDF4')
    f.createDimension('longitude', size=var.shape[-1])
    f.createDimension('latitude', size=var.shape[-2])
    f.createDimension('time', size=var.shape[-3])

    lon0 = f.createVariable('longitude', 'f4', dimensions='longitude')
    lat0 = f.createVariable('latitude', 'f4', dimensions='latitude')
    data = f.createVariable(name, 'f4', dimensions=('time','latitude','longitude'))

    lon0[:], lat0[:], data[:] = lon_out, lat_out, var
    f.close()



def main(year, month, day_of_month):
    DATA_PATH = '/tera06/lilu/ForDs/data/'
    
    # load DEM, lat, lon
    f = nc.Dataset(DATA_PATH+'DEM/SRTM/SRTM_GD_0p1.nc', 'r')
    elevation_coarse = f['hgt'][:]
    f = nc.Dataset(DATA_PATH+'DEM/MERITDEM/MERITDEM_GD_height.nc', 'r')
    elevation_fine = f['cur'][:] #??
    f = nc.Dataset(DATA_PATH+'DEM/SRTM/SRTM_GD_interp.nc', 'r')
    elevation_fine_interp = f['hgt'][:]
    f = nc.Dataset(DATA_PATH+'forcing/ERA5LAND_GD_2018_01_t2m.nc', 'r')
    lat_coarse, lon_coarse = f['latitude'][:], f['longitude'][:]

    
    # downscale air temperature
    f = nc.Dataset(DATA_PATH+'forcing/ERA5LAND_GD_{year:04}_{month:02}_t2m_interp.nc'.format(year=year, month=month), 'r')
    air_temperature_fine_interp = f['t2m'][day_of_month*24:day_of_month*24+24,:,:]
    f = nc.Dataset(DATA_PATH+'forcing/ERA5LAND_GD_{year:04}_{month:02}_t2m.nc'.format(year=year, month=month), 'r')
    air_temperature_coarse = f['t2m'][day_of_month*24:day_of_month*24+24,:,:]
    t1 = time.time()
    air_temperature_fine = downscale_air_temperature(air_temperature_coarse,
                                                     air_temperature_fine_interp,
                                                     elevation_coarse,
                                                     elevation_fine_interp,
                                                     elevation_fine,
                                                     lat_coarse,
                                                     lon_coarse,
                                                     year,
                                                     month,
                                                     day_of_month)
    t2 = time.time()
    print(t2-t1)
    print(air_temperature_fine.shape)
    save2nc('t2m', year, month, day_of_month, air_temperature_fine)


    # downscale dew temperature
    f = nc.Dataset(DATA_PATH+'forcing/ERA5LAND_GD_{year:04}_{month:02}_d2m_interp.nc'.format(year=year, month=month), 'r')
    dew_temperature_fine_interp = f['d2m'][day_of_month*24:day_of_month*24+24,:,:]
    f = nc.Dataset(DATA_PATH+'forcing/ERA5LAND_GD_{year:04}_{month:02}_d2m.nc'.format(year=year, month=month), 'r')
    dew_temperature_coarse = f['d2m'][day_of_month*24:day_of_month*24+24,:,:]
    t1 = time.time()
    dew_temperature_fine = downscale_dew_temperature(dew_temperature_coarse,
                                                     dew_temperature_fine_interp,
                                                     elevation_coarse,
                                                     elevation_fine_interp,
                                                     elevation_fine,
                                                     lat_coarse,
                                                     lon_coarse,
                                                     year,
                                                     month,
                                                     day_of_month)
    t2 = time.time()
    print(t2-t1)
    print(dew_temperature_fine.shape)
    save2nc('d2m', year, month, day_of_month, dew_temperature_fine)


    # downscale air pressure
    f = nc.Dataset(DATA_PATH+'forcing/ERA5LAND_GD_{year:04}_{month:02}_sp_interp.nc'.format(year=year, month=month), 'r')
    air_pressure_fine_interp = f['sp'][day_of_month*24:day_of_month*24+24,:,:]
    t1 = time.time()
    air_pressure_fine = downscale_air_pressure(air_pressure_fine_interp,
                                               air_temperature_fine_interp,
                                               air_temperature_fine,
                                               elevation_coarse,
                                               elevation_fine_interp,
                                               elevation_fine)
    t2 = time.time()
    print(t2-t1)
    print(air_pressure_fine.shape)
    save2nc('sp', year, month, day_of_month, air_pressure_fine)


    # downscale specific humidity
    t1 = time.time()
    specific_humidity_fine = downscale_specific_humidity(air_pressure_fine,
                                                         dew_temperature_fine)
    t2 = time.time()
    print(t2-t1)
    print(specific_humidity_fine.shape)
    save2nc('Q', year, month, day_of_month, specific_humidity_fine)



def par_main(year, month, begin_day, end_day):
    from multiprocessing import Process
    # generate hour length according to year and month
    if ((year%4==0) and (year%100!=0)) or year%400==0:
        month_day = [31,29,31,30,31,30,31,31,30,31,30,31]
    else:
        month_day = [31,28,31,30,31,30,31,31,30,31,30,31]
    
    # generate multiprocessing for each 24 hour interval
    for i in range(begin_day, end_day+1):
        job = Process(target=main,  args=(year,month,i))
        job.start()
    job.join()



if __name__ == '__main__':
    par_main(2018, 1, 0, 5)
    par_main(2018, 1, 6, 11)
    par_main(2018, 1, 12, 17)
    par_main(2018, 1, 18, 23)
    par_main(2018, 1, 24, 31)


