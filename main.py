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
        year=year, month=month, begin_hour=begin_hour, var=name), 'w', format='NETCDF4')
    f.createDimension('longitude', size=var.shape[-1])
    f.createDimension('latitude', size=var.shape[-2])
    f.createDimension('time', size=var.shape[-3])

    lon0 = f.createVariable('longitude', 'f4', dimensions='longitude')
    lat0 = f.createVariable('latitude', 'f4', dimensions='latitude')
    data = f.createVariable(name, 'f4', dimensions=('time','latitude','longitude'))

    lon0[:], lat0[:], data[:] = lon_out, lat_out, var
    f.close()



def main(year, month, begin_hour):
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
    air_temperature_fine_interp = f['t2m'][begin_hour:begin_hour+24,:,:]
    print(air_temperature_fine_interp.shape)
    f = nc.Dataset(DATA_PATH+'forcing/ERA5LAND_GD_{year:04}_{month:02}_t2m.nc'.format(year=year, month=month), 'r')
    air_temperature_coarse = f['t2m'][begin_hour:begin_hour+24,:,:]
    print(air_temperature_coarse.shape)
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
                                                     begin_hour)
    t2 = time.time()
    print(t2-t1)
    print(air_temperature_fine.shape)
    save2nc('t2m', year, month, begin_hour, air_temperature_fine)


def par_main(year, month):
    from multiprocessing import Process
    # generate hour length according to year and month
    if ((year%4==0) and (year%100!=0)) or year%400==0:
        month_day = [31,29,31,30,31,30,31,31,30,31,30,31]
    else:
        month_day = [31,28,31,30,31,30,31,31,30,31,30,31]
    
    # generate multiprocessing for each 24 hour interval
    for i in range(2):#month_day[month-1]):
        job = Process(target=main,  args=(year,month,i))
        job.start()
    job.join()

if __name__ == '__main__':
    par_main(2018, 1)
