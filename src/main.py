#-----------------------------------------------------------------------------
# Main process for topography downscale for meteorological forcing 
#
# Author: Lu Li, Sisi Chen, Zhongwang Wei
#-----------------------------------------------------------------------------
import os
import argparse
import datetime
from multiprocessing import Process

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

from topo import (downscale_air_pressure, 
                  downscale_air_temperature,
                  downscale_dew_temperature,
                  downscale_in_longwave_radiation,
                  downscale_in_shortwave_radiation,
                  downscale_precipitation, 
                  downscale_specific_humidity,
                  downscale_wind_speed, 
                  downscale_precipitation)
from utils import save2nc



def main(
         year, 
         month, 
         day_of_month,  # Start from 0
         blat, ulat,
         llon, rlon,
         region_name,
         DATA_PATH='/tera06/lilu/ForDs/data/GD/'):
    
    print('Downscaling forcing in {year:04}-{month:02}-{day:02}'.format(
        year=year, month=month, day=day_of_month))
    print('The region is {region_name}'.format(region_name=region_name))
    print('The lat range is {blat}-{ulat}'.format(blat=blat,ulat=ulat))
    print('The lon range is {llon}-{rlon}'.format(llon=llon,rlon=rlon))
    print('The authors are Lu Li, Sisi Chen, Zhongwang Wei, SYSU')
    print('\033[1;31m%s\033[0m' % ' Read and Processing input data')
    print("read data begin at: ", datetime.datetime.today())
    
    # load coare DEM data （20x20)
    f = nc.Dataset(DATA_PATH+'DEM/SRTM/ERA5Land_height.nc', 'r')
    lat_coarse, lon_coarse = f['latitude'][:], f['longitude'][:]
    lat_coarse_index = np.where((lat_coarse>blat) & (lat_coarse<ulat))[0]
    lon_coarse_index = np.where((lon_coarse>llon) & (lon_coarse<rlon))[0]
    lat_coarse, lon_coarse = lat_coarse[lat_coarse_index], lon_coarse[lon_coarse_index]
    elev_coarse = f['z'][0][lat_coarse_index][:,lon_coarse_index]/9.8
    
    # save coarse DEM data
    f = nc.Dataset('DEM_coarse.nc', 'w', format='NETCDF4')
    f.createDimension('longitude', size=elev_coarse.shape[-1])
    f.createDimension('latitude', size=elev_coarse.shape[-2])
    lon0 = f.createVariable('longitude', 'f4', dimensions='longitude')
    lat0 = f.createVariable('latitude', 'f4', dimensions='latitude')
    data = f.createVariable('hgt', 'f4', dimensions=('latitude','longitude'))
    lon0[:], lat0[:], data[:] = lon_coarse, lat_coarse, elev_coarse
    f.close()
    
    # load fine DEM data (1100x1100)
    f = nc.Dataset(DATA_PATH+'DEM/MERITDEM/MERITDEM_GD_height.nc', 'r')
    lat_fine, lon_fine = f['lat'][:], f['lon'][:]
    lat_fine_index = np.where((lat_fine>lat_coarse[-1]) & (lat_fine<lat_coarse[0]))[0]
    lon_fine_index = np.where((lon_fine>lon_coarse[0]) & (lon_fine<lon_coarse[-1]))[0]
    lat_fine, lon_fine = lat_fine[lat_fine_index], lon_fine[lon_fine_index]
    elev_fine = f['hgt'][lat_fine_index][:,lon_fine_index]

    # save fine DEM data
    f = nc.Dataset('DEM_fine.nc', 'w', format='NETCDF4')
    f.createDimension('longitude', size=elev_fine.shape[-1])
    f.createDimension('latitude', size=elev_fine.shape[-2])
    lon0 = f.createVariable('longitude', 'f4', dimensions='longitude')
    lat0 = f.createVariable('latitude', 'f4', dimensions='latitude')
    data = f.createVariable('hgt', 'f4', dimensions=('latitude','longitude'))
    lon0[:], lat0[:], data[:] = lon_fine, lat_fine, elev_fine
    f.close()
    
    # load fine DEM interpolate from coarse DEM
    f = nc.Dataset(DATA_PATH+'DEM/SRTM/ERA5Land_height_interp.nc', 'r')
    elev_fine_interp = f['z'][0][lat_fine_index][:,lon_fine_index]/9.8

    
    # load DEM derived params
    f = nc.Dataset(DATA_PATH+'DEM/MERITDEM/MERITDEM_GD_aspect.nc', 'r')
    aspect_fine = f['asp'][lat_fine_index][:,lon_fine_index]
    f = nc.Dataset(DATA_PATH+'DEM/MERITDEM/MERITDEM_GD_slope_deg.nc', 'r')
    slope_fine = np.deg2rad(f['slp'][lat_fine_index][:,lon_fine_index]) # deg -> rad
    f = nc.Dataset(DATA_PATH+'DEM/MERITDEM/MERITDEM_GD_curvature.nc', 'r')
    curvature_fine = f['slp'][lat_fine_index][:,lon_fine_index]
    f = nc.Dataset(DATA_PATH+'DEM/MERITDEM/MERITDEM_GD_SkyViewFactor.nc', 'r')
    svf_fine = f['svf'][lat_fine_index][:,lon_fine_index]
    shadow_mask_LUT = np.load(DATA_PATH+'DEM/MERITDEM/shadow_mask_LUT.npy')[lat_fine_index][:,lon_fine_index]
    

    # get bilinear interpolate weight by cdo
    os.system("cdo griddes DEM_coarse.nc > grid_coarse.txt")
    os.system("sed -i 's/generic/lonlat/g' grid_coarse.txt")
    os.system("cdo griddes DEM_fine.nc > grid_fine.txt")
    os.system("sed -i 's/generic/lonlat/g' grid_fine.txt")
    os.system("cdo setgrid,grid_coarse.txt DEM_coarse.nc DEM_coarse_1.nc")
    os.system("cdo genbil,grid_fine.txt DEM_coarse_1.nc weight.nc")
    os.system("rm -rf DEM_coarse.nc DEM_coarse_1.nc")
    

    print('\033[1;31m%s\033[0m' % ' Downscaling')
    print('We downscale {shape_coarse} to {shape_fine}'.format(
        shape_coarse=elev_coarse.shape, shape_fine=elev_fine.shape))
    print("downscaling begin at: ",datetime.datetime.today())

    # downscale air temperature
    print('processing t2m')
    f = nc.Dataset(DATA_PATH+'forcing/ERA5LAND_GD_{year:04}_{month:02}_t2m_interp.nc'.format(year=year, month=month), 'r')
    t2m_fine_interp = f['t2m'][day_of_month*24:day_of_month*24+24][:,lat_fine_index][:,:,lon_fine_index]
    f = nc.Dataset(DATA_PATH+'forcing/ERA5LAND_GD_{year:04}_{month:02}_t2m.nc'.format(year=year, month=month), 'r')
    t2m_coarse = f['t2m'][day_of_month*24:day_of_month*24+24][:,lat_coarse_index][:,:,lon_coarse_index]
    t2m_fine = downscale_air_temperature(t2m_coarse,
                                         t2m_fine_interp,
                                         elev_coarse,
                                         elev_fine_interp,
                                         elev_fine,
                                         lat_coarse,
                                         lon_coarse,
                                         lat_fine,
                                         lon_fine,
                                         year,
                                         month,
                                         day_of_month)
    save2nc('t2m', year, month, day_of_month, t2m_fine, lat_fine, lon_fine)

    
    # downscale dew temperature
    print('processing d2m')
    f = nc.Dataset(DATA_PATH+'forcing/ERA5LAND_GD_{year:04}_{month:02}_d2m_interp.nc'.format(year=year, month=month), 'r')
    d2m_fine_interp = f['d2m'][day_of_month*24:day_of_month*24+24][:,lat_fine_index][:,:,lon_fine_index]
    f = nc.Dataset(DATA_PATH+'forcing/ERA5LAND_GD_{year:04}_{month:02}_d2m.nc'.format(year=year, month=month), 'r')
    d2m_coarse = f['d2m'][day_of_month*24:day_of_month*24+24][:,lat_coarse_index][:,:,lon_coarse_index]
    d2m_fine = downscale_dew_temperature(d2m_coarse,
                                         d2m_fine_interp,
                                         elev_coarse,
                                         elev_fine_interp,
                                         elev_fine,
                                         lat_coarse,
                                         lon_coarse,
                                         lat_fine,
                                         lon_fine,
                                         year,
                                         month,
                                         day_of_month)
    save2nc('d2m', year, month, day_of_month, d2m_fine, lat_fine, lon_fine)

    # downscale air pressure
    print('processing sp')
    f = nc.Dataset(DATA_PATH+'forcing/ERA5LAND_GD_{year:04}_{month:02}_sp_interp.nc'.format(year=year, month=month), 'r')
    sp_fine_interp = f['sp'][day_of_month*24:day_of_month*24+24][:,lat_fine_index][:,:,lon_fine_index]
    sp_fine = downscale_air_pressure(sp_fine_interp,
                                     t2m_fine_interp,
                                     t2m_fine,
                                     elev_fine_interp,
                                     elev_fine)
    save2nc('sp', year, month, day_of_month, sp_fine, lat_fine, lon_fine)

    # downscale specific humidity
    print('processing Q')
    Q_fine = downscale_specific_humidity(sp_fine, d2m_fine)
    save2nc('Q', year, month, day_of_month, Q_fine, lat_fine, lon_fine)

    # downscale longwave radiation
    print('processing longwave radiation')
    f = nc.Dataset(DATA_PATH+'forcing/ERA5LAND_GD_{year:04}_{month:02}_strd_interp.nc'.format(year=year, month=month), 'r')    
    strd_fine_interp = f['strd'][day_of_month*24:day_of_month*24+24][:,lat_fine_index][:,:,lon_fine_index]    
    f = nc.Dataset(DATA_PATH+'forcing/ERA5LAND_GD_{year:04}_{month:02}_strd.nc'.format(year=year, month=month), 'r')
    strd_coarse = f['strd'][day_of_month*24:day_of_month*24+24][:,lat_coarse_index][:,:,lon_coarse_index] 
    strd_fine = downscale_in_longwave_radiation(strd_coarse,
                                                strd_fine_interp,
                                                t2m_coarse,
                                                d2m_coarse,
                                                t2m_fine,
                                                d2m_fine,
                                                t2m_fine_interp,
                                                lat_coarse,
                                                lon_coarse, 
                                                year, 
                                                month, 
                                                day_of_month)
    save2nc('strd', year, month, day_of_month, strd_fine, lat_fine, lon_fine)
    
    # downscale wind
    print('processing wind speed')
    f = nc.Dataset(DATA_PATH+'forcing/ERA5LAND_GD_{year:04}_{month:02}_u10_interp.nc'.format(year=year, month=month), 'r')
    u10_fine_interp = f['u10'][day_of_month*24:day_of_month*24+24][:,lat_fine_index][:,:,lon_fine_index] 
    f = nc.Dataset(DATA_PATH+'forcing/ERA5LAND_GD_{year:04}_{month:02}_v10_interp.nc'.format(year=year, month=month), 'r')
    v10_fine_interp = f['v10'][day_of_month*24:day_of_month*24+24][:,lat_fine_index][:,:,lon_fine_index] 
    ws_fine = downscale_wind_speed(u10_fine_interp,
                                   v10_fine_interp,
                                   slope_fine,
                                   aspect_fine,
                                   curvature_fine) 
    save2nc('ws', year, month, day_of_month, ws_fine, lat_fine, lon_fine)
    
    
    # downscale short radiation
    print('processing short radiation')
    f = nc.Dataset(DATA_PATH+'Albedo/{year:04}_{month:02}_GD_bsa_interp.nc'.format(year=year, month=month), 'r')
    black_sky_albedo_interp = np.array(f['bsa'][day_of_month,:,:][lat_fine_index][:,lon_fine_index])
    black_sky_albedo_interp[black_sky_albedo_interp<0] = 0
    black_sky_albedo_interp[black_sky_albedo_interp>1] = 0
    black_sky_albedo_interp = np.tile(black_sky_albedo_interp[np.newaxis], (24, 1, 1))
    f = nc.Dataset(DATA_PATH+'Albedo/{year:04}_{month:02}_GD_wsa_interp.nc'.format(year=year, month=month), 'r')
    white_sky_albedo_interp = np.array(f['wsa'][day_of_month,:,:][lat_fine_index][:,lon_fine_index])
    white_sky_albedo_interp[white_sky_albedo_interp<0] = 0
    white_sky_albedo_interp[white_sky_albedo_interp>1] = 0
    white_sky_albedo_interp = np.tile(white_sky_albedo_interp[np.newaxis], (24, 1, 1))

    f = nc.Dataset(DATA_PATH+'forcing/ERA5LAND_GD_{year:04}_{month:02}_ssrd.nc'.format(year=year, month=month), 'r')
    ssrd_coarse = f['ssrd'][day_of_month*24:day_of_month*24+24,:,:][:,lat_coarse_index][:,:,lon_coarse_index]
    ssrd_coarse[ssrd_coarse<0] = 0
    f = nc.Dataset(DATA_PATH+'forcing/ERA5LAND_GD_{year:04}_{month:02}_sp.nc'.format(year=year, month=month), 'r')
    sp_coarse = f['sp'][day_of_month*24:day_of_month*24+24,:,:][:,lat_coarse_index][:,:,lon_coarse_index]
    ssrd_fine = downscale_in_shortwave_radiation(
                                            ssrd_coarse,
                                            sp_coarse,
                                            sp_fine,
                                            sp_fine_interp,
                                            black_sky_albedo_interp,
                                            white_sky_albedo_interp,
                                            slope_fine,
                                            aspect_fine,
                                            svf_fine,
                                            year,
                                            month,
                                            day_of_month,
                                            lat_coarse,
                                            lon_coarse,
                                            lat_fine,
                                            lon_fine,
                                            shadow_mask_LUT)
    save2nc('ssrd', year, month, day_of_month, ssrd_fine, lat_fine, lon_fine)
    
    # downscale precipitation
    """
    # standardization
    norm_const = np.load('norm_params.npy')
    t2m_min, t2m_max = norm_const[0,0], norm_const[0,1]
    d2m_min, d2m_max = norm_const[1,0], norm_const[1,1]
    Q_min, Q_max = norm_const[2,0], norm_const[2,1]
    strd_min, strd_max = norm_const[3,0], norm_const[3,1]
    ssrd_min, ssrd_max = norm_const[4,0], norm_const[4,1]
    sp_min, sp_max = norm_const[5,0], norm_const[5,1]
    ws_min, ws_max = norm_const[6,0], norm_const[6,1]
    elev_max = norm_const[7,0]
    
    t2m_fine = (t2m_fine-t2m_min)/(t2m_max-t2m_min)
    d2m_fine = (d2m_fine-d2m_min)/(d2m_max-d2m_min)
    Q_fine = (Q_fine-Q_min)/(Q_max-Q_min)
    strd_fine = (strd_fine-strd_min)/(strd_max-strd_min)
    ssrd_fine = (ssrd_fine-ssrd_min)/(ssrd_max-ssrd_min)
    sp_fine = (sp_fine-sp_min)/(sp_max-sp_min)
    ws_fine = (ws_fine-ws_min)/(ws_max-ws_min)
    elev_fine = elev_fine/elev_max
    lat_fine = lat_fine/360
    lon_fine = lon_fine/360
    """
    
    downscale_precipitation(
                            t2m_fine,
                            d2m_fine,
                            sp_fine,
                            Q_fine,
                            strd_fine,
                            ssrd_fine,
                            ws_fine,
                            lat_fine,
                            lon_fine,
                            elev_fine,
                            year, 
                            month, 
                            day_of_month)
    print("Please use predict_tp.py to downscale precipitation, \
          this function only save test data")
    print("downscaling end at: ",datetime.datetime.today())
    

def par_main(args):
    # generate hour length according to year and month
    if ((args.year%4==0) and (args.year%100!=0)) or args.year%400==0:
        month_day = [31,29,31,30,31,30,31,31,30,31,30,31]
    else:
        month_day = [31,28,31,30,31,30,31,31,30,31,30,31]
        
    # generate multiprocessing for each 24 hour interval
    for i in range(args.begin_day, args.end_day):
        job = Process(target=main,  args=( 
                                          args.year,
                                          args.month,
                                          i,
                                          args.blat,
                                          args.ulat,
                                          args.llon,
                                          args.rlon,
                                          args.region_name))
        job.start()
    job.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2018)
    parser.add_argument('--month', type=int, default=1)
    parser.add_argument('--begin_day', type=int, default=1)
    parser.add_argument('--end_day', type=int, default=2)
    parser.add_argument('--blat', type=float, default=23.5)
    parser.add_argument('--ulat', type=float, default=24.5)
    parser.add_argument('--llon', type=float, default=112.5)
    parser.add_argument('--rlon', type=float, default=113.5)
    parser.add_argument('--region_name', type=str, default='random')
    args = parser.parse_args()
    
    par_main(args)

