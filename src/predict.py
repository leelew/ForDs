import argparse
from multiprocessing import Process
import cloudpickle

import netCDF4 as nc
import numpy as np

from utils import save2nc



def pred(args, day_of_month, regressor, classifer, region_name, model_name, normalize=True):
    # load coare DEM data
    f = nc.Dataset(DATA_PATH+'DEM/SRTM/ERA5_height.nc', 'r')
    lat_coarse, lon_coarse = f['latitude'][:], f['longitude'][:]
    lat_coarse_index = np.where((lat_coarse>=args.blat) & (lat_coarse<=args.ulat))[0]
    lon_coarse_index = np.where((lon_coarse>=args.llon) & (lon_coarse<=args.rlon))[0]
    lat_coarse, lon_coarse = lat_coarse[lat_coarse_index], lon_coarse[lon_coarse_index]
    elev_coarse = f['hgt'][lat_coarse_index][:,lon_coarse_index]

    # load fine DEM data
    f = nc.Dataset(DATA_PATH+'DEM/MERITDEM/MERITDEM_height.nc', 'r')
    lat_fine, lon_fine = f['latitude'][:], f['longitude'][:]
    lat_fine_index = np.where((lat_fine>=lat_coarse[-1]) & (lat_fine<=lat_coarse[0]))[0]
    lon_fine_index = np.where((lon_fine>=lon_coarse[0]) & (lon_fine<=lon_coarse[-1]))[0]
    lat_fine, lon_fine = lat_fine[lat_fine_index], lon_fine[lon_fine_index]
    elev_fine = f['hgt'][lat_fine_index][:,lon_fine_index]
    nx, ny = elev_fine.shape

    # load test data
    x = np.load('x_test/{region_name}/x_test_{year:04}_{month}_{day}.npy'.format(
        year=args.year, month=args.month, day=day_of_month, region_name=region_name))
    idx = np.unique(np.where(np.isnan(x))[0])
    all_idx = np.arange(x.shape[0])
    rest_idx = np.delete(all_idx, idx, axis=0)
    x1 = np.delete(x, idx, axis=0)
    y = np.full((x.shape[0],1), np.nan)
    del x

    # 
    if normalize:
        norm = np.load('norm.npy')
        t2m_min, t2m_max = norm[:,0,0], norm[:,0,1]
        d2m_min, d2m_max = norm[:,1,0], norm[:,1,1]
        Q_min, Q_max = norm[:,2,0], norm[:,2,1]
        strd_min, strd_max = norm[:,3,0], norm[:,3,1]
        ssrd_min, ssrd_max = norm[:,4,0], norm[:,4,1]
        sp_min, sp_max = norm[:,5,0], norm[:,5,1]
        ws_min, ws_max = norm[:,6,0], norm[:,6,1]
        elev_max = norm[:,7,0]

    # predict
    value_fine = regressor.predict(x1)
    mask_fine = classifer.predict(x1)
    value_fine = 10**(value_fine)-1 # log-trans
    value_fine[mask_fine==0] = 0
    del mask_fine
    y[rest_idx] = value_fine[:,np.newaxis]
    del value_fine
    y = y.reshape(-1, nx, ny)
    y[y<0] = 0
    save2nc('tp_{n}'.format(n=model_name), args.year, args.month, day_of_month, np.array(y), lat_fine, lon_fine)


def par_pred(args, regressor, classifer):
    # generate hour length according to year and month
    if ((args.year%4==0) and (args.year%100!=0)) or args.year%400==0:
        month_day = [31,29,31,30,31,30,31,31,30,31,30,31]
    else:
        month_day = [31,28,31,30,31,30,31,31,30,31,30,31]
    
    for i in range(args.begin_day, args.end_day):
        job = Process(target=pred,  args=(args, i, regressor, classifer, args.region_name, args.model_name))
        job.start()
    job.join()



if __name__ == '__main__':

    # Perform downscaling of precipitation if you have prepared the trained model
    DATA_PATH='/tera05/lilu/ForDs/data/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2018)
    parser.add_argument('--month', type=int, default=1)
    parser.add_argument('--begin_day', type=int, default=1)
    parser.add_argument('--end_day', type=int, default=2)
    parser.add_argument('--blat', type=float, default=23.5)
    parser.add_argument('--ulat', type=float, default=25.5)
    parser.add_argument('--llon', type=float, default=112.5)
    parser.add_argument('--rlon', type=float, default=114.5)
    parser.add_argument('--region_name', type=str, default='SG')
    parser.add_argument('--model_name', type=str, default='SG')
    args = parser.parse_args()
    DATA_PATH=DATA_PATH+args.region_name+'/'

    # load trained model
    MODEL_PATH = "/tera05/lilu/ForDs/run/models/" + args.region_name + '/'
    f = open(MODEL_PATH+"{model_name}_cls_{year}.pkl".format(model_name=args.model_name,year=args.year, region_name=args.region_name),'rb')
    classifer = cloudpickle.load(f)
    f = open(MODEL_PATH+"{model_name}_reg_{year}.pkl".format(model_name=args.model_name, year=args.year, region_name=args.region_name),'rb') 
    regressor = cloudpickle.load(f)
    classifer.n_jobs = 1
    regressor.n_jobs = 1
    
    par_pred(args, regressor, classifer)
