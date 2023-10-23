import numpy as np
import argparse
import pickle
import netCDF4 as nc
import tensorflow as tf

from utils import save2nc


# Perform downscaling of precipitation if you have prepared the trained model
# TODO: Only predict for each day, expand to begin/end day type
DATA_PATH='/tera06/lilu/ForDs/data/GD/'
parser = argparse.ArgumentParser()
parser.add_argument('--year', type=int, default=2018)
parser.add_argument('--month', type=int, default=1)
parser.add_argument('--day', type=int, default=3) # Start from 0
parser.add_argument('--blat', type=float, default=23.5)
parser.add_argument('--ulat', type=float, default=25.5)
parser.add_argument('--llon', type=float, default=112.5)
parser.add_argument('--rlon', type=float, default=114.5)
parser.add_argument('--region_name', type=str, default='SG')
args = parser.parse_args()

# load trained model
try:
    mtl_dnn = tf.keras.models.load_model("/tera06/lilu/ForDs/mtl_dnn_2cls_{year}.h5".format(year=args.year))
except:
    print("Please train the multi-dnn first!")

try:
    with open("/tera06/lilu/ForDs/run/rf_classifer_{year}.pickle".format(year=args.year),'rb') as f:
        rf_classifer = pickle.load(f)
    with open("/tera06/lilu/ForDs/run/rf_regressor_{year}.pickle".format(year=args.year),'rb') as f:
        rf_regressor = pickle.load(f)
except:
    print("please train the RF model first!")
 
# load coare DEM data （20x20)
f = nc.Dataset(DATA_PATH+'DEM/SRTM/ERA5Land_height.nc', 'r')
lat_coarse, lon_coarse = f['latitude'][:], f['longitude'][:]
lat_coarse_index = np.where((lat_coarse>args.blat) & (lat_coarse<args.ulat))[0]
lon_coarse_index = np.where((lon_coarse>args.llon) & (lon_coarse<args.rlon))[0]
lat_coarse, lon_coarse = lat_coarse[lat_coarse_index], lon_coarse[lon_coarse_index]
elev_coarse = f['z'][0][lat_coarse_index][:,lon_coarse_index]/9.8

# load fine DEM data (1100x1100)
f = nc.Dataset(DATA_PATH+'DEM/MERITDEM/MERITDEM_GD_height.nc', 'r')
lat_fine, lon_fine = f['lat'][:], f['lon'][:]
lat_fine_index = np.where((lat_fine>lat_coarse[-1]) & (lat_fine<lat_coarse[0]))[0]
lon_fine_index = np.where((lon_fine>lon_coarse[0]) & (lon_fine<lon_coarse[-1]))[0]
lat_fine, lon_fine = lat_fine[lat_fine_index], lon_fine[lon_fine_index]
elev_fine = f['hgt'][lat_fine_index][:,lon_fine_index]
nx, ny = elev_fine.shape

# load test data
x = np.load('/tera06/lilu/ForDs/run/x_test_{year}_{month}_{day}.npy'.format(
    year=args.year, month=args.month, day=args.day))
idx = np.unique(np.where(np.isnan(x))[0])
all_idx = np.arange(x.shape[0])
rest_idx = np.delete(all_idx, idx, axis=0)
x1 = np.delete(x, idx, axis=0)
y = np.full((x.shape[0],1), np.nan)
del x

"""
# pred by multi-dnn
y = np.full((x.shape[0],1), np.nan)
value_fine, mask_fine_prob = mtl_dnn(x1)
value_fine, mask_fine_prob = np.array(value_fine), np.array(mask_fine_prob)
mask_fine = np.zeros_like(mask_fine_prob)
mask_fine[mask_fine_prob>0.5] = 1
value_fine = 10**(value_fine)-1 # log-trans
value_fine[mask_fine==0] = 0
y[rest_idx] = value_fine
tp_fine = y.reshape(-1, nx, ny)
tp_fine[tp_fine<0] = 0
save2nc('tp_mtl_dnn', args.year, args.month, args.day, np.array(tp_fine), lat_fine, lon_fine)
"""

# pred by rf
value_fine = rf_regressor.predict(x1)
mask_fine = rf_classifer.predict(x1)
value_fine = 10**(value_fine)-1 # log-trans
value_fine[mask_fine==0] = 0
y[rest_idx] = value_fine[:,np.newaxis]
tp_fine = y.reshape(-1, nx, ny)
tp_fine[tp_fine<0] = 0
save2nc('tp_rf', args.year, args.month, args.day, np.array(tp_fine), lat_fine, lon_fine)

