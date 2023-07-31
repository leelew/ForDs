import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from mpl_toolkits.basemap import Basemap

DATA_PATH = '/tera06/lilu/ForDs/data/'

f = nc.Dataset(DATA_PATH+'DEM/MERITDEM/MERITDEM_GD_height.nc', 'r')
elevation_fine = f['cur'][:]

f = nc.Dataset("/tera06/lilu/ForDs/data/forcing/ERA5LAND_GD_2018_01_t2m.nc",'r')
tair_coarse = f['t2m'][:]
lat_coarse, lon_coarse = f['latitude'][:], f['longitude'][:]

f = nc.Dataset("/tera06/lilu/ForDs/src/ERA5LAND_GD_fine_2018_01_01_t2m.nc",'r')
tair_fine = f['t2m'][:]
lat_fine, lon_fine =  f['latitude'][:], f['longitude'][:]

grid_lon_coarse, grid_lat_coarse = np.meshgrid(lon_coarse, lat_coarse)
grid_lon_fine, grid_lat_fine = np.meshgrid(lon_fine, lat_fine)


plt.figure()
ax = plt.subplot(2,2,1)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)

m = Basemap(projection='mill',
            llcrnrlat=np.min(lat_fine)-1, urcrnrlat=np.max(lat_fine)+1,
            llcrnrlon=np.max(lon_fine)-1, urcrnrlon=np.max(lon_fine)+1,
            ax=ax)
x, y = m(grid_lon_coarse, grid_lat_coarse)
sc = m.pcolormesh(x, y, tair_coarse[0] ,vmin=np.min(tair_coarse), vmax=np.max(tair_coarse), cmap='jet')
m.readshapefile("guangdong_shp/guangdong", "guangdong", drawbounds=True)


ax = plt.subplot(2,2,2)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)

m = Basemap(projection='mill',
            llcrnrlat=np.min(lat_fine)-1, urcrnrlat=np.max(lat_fine)+1,
            llcrnrlon=np.max(lon_fine)-1, urcrnrlon=np.max(lon_fine)+1,
            ax=ax)
x, y = m(grid_lon_fine, grid_lat_fine)
sc = m.pcolormesh(x, y, tair_fine[0] ,vmin=np.min(tair_coarse), vmax=np.max(tair_coarse), cmap='jet')
m.readshapefile("guangdong_shp/guangdong", "guangdong", drawbounds=True)



ax = plt.subplot(2,2,3)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)

m = Basemap(projection='mill',
            llcrnrlat=np.min(lat_fine)-1, urcrnrlat=np.max(lat_fine)+1,
            llcrnrlon=np.max(lon_fine)-1, urcrnrlon=np.max(lon_fine)+1,
            ax=ax)
x, y = m(grid_lon_fine, grid_lat_fine)
sc = m.pcolormesh(x, y, elevation_fine, cmap='jet')
m.readshapefile("guangdong_shp/guangdong", "guangdong", drawbounds=True)





plt.savefig('tair.pdf')

