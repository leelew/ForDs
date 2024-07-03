import argparse

import netCDF4 as nc
import numpy as np

from train_factory import train_rf, train_automl



def load_forcing(year, name, region_name='HH', DATA_PATH="/tera05/lilu/ForDs/data/"):
    output = []
    for i in range(1, 13):
        f = nc.Dataset(DATA_PATH+"forcing/ERA5_{r}_{year:04}_{month:02}_{name}.nc".format(r=region_name, year=year,month=i, name=name),'r')
        output.append(f[name][:])
    output = np.concatenate(output, axis=0)
    return output


def train(year, 
          blat, ulat,
          llon, rlon,
          model_name,
          region_name,
          normalize=False,
          DATA_PATH='/data/lilu/ForDs/data/',
          ):
    print('Train downscaling model in {year:04}'.format(year=year))
    print('The lat range is {blat}-{ulat}'.format(blat=blat,ulat=ulat))
    print('The lon range is {llon}-{rlon}'.format(llon=llon,rlon=rlon))
    print('The authors are Lu Li, SYSU')
    print('\033[1;31m%s\033[0m' % 'The model is {model_name}'.format(model_name=model_name))
    DATA_PATH = DATA_PATH + region_name + '/'

    # load coare DEM data（20x20)
    f = nc.Dataset(DATA_PATH+'DEM/SRTM/ERA5_height.nc', 'r')
    lat_coarse, lon_coarse = f['latitude'][:], f['longitude'][:]
    lat_coarse_index = np.where((lat_coarse>=blat) & (lat_coarse<=ulat))[0]
    lon_coarse_index = np.where((lon_coarse>=llon) & (lon_coarse<=rlon))[0]
    lat_coarse, lon_coarse = lat_coarse[lat_coarse_index], lon_coarse[lon_coarse_index]
    elev_coarse = f['hgt'][lat_coarse_index][:,lon_coarse_index]

    # load forcing data 
    f = nc.Dataset(DATA_PATH+"forcing/ERA5_{r}_{year}_01_t2m.nc".format(r=region_name, year=year),'r')
    lat_coarse, lon_coarse = f['latitude'][:], f['longitude'][:]
    t2m_coarse = load_forcing(year,'t2m',region_name,DATA_PATH)
    d2m_coarse = load_forcing(year,'d2m',region_name,DATA_PATH)
    Q_coarse = load_forcing(year,'q',region_name,DATA_PATH)
    strd_coarse = load_forcing(year,'msdwlwrf',region_name,DATA_PATH) 
    ssrd_coarse = load_forcing(year,'msdwswrf',region_name,DATA_PATH)  
    sp_coarse = load_forcing(year,'sp',region_name,DATA_PATH)
    u10_coarse = load_forcing(year,'u10',region_name,DATA_PATH)
    v10_coarse = load_forcing(year,'v10',region_name,DATA_PATH)
    ws_coarse = np.sqrt(u10_coarse**2+v10_coarse**2)
    tp_coarse = load_forcing(year,'mtpr',region_name,DATA_PATH)*3600  # m/hour -> mm/hour
    
    # standardization
    if normalize:
        t2m_min, t2m_max = np.nanmin(t2m_coarse), np.nanmax(t2m_coarse)
        d2m_min, d2m_max = np.nanmin(d2m_coarse), np.nanmax(d2m_coarse)
        Q_min, Q_max = np.nanmin(Q_coarse), np.nanmax(Q_coarse)
        strd_min, strd_max = np.nanmin(strd_coarse), np.nanmax(strd_coarse)
        ssrd_min, ssrd_max = np.nanmin(ssrd_coarse), np.nanmax(ssrd_coarse)
        sp_min, sp_max = np.nanmin(sp_coarse), np.nanmax(sp_coarse)
        ws_min, ws_max = np.nanmin(ws_coarse), np.nanmax(ws_coarse)
        elev_max = np.nanmax(elev_coarse)
        A = np.array([t2m_min, d2m_min, Q_min, strd_min, ssrd_min, sp_min, ws_min, elev_max])
        B = np.array([t2m_max, d2m_max, Q_max, strd_max, ssrd_max, sp_max, ws_max, elev_max])
        np.save('norm.npy',  np.stack([A,B], axis=-1))

        t2m_coarse = (t2m_coarse-t2m_min)/(t2m_max-t2m_min)
        d2m_coarse = (d2m_coarse-d2m_min)/(d2m_max-d2m_min)
        Q_coarse = (Q_coarse-Q_min)/(Q_max-Q_min)
        strd_coarse = (strd_coarse-strd_min)/(strd_max-strd_min)
        ssrd_coarse = (ssrd_coarse-ssrd_min)/(ssrd_max-ssrd_min)
        sp_coarse = (sp_coarse-sp_min)/(sp_max-sp_min)
        ws_coarse = (ws_coarse-ws_min)/(ws_max-ws_min) 
        elev_coarse = elev_coarse/elev_max
        lat_coarse = lat_coarse/360
        lon_coarse = lon_coarse/360
    
    # train
    if model_name == 'rf':
        train_rf(t2m_coarse,
                 d2m_coarse,
                 sp_coarse,
                 Q_coarse,
                 strd_coarse,
                 ssrd_coarse,
                 ws_coarse,
                 tp_coarse,
                 lat_coarse,
                 lon_coarse,
                 elev_coarse,
                 year,
                 0)
    elif model_name == 'automl':
        train_automl(t2m_coarse,
                      d2m_coarse,
                      sp_coarse,
                      Q_coarse,
                      strd_coarse,
                      ssrd_coarse,
                      ws_coarse,
                      tp_coarse,
                      lat_coarse,
                      lon_coarse,
                      elev_coarse,
                      year,
                      0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2021)
    parser.add_argument('--blat', type=float, default=20.05)
    parser.add_argument('--ulat', type=float, default=29.05)
    parser.add_argument('--llon', type=float, default=108.95)
    parser.add_argument('--rlon', type=float, default=117.95)
    parser.add_argument('--model_name', type=str, default='automl')
    parser.add_argument('--region_name', type=str, default='HH')
    parser.add_argument('--normalize', type=bool, default=False)
    args = parser.parse_args()
    
    train(args.year, 
          args.blat, args.ulat,
          args.llon, args.rlon,
          args.model_name,
          args.region_name,
          args.normalize,
          DATA_PATH='/tera05/lilu/ForDs/data/')
