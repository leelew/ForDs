import argparse

import netCDF4 as nc
import numpy as np

from train_factory import train_mtl_dnn, train_stl_dnn, train_rf



def load_forcing(year, name, DATA_PATH="/tera06/lilu/ForDs/data/GD/"):
    output = []
    for i in range(1, 13):
        f = nc.Dataset(DATA_PATH+"forcing/ERA5LAND_GD_{year:04}_{month:02}_{name}.nc".format(
            year=year,month=i, name=name),'r')
        output.append(f[name][:])
    output = np.concatenate(output, axis=0)
    return output

# Only for Guangdong test
def train(year, 
          blat, ulat,
          llon, rlon,
          model_name,
          DATA_PATH='/tera06/lilu/ForDs/data/GD/',
          OUT_PATH='/tera03/lilu/'):
    print('Train downscaling model in {year:04}'.format(year=year))
    print('The lat range is {blat}-{ulat}'.format(blat=blat,ulat=ulat))
    print('The lon range is {llon}-{rlon}'.format(llon=llon,rlon=rlon))
    print('The authors are Lu Li, SYSU')
    print('\033[1;31m%s\033[0m' % 'The model is {model_name}'.format(model_name=model_name))

    # load coare DEM data（20x20)
    f = nc.Dataset(DATA_PATH+'DEM/SRTM/ERA5Land_height.nc', 'r')
    lat_coarse, lon_coarse = f['latitude'][:], f['longitude'][:]
    lat_coarse_index = np.where((lat_coarse>blat) & (lat_coarse<ulat))[0]
    lon_coarse_index = np.where((lon_coarse>llon) & (lon_coarse<rlon))[0]
    lat_coarse, lon_coarse = lat_coarse[lat_coarse_index], lon_coarse[lon_coarse_index]
    elev_coarse = f['z'][0][lat_coarse_index][:,lon_coarse_index]/9.8  

    # load forcing data 
    f = nc.Dataset(DATA_PATH+"forcing/ERA5LAND_GD_2018_01_t2m.nc",'r')
    lat_coarse, lon_coarse = f['latitude'][:], f['longitude'][:]
    t2m_coarse = load_forcing(year,'t2m')
    d2m_coarse = load_forcing(year,'d2m')
    Q_coarse = load_forcing(year, 'Q')
    Q_coarse[Q_coarse>100] = np.nan
    strd_coarse = load_forcing(year, 'strd') 
    ssrd_coarse = load_forcing(year, 'ssrd')  
    sp_coarse = load_forcing(year, 'sp')
    u10_coarse = load_forcing(year, 'u10')
    v10_coarse = load_forcing(year, 'v10')
    ws_coarse = np.sqrt(u10_coarse**2+v10_coarse**2)
    tp_coarse = load_forcing(year, 'tp')*1000  # m/hour -> mm/hour

    """
    # standardization
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
    np.save('norm_params.npy',  np.stack([A,B], axis=-1))

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
    """
    
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
                 0.1)
    elif model_name == 'mtl_dnn':
        train_mtl_dnn(t2m_coarse,
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
                      0.1)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Defaut regions constarined by Guangdong
    parser.add_argument('--year', type=int, default=2018)
    parser.add_argument('--blat', type=float, default=20.05)
    parser.add_argument('--ulat', type=float, default=29.05)
    parser.add_argument('--llon', type=float, default=108.95)
    parser.add_argument('--rlon', type=float, default=117.95)
    parser.add_argument('--model_name', type=str, default='rf')
    args = parser.parse_args()
    
    train(args.year, 
          args.blat, args.ulat,
          args.llon, args.rlon,
          args.model_name,
          DATA_PATH='/tera06/lilu/ForDs/data/GD/',
          OUT_PATH='/tera03/lilu/')
