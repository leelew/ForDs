#-----------------------------------------------------------------------------
# The machine learning downscale methods for precipitation 
#
# Author: Lu Li
# Reference:
#   Mei et al. (2020): A Nonparametric Statistical Technique for Spatial 
#       Downscaling of Precipitation Over High Mountain Asia, 
#       Water Resourse Research, 56, e2020WR027472.
#   Sisi Chen, Lu Li, Yongjiu Dai, et al. Exploring Topography Downscaling 
#       Methods for Hyper-Resolution Land Surface Modeling.
#-----------------------------------------------------------------------------
import cloudpickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from flaml import AutoML


#-----------------------------------------------------------------------------#
# RF from Mei et al. (2022), WRR
#-----------------------------------------------------------------------------#
def train_rf(air_temperature_coarse,
             dew_temperature_coarse,
             air_pressure_coarse,
             specific_pressure_coarse,
             in_longwave_radiation_coarse,
             in_shortwave_radiation_coarse,
             wind_speed_coarse,
             precipitation_coarse,
             latitude_coarse,
             longtitude_coarse,
             elevation_coarse,
             year,
             RainThres=0.1):
    # calculate precipitation mask in coarse resolution 
    precipitation_mask_coarse = np.zeros_like(precipitation_coarse)
    precipitation_mask_coarse[precipitation_coarse>RainThres] = 1
    
    # build julian/lat/lon/dem
    Nt, Nlat, Nlon = air_temperature_coarse.shape
    julian_day = []
    for i in range(int(Nt/24)): # day
        for j in range(24): 
            julian_day.append((i+1))
    julian_day = np.array(julian_day)
    julian_day = np.tile(julian_day[:,np.newaxis,np.newaxis], (1, Nlat, Nlon)) 
    latitude_coarse = np.tile(latitude_coarse[np.newaxis,:,np.newaxis], (julian_day.shape[0], 1, Nlon))
    longtitude_coarse = np.tile(longtitude_coarse[np.newaxis,np.newaxis], (julian_day.shape[0], Nlat, 1))
    elevation_coarse = np.tile(elevation_coarse[np.newaxis], (julian_day.shape[0],1,1))
    
    # construct input features
    x = np.stack([air_temperature_coarse,
                  dew_temperature_coarse,
                  air_pressure_coarse,
                  specific_pressure_coarse,
                  in_longwave_radiation_coarse,
                  in_shortwave_radiation_coarse,
                  wind_speed_coarse,
                  julian_day,
                  latitude_coarse,
                  longtitude_coarse, 
                  elevation_coarse], axis=-1)
    x = x.reshape(-1, x.shape[-1])
    mask = precipitation_mask_coarse.reshape(-1,1)
    value = precipitation_coarse.reshape(-1,1)
    value = np.log10(1+value) 

    # clean np.nan
    mask = np.delete(mask, np.where(np.isnan(x))[0], axis=0)
    value = np.delete(value, np.where(np.isnan(x))[0], axis=0)
    x = np.delete(x, np.where(np.isnan(x))[0], axis=0)
    value = np.delete(value, np.where(np.isnan(mask))[0], axis=0)
    x = np.delete(x, np.where(np.isnan(mask))[0], axis=0)
    mask = np.delete(mask, np.where(np.isnan(mask))[0], axis=0)
    x = np.delete(x, np.where(np.isnan(value))[0], axis=0)
    mask = np.delete(mask, np.where(np.isnan(value))[0], axis=0)
    value = np.delete(value, np.where(np.isnan(value))[0], axis=0)

    # train classifer
    classifer = RandomForestClassifier(n_jobs=-1)
    classifer.fit(x, mask)
    f = open('rf_classifer_{year}.pickle'.format(year=year), 'wb')
    cloudpickle.dump(classifer, f)
    f.close()

    # train regressor
    regressor = RandomForestRegressor(n_jobs=-1)
    regressor.fit(x[np.where(mask==1)[0]], value[np.where(mask==1)[0]])
    f = open('rf_regressor_{year}.pickle'.format(year=year), 'wb')
    cloudpickle.dump(regressor, f)
    f.close()
    

#-----------------------------------------------------------------------------#
# AutoML
#-----------------------------------------------------------------------------#
def train_automl(air_temperature_coarse,
                dew_temperature_coarse,
                air_pressure_coarse,
                specific_pressure_coarse,
                in_longwave_radiation_coarse,
                in_shortwave_radiation_coarse,
                wind_speed_coarse,
                precipitation_coarse,
                latitude_coarse,
                longtitude_coarse,
                elevation_coarse,
                year,
                RainThres=0.1):
    # calculate precipitation mask in coarse resolution 
    precipitation_mask_coarse = np.zeros_like(precipitation_coarse)
    precipitation_mask_coarse[precipitation_coarse>RainThres] = 1
    
    # build julian/lat/lon/dem
    Nt, Nlat, Nlon = air_temperature_coarse.shape
    julian_day = []
    for i in range(int(Nt/24)): # day
        for j in range(24): 
            julian_day.append((i+1))
    julian_day = np.array(julian_day)
    julian_day = np.tile(julian_day[:,np.newaxis,np.newaxis], (1, Nlat, Nlon)) 
    latitude_coarse = np.tile(latitude_coarse[np.newaxis,:,np.newaxis], (julian_day.shape[0], 1, Nlon))
    longtitude_coarse = np.tile(longtitude_coarse[np.newaxis,np.newaxis], (julian_day.shape[0], Nlat, 1))
    elevation_coarse = np.tile(elevation_coarse[np.newaxis], (julian_day.shape[0],1,1))
    
    # construct input features
    x = np.stack([air_temperature_coarse,
                  dew_temperature_coarse,
                  air_pressure_coarse,
                  specific_pressure_coarse,
                  in_longwave_radiation_coarse,
                  in_shortwave_radiation_coarse,
                  wind_speed_coarse,
                  julian_day,
                  latitude_coarse,
                  longtitude_coarse, 
                  elevation_coarse], axis=-1)
    x = x.reshape(-1, x.shape[-1])
    mask = precipitation_mask_coarse.reshape(-1,1)
    value = precipitation_coarse.reshape(-1,1)
    value = np.log10(1+value) 

    # clean np.nan
    mask = np.delete(mask, np.where(np.isnan(x))[0], axis=0)
    value = np.delete(value, np.where(np.isnan(x))[0], axis=0)
    x = np.delete(x, np.where(np.isnan(x))[0], axis=0)
    value = np.delete(value, np.where(np.isnan(mask))[0], axis=0)
    x = np.delete(x, np.where(np.isnan(mask))[0], axis=0)
    mask = np.delete(mask, np.where(np.isnan(mask))[0], axis=0)
    x = np.delete(x, np.where(np.isnan(value))[0], axis=0)
    mask = np.delete(mask, np.where(np.isnan(value))[0], axis=0)
    value = np.delete(value, np.where(np.isnan(value))[0], axis=0)
    
    # train classifer
    automl = AutoML()
    automl.fit(x, 
               mask, 
               estimator_list='auto',
               task='classification',
               metric='accuracy',
               split_ratio=0.2,
               time_budget=500,
               n_jobs=-1)
    am_output = open("automl_cls_{year}.pkl".format(year=year), 'wb')
    cloudpickle.dump(automl, am_output)
    am_output.close()

    # train regressor
    am = AutoML()
    am.fit(x[np.where(mask==1)[0]],
           value[np.where(mask==1)[0]],
           estimator_list='auto',
           task='regression',
           metric='mse',
           split_ratio=0.2,
           time_budget=500,
           n_jobs=-1)
    am_output = open("automl_reg_{year}.pkl".format(year=year), 'wb')
    cloudpickle.dump(am, am_output)
    am_output.close()