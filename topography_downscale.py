#-----------------------------------------------------------------------------
# The topography downscale methods for forcing except precipitation
#
# Author: Lu Li
# Reference:
#   Mei et al. (2020): A Nonparametric Statistical Technique for Spatial 
#       Downscaling of Precipitation Over High Mountain Asia, 
#       Water Resourse Research, 56, e2020WR027472.
#   Rouf et al. (2020): A Physically Based Atmospheric Variables Downscaling 
#       Technique. Journal of Hydrometeorology, 21, 93-108.
#-----------------------------------------------------------------------------

import numpy as np
import xesmf as xe
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor



def calc_vapor_pressure(temperature):
    """Calculate vapor pressure according to Buck (1981).
    
    If we input dew temperature, we get actual vapor pressure.
    If we input air temperature, we get saturated vapor pressure.
    
    Reference:
        Buck, 1981: New Equations for Computing Vapor Pressure and
            Enhancement Factor, Journal of Applied Meteorology.
    """
    # constant (for water/ice)
    AW1, BW1, CW1 = 611.21, 17.502, 240.97  
    AW2, BW2, CW2 = 611.21, 17.368, 238.88
    AI,  BI,  CI  = 611.15, 22.452, 272.55
    
    # construct coefficient of Magnus formula
    # If temp>5, adopt ew2 curve;
    # If 5>temp>-5, adopt ew1 curve;
    # If temp<-5, adopt ei2 curve;
    a = np.ones_like(temperature)*AW1
    b = np.ones_like(temperature)*BW1
    c = np.ones_like(temperature)*CW1
    a[temperature>5], b[temperature>5], c[temperature>5] = AW2, BW2, CW2
    a[temperature<-5], b[temperature<-5], c[temperature<-5] = AI, BI, CI
    
    # calculate vapor pressure by Magnus formula
    vapor_pressure = a*np.exp(b*(temperature-273.15)/(temperature-273.15+c))
    return vapor_pressure, a, b, c


def calc_dew_temperature(specific_humidity, 
                         air_pressure, 
                         air_temperature):
    """Calculate dew temperature by vapor pressure according to Lawrence (2005)
    
    Reference:
        Lawrence, 2005: The Relationship between Relative Humidity and the 
            Dewpoint Temperature in Moist Air. BAMS.
    """
    # calculate coefficient of magnus formula by Buck (1981)
    _, a, b, c = calc_vapor_pressure(air_temperature)

    # calculate vapor pressure by pressure and specific humidity
    epsi = 0.62198 # ratio of molecular weight of water and dry air
    vapor_pressure = air_pressure/(epsi+(1-epsi)*specific_humidity)
    
    # calculate dew temperature by vapor pressure by inversing Magnus formula
    dew_temperature = c*np.log(vapor_pressure/a)/ \
        (b-np.log(vapor_pressure/a))+273.15
    return dew_temperature


def calc_lapse_rate(input_coarse, elevation_coarse): 
    """Calculate lapse rate by regional regression method."""
    
    def calc_space_diff(x):
        """calculate difference between center grids with nearby eight grids."""
        center = x[1:-1,1:-1]
        return np.stack([x[1:-1,1:-1]-x[:-2,:-2], x[1:-1,1:-1]-x[:-2,1:-1], 
                         x[1:-1,1:-1]-x[:-2,2:],  x[1:-1,1:-1]-x[1:-1,:-2], 
                         x[1:-1,1:-1]-x[1:-1,2:], x[1:-1,1:-1]-x[2:,:-2], 
                         x[1:-1,1:-1]-x[2:,1:-1], x[1:-1,1:-1]-x[2:,2:]], axis=-1)
    
    def regression(y, x):
        """regression center grids with eight nearby grids"""
        nx, ny, ngrid = x.shape
        ones = np.ones_like(x[0,0])
        lapse_rate = np.full((nx, ny), np.nan)
        for i in range(nx):
            for j in range(ny):
                lapse_rate[i,j] = np.linalg.lstsq(
                    y[i,j], np.stack([x[i,j], ones])[0][0])
        return lapse_rate
    
    # calculate different matrix between center grids with nearby eight grids
    y = calc_space_diff(input_coarse)
    x = calc_space_diff(elevation_coarse)
    
    # calculate laspe rate of input according to elevation
    laspe_rate = regression(y, x)
    
    # enlarge the boundary
    laspe_rate_full = np.zeros_like(x)*np.nan
    laspe_rate_full[1:-1,1:-1] = laspe_rate
    return laspe_rate_full


def calc_clear_sky_emissivity(air_temperature, case='satt'):
    """Calculate emissivity in clear-sky."""
    vapor_pressure, _, _, _ = calc_vapor_pressure(air_temperature)
    if case == 'brut': # Brutsaert (1975)
        return 1.24*(vapor_pressure/air_temperature)**(1/7)
    elif case == 'brut': # Brutsaert (1975)
        return 1.24*(vapor_pressure/air_temperature)**(1/7)
    elif case == 'satt': # Satterlund (1979)
        return 1.08*(1-np.exp(-vapor_pressure**(air_temperature/2016)))
    elif case == 'idso': # Idso (1981)
        return 0.7+5.95e-5*vapor_pressure*np.exp(1500/air_temperature)
    elif case == 'konz': # Konzelmann et al. (1994)
        return 0.23+0.443*(vapor_pressure/air_temperature)**(1/8)
    elif case == 'prat': # Prata (1996) 
        return 1-(1+46.5*vapor_pressure/air_temperature)* \
                    np.exp(-(1.2+3*46.5*vapor_pressure/air_temperature)**5)
    elif case == 'izio': # Iziomon et al. (2003)
        return 1-0.43*np.exp(-11.5*vapor_pressure/air_temperature)
    else:
        print('Please select scheme in satt, konz, brut, izio, prat, idso!')
    

def calc_solar_angles_chelsa(julian_day, hour, latitude):
    """Calculate solar zenith/azimuth angles by CHELSA scheme.
    
    Reference:
        Karger et al. (2023): CHELSA-W5E5: daily 1 km meteorological forcing 
            data for climate impact studies. Earth Syst. Sci. Data, 15, 2445–2464.
    """
    # calculate solar hour angle
    hour_angle = 15*(12-hour-0.5)
    # calculate solar declination angle
    declin_angle = 23.45*np.sin(360*(284+julian_day)/365)
    # calculate sin zenith 
    sin_zenith = np.cos(declin_angle)*np.cos(latitude)*np.cos(hour_angle)+ \
        np.sin(latitude)*np.sin(declin_angle)
    cos_zenith = np.sqrt(1-sin_zenith**2)
    # calcualte cosine azimuth angle
    cos_azimuth = (np.cos(declin_angle)*np.cos(hour_angle)- \
        sin_zenith*np.cos(declin_angle))/ \
        (np.sin(latitude)*cos_zenith)
    return np.arccos(cos_zenith), np.arccos(cos_azimuth)


def calc_solar_angles_wei(julian_day, hour, latitude, longtitude, center_longtitude_time_zone):
    """Calculate solar zenith/azimuth angles by Zhongwang Wei."""
    # Calculate declination
    declination = 0.409 * np.sin((2.0 * np.pi * julian_day / 365.0) - 1.39)
    EOT = (0.258 * np.cos(declination) - 7.416 * np.sin(declination)
           - 3.648 * np.cos(2.0 * declination) - 9.228 * np.sin(2.0 * declination))
    LC = (center_longtitude_time_zone - longtitude) / 15.
    time_corr = (-EOT / 60.) + LC
    solar_time = hour - time_corr

    # Get the hour angle
    w =(solar_time - 12.0) * 15.

    # Get solar elevation angle
    sin_thetha = (np.cos(np.radians(w)) * np.cos(declination) * np.cos(np.radians(latitude))
                  + np.sin(declination) * np.sin(np.radians(latitude)))
    sun_elev = np.arcsin(sin_thetha)

    # Get solar zenith angle
    sza = np.pi / 2.0 - sun_elev
    sza = np.degrees(sza)

    # Get solar azimuth angle
    cos_phi = (np.sin(declination) * np.cos(np.radians(latitude))- np.cos(np.radians(w)) * np.cos(declination) * np.sin(np.radians(latitude)))/ np.cos(sun_elev)
    saa = np.zeros(sza.shape)
    saa[w <= 0.0] = np.degrees(np.arccos(cos_phi[w <= 0.0]))
    saa[w > 0.0] = 360. - np.degrees(np.arccos(cos_phi[w > 0.0]))
    return sza, saa


def calc_solar_angles_colm(julian_day, latitude, longitude):
    """CoLM scheme for calculate cosine zenith angle"""
    # constant
    day_per_year = 365.0
    vernal_equinox = 80.5 # assume Jan 1 is julian day 1
    eccentricity = 1.672393084e-2
    obliquity = 0.409214646
    lambm0 = -3.2625366e-2 # mean long of perihelion at the vernal equinox (radians)
    mvelpp=4.92251015 #moving vernal equinox longitude of perihelion plus pi (radians)

    # Lambda m, mean long of perihelion (rad)
    lambm = lambm0 + (julian_day-vernal_equinox)*2*np.pi/day_per_year
    # intermediate argument involving lambm
    lmm = lambm-mvelpp
    sinl = np.sin(lmm)
    # Lambda, the earths long of perihelion
    lamb = lambm + eccentricity*(2*sinl+ \
        eccentricity*(1.25*np.sin(2*lmm))+ \
            eccentricity*((13.0/12.0)*np.sin(3*lmm)-0.25*sinl))
    # inverse normalized sun/earth distance
    invrho = (1+eccentricity*np.cos(lamb-mvelpp))/(1-eccentricity**2)
    # solar declination (radians)
    declin = np.arcsin(np.sin(obliquity)*np.sin(lamb))
    # earth-sun distance factor (i.e.,(1/r)**2)
    eccf = invrho**2
    # cos zenith
    orb_coszen = np.sin(latitude)*np.sin(declin) - \
        np.cos(latitude)*np.cos(declin)*np.cos(julian_day*2*np.pi+longitude)
    return np.arccos(orb_coszen)


def calc_shadow_mask():
    pass


def downscale_air_temperature(air_temperature_coarse, 
                              elevation_coarse, 
                              elevation_fine,
                              regridder):
    # calculate lapse rate for air temperature
    laspe_rate_coarse = calc_lapse_rate(air_temperature_coarse, elevation_coarse)

    # bilinear interpolate of laspe rate and air temperature and elevation
    air_temperature_fine_interp = regridder(air_temperature_coarse)
    laspe_rate_fine_interp = regridder(laspe_rate_coarse)
    elevation_fine_interp = regridder(elevation_coarse)

    # downscaling
    air_temperature_fine = air_temperature_fine_interp + laspe_rate_fine_interp*(
        elevation_fine-elevation_fine_interp)
    return air_temperature_fine
    

def downscale_dew_temperature(dew_temperature_coarse, 
                              elevation_coarse, 
                              elevation_fine,
                              regridder):
    # calculate lapse rate for dew temperature
    laspe_rate_coarse = calc_lapse_rate(dew_temperature_coarse, elevation_coarse)

    # bilinear interpolate of laspe rate and air temperature and elevation
    dew_temperature_fine_interp = regridder(dew_temperature_coarse)
    laspe_rate_fine_interp = regridder(laspe_rate_coarse)
    elevation_fine_interp = regridder(elevation_coarse)

    # downscaling
    dew_temperature_fine = dew_temperature_fine_interp + laspe_rate_fine_interp*(
        elevation_fine-elevation_fine_interp)
    return dew_temperature_fine
    

def downscale_air_pressure(air_pressure_coarse,
                           air_temperature_coarse,
                           air_temperature_fine,
                           elevation_coarse, 
                           elevation_fine,
                           regridder):
    """The topographic correction method for air pressure. 
    
    It is based on the hydrostatic equation and the Ideal Gas Law 
    
    Reference:
        Cosgrove et al. (2003): Real-time and retrospective forcing in the 
            North American Land Data Assimilation System (NLDAS) project.
            Journal of Geophysical Research, 108.
    """
    # constant
    G = 9.81 # gravitational acceleration [m/s^2]
    R = 287 # ideal gas constant [J/kg*K]

    # bilinear interpolate of air temperature, air pressure and elevation
    air_temperature_fine_interp = regridder(air_temperature_coarse)
    air_pressure_fine_interp = regridder(air_pressure_coarse)
    elevation_fine_interp = regridder(elevation_coarse)

    # downscaling
    air_pressure_fine = air_pressure_fine_interp*np.exp(
        (-G*(elevation_fine-elevation_fine_interp))/(
            R*(air_temperature_fine_interp+air_temperature_fine)/2))
    return air_pressure_fine
    

def downscale_specific_humidity(air_pressure_fine,
                                dew_temperature_fine):
    # calculate vapor pressure
    vapor_pressure_fine, _, _, _ = calc_vapor_pressure(dew_temperature_fine)
    # downscaling
    specific_humidity_fine = (0.622*vapor_pressure_fine)/(
        air_pressure_fine-0.378*vapor_pressure_fine)
    return specific_humidity_fine
    

def downscale_relative_humidity(air_pressure_fine, 
                                dew_temperature_fine, 
                                air_temperature_fine):
    # calculate actual vapor pressure [Pa]
    vapor_pressure_fine, _, _, _ = calc_vapor_pressure(dew_temperature_fine)
    # calculate saturated vapor pressure [Pa]
    s_vapor_pressure_fine, _, _, _ = calc_vapor_pressure(air_temperature_fine)
    # downscaling
    relative_humidity_fine = 100* \
        (vapor_pressure_fine/(air_pressure_fine-vapor_pressure_fine))/ \
            (s_vapor_pressure_fine/(air_pressure_fine-s_vapor_pressure_fine))
    return relative_humidity_fine
    
    
def downscale_in_longwave_radiation(in_longwave_radiation_coarse,
                                    air_temperature_coarse,
                                    air_temperature_fine,
                                    regridder):
    # calculate emissivity of coarse and fine resolution
    emissivity_clear_sky_coarse = calc_clear_sky_emissivity(air_temperature_coarse)
    emissivity_clear_sky_fine = calc_clear_sky_emissivity(air_temperature_fine)

    # calculate all emissivity including both clear-sky and cloudy
    SIGMA = 5.67e-8 # Stefan-Boltzmann constant
    emissivity_all_coarse = in_longwave_radiation_coarse/(SIGMA*air_temperature_coarse**4)

    # calculate cloudy emissivity
    emissivity_cloudy_coarse = emissivity_all_coarse-emissivity_clear_sky_coarse

    # bilinear interpolate longwave radiation, air temperature, cloudy emissivity
    in_longwave_radiation_fine_interp = regridder(in_longwave_radiation_coarse)
    emissivity_cloudy_fine_interp = regridder(emissivity_cloudy_coarse)
    air_temperature_fine_interp = regridder(air_temperature_coarse)

    # calculate all emissivity in fine
    emissivity_all_fine = emissivity_clear_sky_fine+emissivity_cloudy_fine_interp

    # downscaling
    in_longwave_radiation_fine = in_longwave_radiation_fine_interp* \
        (emissivity_all_fine/emissivity_all_coarse)* \
            ((air_temperature_fine/air_temperature_fine_interp)**4)
    return in_longwave_radiation_fine


def downscale_in_shortwave_radiation(in_short_radiation_coarse,
                                     air_pressure_coarse,
                                     air_pressure_fine,
                                     albedo_fine,
                                     slope_fine,
                                     sky_view_factor_fine,
                                     terrain_factor_fine,
                                     aspect_fine,
                                     julian_day,
                                     hour,
                                     latitude,
                                     regridder):
    # ------------------------------------------------------------
    # 1. partition short radiation into beam and diffuse radiation
    # ------------------------------------------------------------
    # calculate solar angles
    zenith_angle_coarse, azimuth_angle_coarse = calc_solar_angles_chelsa(julian_day,hour,latitude)
    
    # calculate top-of-atmosphere incident short radiation
    S = 1370 # solar constant [W/m^2]
    toa_in_short_radiation_coarse = S*np.cos(zenith_angle_coarse)
    
    # calculate clearness index
    clearness_index_coarse = in_short_radiation_coarse/toa_in_short_radiation_coarse
    
    # calculate diffuse weight
    diffuse_weight_coarse = 0.952-1.041*np.exp(-np.exp(2.3-4.702*clearness_index_coarse))
    
    # calculate diffuse and beam raidation
    diffuse_radiation_coarse = in_short_radiation_coarse*diffuse_weight_coarse
    beam_radiation_coarse = in_short_radiation_coarse*(1-diffuse_weight_coarse)
    
    # ------------------------------------------------------------
    # 2. downscaling beam radiation 
    # ------------------------------------------------------------
    # calcualte broadband attenuation coefficient [Pa^-1]
    k_coarse = np.log(clearness_index_coarse)/air_pressure_coarse
    
    # bilinear interpolate k, air pressure
    air_pressure_fine_interp = regridder(air_pressure_coarse)
    k_fine_interp = regridder(k_coarse)
    diffuse_radiation_fine_interp = regridder(diffuse_radiation_coarse)
    beam_radiation_fine_interp = regridder(beam_radiation_coarse)
    # FIXME: I am not sure if we could regridd solar angles,
    #        But Mei did. We actually could use new latitude 
    #        longtitude to re-calculate it.
    zenith_angle_fine_interp = regridder(zenith_angle_coarse)
    azimuth_angle_fine_interp = regridder(azimuth_angle_coarse)

    # calculate factor to account for the difference of 
    # optical path length due to pressure difference
    optical_length_factor_fine = np.exp(k_fine_interp*(
        air_pressure_fine-air_pressure_fine_interp))
    
    # calcualte the cosine of solar illumination angle, cos(θ), 
    # ranging between −1 and 1, indicates if the sun is below or 
    # above the local horizon (note that values lower than 0 are set to 0);
    cos_illumination = np.cos(zenith_angle_fine_interp)*np.cos(slope_fine)+ \
        np.sin(zenith_angle_fine_interp)*np.sin(slope_fine)* \
            np.cos(azimuth_angle_fine_interp-aspect_fine)
    
    # calculate binary shadow mask
    shadow_mask = calc_shadow_mask()
    
    # downscaling beam radiation
    beam_radiation_fine = shadow_mask*cos_illumination* \
        optical_length_factor_fine*beam_radiation_fine_interp
    
    # ------------------------------------------------------------
    # 3. downscaling diffuse radiation 
    # ------------------------------------------------------------ 
    # downscaling diffuse radiation
    diffuse_radiation_fine = sky_view_factor_fine*diffuse_radiation_fine_interp
    
    # ------------------------------------------------------------
    # 4. calculate reflected radiation 
    # ------------------------------------------------------------ 
    reflected_radiation_fine = albedo_fine*terrain_factor_fine* \
        (beam_radiation_fine+(1-sky_view_factor_fine)*diffuse_radiation_fine)
    
    return beam_radiation_fine+ \
           diffuse_radiation_fine+ \
           reflected_radiation_fine


def downscale_wind_speed(u_wind_speed_coarse,
                         v_wind_speed_coarse,
                         slope_fine,
                         aspect_fine,
                         curvature_fine,
                         regridder): 
    # calculate wind direction
    wind_direction_coarse = np.arctanh(v_wind_speed_coarse/u_wind_speed_coarse)
    wind_speed_coarse = np.sqrt(u_wind_speed_coarse**2+v_wind_speed_coarse**2)

    # bilinear interpolate wind speed and direction
    wind_speed_fine_interp = regridder(wind_speed_coarse)
    wind_direction_fine_interp = regridder(wind_direction_coarse)

    # compute the slope in the direction of the wind
    slope_wind_direction_fine = slope_fine*np.cos(wind_direction_fine_interp-aspect_fine)

    # normalize the slope in the direction of the wind into (-0.5,0.5)
    slope_wind_direction_fine = slope_wind_direction_fine/(2*np.max(slope_wind_direction_fine))

    # normalize curvature into (-0.5,0.5)
    curvature_fine = curvature_fine/(2*np.max(curvature_fine))

    # compute wind speed ajustment
    wind_speed_fine = wind_speed_fine_interp* \
        (1+(0.58*slope_wind_direction_fine)+0.42*curvature_fine)
    return wind_speed_fine


def downscale_precipitation_colm(precipitation_coarse,
                                 elevation_coarse,
                                 elevation_fine,
                                 case='tesfa'):
    scale = elevation_fine.shape[0]/elevation_coarse.shape[0]
    precipitation_fine = np.full((elevation_fine.shape[0], 
                                  elevation_fine.shape[1],
                                  precipitation_coarse.shape[-1]), np.nan)
    for i in range(elevation_coarse.shape[0]):
        for j in range(elevation_coarse.shape[-1]):
            zs = elevation_fine[i*scale:(i+1)*scale, j*scale:(j+1)*scale]
            zs_max = np.nanmax(zs)
            pg, zg = precipitation_coarse[i,j], elevation_coarse[i,j]
            if case == 'tesfa':
                # Tesfa et al, 2020: Exploring Topography-Based Methods for Downscaling
                # Subgrid Precipitation for Use in Earth System Models. Equation (5)
                # https://doi.org/ 10.1029/2019JD031456. ERMM methods.
                precipitation_fine[i*scale:(i+1)*scale, j*scale:(j+1)*scale] = \
                    pg*(zs-zg)/zs_max
            elif case == 'liston':
                #Liston, G. E. and Elder, K.: A meteorological distribution system
                # for high-resolution terrestrial modeling (MicroMet), J. Hydrometeorol., 7, 217-234, 2006. Equation (33) and Table 1: chi range from January to December:
                # [0.35,0.35,0.35,0.30,0.25,0.20,0.20,0.20,0.20,0.25,0.30,0.35] (1/m)
                precipitation_fine[i*scale:(i+1)*scale, j*scale:(j+1)*scale] = \
                pg*2.0*0.27e-3*(zs-zg)/(1.0-0.27e-3*(zs-zg))
    return precipitation_fine


def downscale_precipitation_mei(air_temperature_coarse,
                                dew_temperature_coarse,
                                air_pressure_coarse,
                                specific_pressure_coarse,
                                in_longwave_radiation_coarse,
                                in_shortwave_radiation_coarse,
                                wind_speed_coarse,
                                precipitation_coarse,
                                air_temperature_fine,
                                dew_temperature_fine,
                                air_pressure_fine,
                                specific_pressure_fine,
                                in_longwave_radiation_fine,
                                in_shortwave_radiation_fine,
                                wind_speed_fine,
                                julian_day,
                                latitude_coarse,
                                longtitude_coarse,
                                latitude_fine,
                                longtitude_fine
                                #LAI_coarse,
                                #LAI_fine
                                ):
    # calculate precipitation mask in coarse resolution
    precipitation_mask_coarse = np.zeros_like(precipitation_coarse)
    precipitation_mask_coarse[precipitation_coarse>0.1] = 1

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
                  longtitude_coarse], axis=-1)
    x = x.reshape(-1,x.shape[-1]) # reshape for train
    mask = precipitation_mask_coarse.reshape(-1,1)
    value = precipitation_coarse.reshape(-1,1)

    # train classifer
    rf_classifer = RandomForestClassifier(n_estimators=50, min_samples_leaf=10)
    rf_classifer.fit(x, mask)

    # train regressor
    rf_regressor = RandomForestRegressor(n_estimators=50, min_samples_leaf=10)
    rf_regressor.fit(x[mask==1], value[mask==1])

    # downscaling
    x = np.stack([air_temperature_fine,
                  dew_temperature_fine,
                  air_pressure_fine,
                  specific_pressure_fine,
                  in_longwave_radiation_fine,
                  in_shortwave_radiation_fine,
                  wind_speed_fine,
                  julian_day,
                  latitude_fine,
                  longtitude_fine], axis=-1)
    x_fine = x.reshape(-1,x.shape[-1]) # reshape for valid
    mask_fine = rf_classifer(x_fine)
    value_fine = rf_regressor(x_fine)
    precipitation_fine = mask_fine+value_fine
    # FIXME: Maybe this reshape method is wrong, need test
    precipitation_fine = precipitation_fine.reshape(len(latitude_fine),len(longtitude_fine),-1)
    return precipitation_fine





