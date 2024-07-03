#-----------------------------------------------------------------------------
# The topography downscaling methods for atmospheric forcing 
#
# Author: Lu Li, Sisi Chen
# Reference:
#   Mei et al. (2020): A Nonparametric Statistical Technique for Spatial 
#       Downscaling of Precipitation Over High Mountain Asia, 
#       Water Resourse Research, 56, e2020WR027472.
#   Rouf et al. (2020): A Physically Based Atmospheric Variables Downscaling 
#       Technique. Journal of Hydrometeorology, 21, 93-108.
#   Sisi Chen, Lu Li, Yongjiu Dai, et al. Exploring Topography Downscaling 
#       Methods for Hyper-Resolution Land Surface Modeling. 
#-----------------------------------------------------------------------------
import numpy as np
from sklearn.linear_model import LinearRegression

from utils import bilInterp



def date2jd(year, month, day_of_month):
    if (year%4==0 and year%100!=0) or (year%400==0):
        tmp = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        tmp = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    jday = 0
    for i in range(month-1):
        jday =jday+tmp[i]

    jday = jday+day_of_month
    return jday


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
    """Calculate dew temperature by air pressure and 
       specific humidity according to Lawrence (2005)
    
    Reference:
        Lawrence, 2005: The Relationship between Relative Humidity 
            and the Dewpoint Temperature in Moist Air. BAMS.
    """
    # calculate coefficient of magnus formula by Buck (1981)
    _, a, b, c = calc_vapor_pressure(air_temperature)

    # calculate vapor pressure by pressure and specific humidity
    epsi = 0.62198 # ratio of molecular weight of water and dry air
    vapor_pressure = specific_humidity*air_pressure/(epsi+(1-epsi)*specific_humidity)
    
    # calculate dew temperature by vapor pressure by inversing Magnus formula
    dew_temperature = c*np.log(vapor_pressure/a)/ \
        (b-np.log(vapor_pressure/a))+273.15
    return dew_temperature


def calc_lapse_rate(input_coarse, elevation_coarse): 
    """Calculate lapse rate by regional regression method.

    NOTE: This func received matrix that time dimension lastly,
          and output matrix that time dimension firstly
    """
     
    def calc_space_diff(x):
        """calculate difference between center grids with nearby eight grids."""
        center = x[1:-1,1:-1]
        return np.stack([x[1:-1,1:-1]-x[:-2,:-2], x[1:-1,1:-1]-x[:-2,1:-1], 
                         x[1:-1,1:-1]-x[:-2,2:],  x[1:-1,1:-1]-x[1:-1,:-2], 
                         x[1:-1,1:-1]-x[1:-1,2:], x[1:-1,1:-1]-x[2:,:-2], 
                         x[1:-1,1:-1]-x[2:,1:-1], x[1:-1,1:-1]-x[2:,2:]], axis=0)
    
    def regression(y, x):
        """regression center grids with eight nearby grids"""
        ngrid, nx, ny, nt = y.shape
        ones = np.ones_like(x[:,0,0])
        coef = np.full((nx, ny, nt), np.nan)
        for i in range(nx):
            for j in range(ny):
                if (np.isnan(x[:,i,j]).any()) or (np.isnan(y[:,i,j]).any()):
                    pass
                else:
                    reg = LinearRegression().fit(np.stack([x[:,i,j], ones],axis=-1),y[:,i,j])
                    coef[i,j] = reg.coef_[:,0]
        return coef
    
    # calculate different matrix between center grids with nearby eight grids
    y = calc_space_diff(input_coarse)
    x = calc_space_diff(elevation_coarse)
    
    # calculate laspe rate of input according to elevation
    try:
        laspe_rate = regression(y, x)
    except:
        laspe_rate = np.ones_like(y[0])*0.006

    # control the boundary of lapse rate
    for i in range(laspe_rate.shape[-1]):
        tmp = laspe_rate[:,:,i]
        up_bound, low_bound = np.nanquantile(tmp, 0.95), np.nanquantile(tmp, 0.05)
        tmp[tmp>up_bound] = up_bound
        tmp[tmp<low_bound] = low_bound
        laspe_rate[:,:,i] = tmp

    # enlarge the boundary
    laspe_rate_full = np.ones_like(input_coarse)*np.nanmedian(laspe_rate)
    laspe_rate_full[1:-1,1:-1] = laspe_rate
    return np.transpose(laspe_rate_full,(2,0,1)) # time first


def calc_clear_sky_emissivity(air_temperature, dew_temperature, case='konz'):
    """Calculate emissivity in clear-sky by vapor pressure and air temperature."""
    vapor_pressure, _, _, _ = calc_vapor_pressure(dew_temperature)
    if case == 'brut': # Brutsaert (1975)
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
        print('Please select schemes in satt, konz, brut, izio, prat, idso!')
    

def downscale_air_temperature(air_temperature_coarse,
                              air_temperature_fine_interp, 
                              elevation_coarse,
                              elevation_fine_interp, 
                              elevation_fine,
                              lat_coarse,
                              lon_coarse,
                              year, 
                              month, 
                              day_of_month):
    # calculate lapse rate for air temperature
    laspe_rate_coarse = calc_lapse_rate(
        np.transpose(air_temperature_coarse, (1,2,0)), elevation_coarse)
    # bilinear interpolate of laspe rate 
    laspe_rate_fine_interp = bilInterp('laspe_rate', 
                                        lat_coarse, 
                                        lon_coarse, 
                                        laspe_rate_coarse, 
                                        year,
                                        month,
                                        day_of_month)
    # downscaling
    dz = (elevation_fine-elevation_fine_interp)[np.newaxis]
    air_temperature_fine = air_temperature_fine_interp + np.multiply(
            laspe_rate_fine_interp,dz)
    return np.array(air_temperature_fine)
    

def downscale_dew_temperature(dew_temperature_coarse, 
                              dew_temperature_fine_interp, 
                              elevation_coarse,
                              elevation_fine_interp, 
                              elevation_fine,
                              lat_coarse,
                              lon_coarse, 
                              year, 
                              month, 
                              day_of_month):
    
    # calculate lapse rate for dew temperature
    laspe_rate_coarse = calc_lapse_rate(
        np.transpose(dew_temperature_coarse, (1,2,0)), elevation_coarse)
    # bilinear interpolate of laspe rate
    laspe_rate_fine_interp = bilInterp('laspe_rate', 
                                        lat_coarse, 
                                        lon_coarse, 
                                        laspe_rate_coarse,
                                        year,
                                        month,
                                        day_of_month)
    # downscaling
    dz = (elevation_fine-elevation_fine_interp)[np.newaxis]
    dew_temperature_fine = dew_temperature_fine_interp + np.multiply(
            laspe_rate_fine_interp, dz)
    return np.array(dew_temperature_fine)
    

def downscale_air_pressure(air_pressure_fine_interp,
                           air_temperature_fine_interp,
                           air_temperature_fine, 
                           elevation_fine_interp,
                           elevation_fine):
    """The topographic correction method for air pressure. 
    
    It is based on the hydrostatic equation and the Ideal Gas Law 
    
    Reference:
        Cosgrove et al. (2003): Real-time and retrospective forcing in the 
            North American Land Data Assimilation System (NLDAS) project.
            Journal of Geophysical Research.
    """
    # constant
    G = 9.81 # gravitational acceleration [m/s^2]
    R = 287 # ideal gas constant [J/kg*K]

    # downscaling
    air_pressure_fine = air_pressure_fine_interp* \
                        np.exp((-G*(elevation_fine-elevation_fine_interp))/ \
                        (R*(air_temperature_fine_interp+air_temperature_fine)/2))
    return np.array(air_pressure_fine)
    

def downscale_specific_humidity(air_pressure_fine,
                                dew_temperature_fine):
    # calculate vapor pressure
    vapor_pressure_fine, _, _, _ = calc_vapor_pressure(dew_temperature_fine)
    # downscaling
    specific_humidity_fine = (0.622*vapor_pressure_fine)/ \
        (air_pressure_fine-0.378*vapor_pressure_fine)
    # constrain bound
    specific_humidity_fine[specific_humidity_fine<0] = 0
    specific_humidity_fine[specific_humidity_fine>0.1] = 0.1    
    return np.array(specific_humidity_fine)


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
    return  np.array(relative_humidity_fine)
    
    
def downscale_in_longwave_radiation(in_longwave_radiation_coarse,
                                    in_longwave_radiation_fine_interp,
                                    air_temperature_coarse,
                                    dew_temperature_coarse,
                                    air_temperature_fine,
                                    dew_temperature_fine,
                                    air_temperature_fine_interp,
                                    lat_coarse,
                                    lon_coarse, 
                                    year, 
                                    month, 
                                    day_of_month):
    # calculate emissivity of coarse and fine resolution
    emissivity_clear_sky_coarse = calc_clear_sky_emissivity(air_temperature_coarse, dew_temperature_coarse)
    emissivity_clear_sky_fine = calc_clear_sky_emissivity(air_temperature_fine, dew_temperature_fine)

    # calculate all emissivity including both clear-sky and cloudy
    SIGMA = 5.67e-8 # Stefan-Boltzmann constant
    emissivity_all_coarse = in_longwave_radiation_coarse/(SIGMA*air_temperature_coarse**4)

    # calculate cloudy emissivity
    emissivity_cloudy_coarse = emissivity_all_coarse-emissivity_clear_sky_coarse

    # bilinear interpolate cloudy emissivity
    emissivity_cloudy_fine_interp = bilInterp('emissivity_cloudy',
                                              lat_coarse,
                                              lon_coarse,
                                              emissivity_cloudy_coarse,
                                              year, 
                                              month, 
                                              day_of_month)
    emissivity_all_fine_interp = bilInterp('emissivity_all',
                                            lat_coarse,
                                            lon_coarse,
                                            emissivity_all_coarse,
                                            year, 
                                            month, 
                                            day_of_month)
    # calculate all emissivity in fine
    emissivity_all_fine = np.array(emissivity_clear_sky_fine+emissivity_cloudy_fine_interp)
    
    # downscaling
    in_longwave_radiation_fine = in_longwave_radiation_fine_interp* \
        (emissivity_all_fine/emissivity_all_fine_interp)* \
            ((air_temperature_fine/air_temperature_fine_interp)**4)
    return np.array(in_longwave_radiation_fine)


def downscale_wind_speed(u_wind_speed_fine_interp,
                         v_wind_speed_fine_interp,
                         slope_fine,
                         aspect_fine,
                         curvature_fine): 
    # calculate wind direction
    wind_direction_fine_interp = np.arctan(v_wind_speed_fine_interp/u_wind_speed_fine_interp)
    wind_speed_fine_interp = np.sqrt(u_wind_speed_fine_interp**2+v_wind_speed_fine_interp**2)

    # compute the slope in the direction of the wind
    slope_wind_direction_fine = slope_fine[np.newaxis]*np.cos(wind_direction_fine_interp-aspect_fine[np.newaxis])

    # normalize the slope in the direction of the wind into (-0.5,0.5)
    slope_wind_direction_fine = slope_wind_direction_fine/(2*np.max(slope_wind_direction_fine))

    # normalize curvature into (-0.5,0.5)
    curvature_fine = curvature_fine/(2*np.max(curvature_fine))

    # compute wind speed ajustment
    wind_speed_fine = wind_speed_fine_interp* \
        (1+(0.58*slope_wind_direction_fine)+0.42*curvature_fine[np.newaxis])
    return np.array(wind_speed_fine)


def calc_solar_angles(julian_day, hour, latitude, longitude):
    """Calculate solar zenith/azimuth angles by CHELSA scheme.
    
    Reference:
        Karger et al. (2023): CHELSA-W5E5: daily 1 km meteorological forcing 
            data for climate impact studies. Earth Syst. Sci. Data.
    """
    # calculate solar hour angle (deg)
    # Calculate 均时差E_qt, derived from SZA func in R Atmosphere library 
    # @Sisi Chen, 2023-09-18
    if julian_day <= 106 :
        E_qt = -14.2 * np.sin(np.pi*(julian_day + 7)/111)
    elif julian_day <= 166:
        E_qt = 4 * np.sin(np.pi * (julian_day - 106)/59)
    elif julian_day <= 246:
        E_qt = -6.5 * np.sin(np.pi * (julian_day - 166)/80)
    else:
        E_qt = 16.4 * np.sin(np.pi * (julian_day - 247)/113)
    lon, lat = np.meshgrid(longitude,latitude)
    hour_angle = 15*(12-(hour+E_qt/60+lon/15))
   
    # calculate solar declination angle (deg)
    declin_angle = 23.45*np.sin(np.deg2rad(360*(284+julian_day)/365))
    
    # calculate sin zenith (deg)
    cos_zenith = np.cos(np.deg2rad(declin_angle))* \
                 np.cos(np.deg2rad(lat))* \
                 np.cos(np.deg2rad(hour_angle))+ \
                 np.sin(np.deg2rad(lat))* \
                 np.sin(np.deg2rad(declin_angle))
    sin_zenith = np.sqrt(1-cos_zenith**2)
    
    # calcualte cosine azimuth angle
    cos_azimuth = (np.cos(np.deg2rad(declin_angle))* \
                   np.cos(np.deg2rad(hour_angle))- \
                   cos_zenith*np.cos(np.deg2rad(lat)))/ \
                   (np.sin(np.deg2rad(lat))*sin_zenith)
    return np.arccos(cos_zenith), np.arccos(cos_azimuth) # (rad)


def diff_rad_adjust(sky_view_factor_fine, 
                    diffuse_radiation_fine_interp, 
                    cos_illumination, 
                    beam_radiation_fine,
                    toa_in_short_radiation_fine_interp,
                    slope_fine, 
                    beam_radiation_fine_interp,
                    scheme=1):
    if scheme == 1:
        # isotropic scattering
        return sky_view_factor_fine*diffuse_radiation_fine_interp
    elif scheme == 2:
        # consider antisotropic scattering. 
        # 
        # Huang et al., (2022). Development of a clear‐sky 3D sub‐grid terrain 
        #   solar radiative effect parameterization scheme based on the mountain radiation theory. 
        #   Journal of Geophysical Research: Atmospheres, 127(13), e2022JD036449. equation (4)
        # TODO: Need reframe variable names @Chen
        diffuse_radiation_fine = diffuse_radiation_fine_interp
        for i in range(24):
            idx1 = np.where(cos_illumination[i]>0)
            ratio1 = beam_radiation_fine[i][idx1]/toa_in_short_radiation_fine_interp[i][idx1]
            ratio2 = beam_radiation_fine_interp[i][idx1]/toa_in_short_radiation_fine_interp[i][idx1]
            idx3 = np.where((ratio1>1)|(ratio2>1))
            ratio1[idx3] = 0
            ratio2[idx3] = 0
            ratio1[ratio1<0] = 0
            ratio2[ratio2<0] = 0
            diffuse_radiation_fine[i][idx1] = diffuse_radiation_fine_interp[i][idx1]*(ratio1+\
                                0.5*sky_view_factor_fine[idx1]*(1+np.cos(slope_fine[idx1]))*(1-ratio2))
            idx2 = np.where(cos_illumination[i]<=0)
            ratio3 = beam_radiation_fine_interp[i][idx2]/toa_in_short_radiation_fine_interp[i][idx2]
            idx4 = np.where(ratio3>1)
            ratio3[idx4] = 0
            ratio3[ratio3<0] = 0
            diffuse_radiation_fine[i][idx2] = diffuse_radiation_fine_interp[i][idx2]*(
                0.5*sky_view_factor_fine[idx2]*(1+np.cos(slope_fine[idx2]))*(1-ratio3))
        return diffuse_radiation_fine


def cal_sf(azimuth_angle_fine, zenith_angle_fine, sf_lut, sf_lut_f, sf_lut_b, scheme=1):
    shadow_mask = np.zeros_like(azimuth_angle_fine)
    idx = np.array(np.int64(azimuth_angle_fine/1.5)) # (24,lat,lon)
    for i in range(24):
        for n in range(idx.shape[1]):
            for m in range(idx.shape[2]):
                if scheme == 1:
                    theta1=np.arcsin(sf_lut_f[n,m,idx[i,n,m]])
                    theta2=np.arcsin(sf_lut_b[n,m,idx[i,n,m]])
                    if np.pi/2-zenith_angle_fine[i,n,m] < theta2:
                        shadow_mask[i,n,m] = 0
                    elif np.pi/2-zenith_angle_fine[i,n,m] > theta1:
                        shadow_mask[i,n,m] =1
                    else:
                        shadow_mask[i,n,m] = ((np.pi/2-zenith_angle_fine[i,n,m])-theta2)/(theta1-theta2)
                else:
                    if sf_lut[n,m,idx[i,n,m]] < np.cos(zenith_angle_fine[i,n,m]):
                        shadow_mask[i,n,m] = 1
                    else:
                        shadow_mask[i,n,m] = 0   
    return shadow_mask
                

def downscale_in_shortwave_radiation(in_short_radiation_coarse,
                                     air_pressure_coarse,
                                     air_pressure_fine,
                                     air_pressure_fine_interp,
                                     black_sky_albedo,
                                     white_sky_albedo,
                                     slope_fine,
                                     aspect_fine,
                                     sky_view_factor_fine,
                                     year, 
                                     month, 
                                     day_of_month, 
                                     lat_coarse,
                                     lon_coarse,
                                     lat_fine,
                                     lon_fine,
                                     sf_lut,
                                     sf_lut_f,
                                     sf_lut_b,
                                     shadow_mask_scheme=1,
                                     diff_rad_adjust_scheme=1):
    # ------------------------------------------------------------
    # 1. partition short radiation into beam and diffuse radiation
    # ------------------------------------------------------------
    julian_day = date2jd(year, month, day_of_month)

    # calculate solar angles (rad)
    zenith_angle_coarse = np.zeros_like(in_short_radiation_coarse)*np.nan
    azimuth_angle_coarse = np.zeros_like(in_short_radiation_coarse)*np.nan
    zenith_angle_fine = np.zeros_like(air_pressure_fine)*np.nan
    azimuth_angle_fine = np.zeros_like(air_pressure_fine)*np.nan
    for i in range(24):
        zenith_angle_coarse[i], azimuth_angle_coarse[i] = calc_solar_angles(julian_day,i,lat_coarse,lon_coarse)
        zenith_angle_fine[i], azimuth_angle_fine[i] = calc_solar_angles(julian_day,i,lat_fine,lon_fine)
        
    # calculate binary shadow mask
    # NOTE(@Li): We did not calculate shadow mask for each timestep
    #       Instead, we calculate it by LUT. See Huang, JAMES.
    # NOTE(@Chen): We trans azimuth angle from rad to degree
    #       because the index mostly equal to 0 if we 
    #       use rad (i.e., azimuth/1.5<1). 
    azimuth_angle_fine = azimuth_angle_fine*180/np.pi # turn deg
    shadow_mask = cal_sf(azimuth_angle_fine, zenith_angle_fine, sf_lut, sf_lut_f, sf_lut_b, scheme=shadow_mask_scheme)
    azimuth_angle_fine = azimuth_angle_fine*np.pi/180 # turn rad
        
    # calculate top-of-atmosphere incident short radiation
    S = 1370 # solar constant [W/m^2]
    rt_R = 1-0.01672*np.cos(0.9856*(julian_day-4))
    toa_in_short_radiation_coarse = S*(rt_R**2)*np.cos(zenith_angle_coarse)

    # calculate clearness index
    clearness_index_coarse = in_short_radiation_coarse/toa_in_short_radiation_coarse
    clearness_index_coarse[clearness_index_coarse>1] = 1

    # calculate diffuse weight
    diffuse_weight_coarse = 0.952-1.041*np.exp(-np.exp(2.3-4.702*clearness_index_coarse))
    diffuse_weight_coarse[diffuse_weight_coarse>1] = 1
    diffuse_weight_coarse[diffuse_weight_coarse<0] = 0

    # calculate diffuse and beam raidation
    diffuse_radiation_coarse = in_short_radiation_coarse*diffuse_weight_coarse
    beam_radiation_coarse = in_short_radiation_coarse*(1-diffuse_weight_coarse)

    # ------------------------------------------------------------
    # 2. downscaling beam radiation 
    # ------------------------------------------------------------
    # calcualte broadband attenuation coefficient [Pa^-1]
    k_coarse = np.log(clearness_index_coarse)/air_pressure_coarse
    
    # bilinear interpolate k, air pressure
    k_fine_interp = bilInterp('k',
                            lat_coarse,
                            lon_coarse,
                            k_coarse,
                            year, 
                            month, 
                            day_of_month)
    diffuse_radiation_fine_interp = bilInterp('diffuse_radiation',
                                                lat_coarse,
                                                lon_coarse,
                                                diffuse_radiation_coarse,
                                                year, 
                                                month, 
                                                day_of_month)    
    beam_radiation_fine_interp = bilInterp('beam_radiation',
                                            lat_coarse,
                                            lon_coarse,
                                            beam_radiation_coarse,
                                            year, 
                                            month, 
                                            day_of_month)
    diffuse_weight_fine_interp = bilInterp('diffuse_weight',
                                            lat_coarse,
                                            lon_coarse,
                                            diffuse_weight_coarse,
                                            year, 
                                            month, 
                                            day_of_month)  
    toa_in_short_radiation_fine_interp = bilInterp('toa_in_short_radiation',
                                                   lat_coarse, 
                                                   lon_coarse,
                                                   toa_in_short_radiation_coarse,
                                                   year,
                                                   month,
                                                   day_of_month)
    
    # calculate factor to account for the difference of 
    # optical path length due to pressure difference
    optical_length_factor_fine = np.exp(k_fine_interp*(
        air_pressure_fine-air_pressure_fine_interp))
    optical_length_factor_fine[optical_length_factor_fine>10000] = 0

    # adjust zeith 
    thr = (85*2*np.pi)/90
    zenith_angle_fine[zenith_angle_fine>thr] = thr
    
    # calcualte the cosine of solar illumination angle, cos(θ), 
    # ranging between −1 and 1, indicates if the sun is below or 
    # above the local horizon (note that values lower than 0 are set to 0);
    cos_illumination = np.cos(slope_fine)+ \
        np.tan(zenith_angle_fine)*np.sin(slope_fine)* \
            np.cos(azimuth_angle_fine-aspect_fine)
    cos_illumination[:,np.cos(slope_fine)==1] = 1
    cos_illumination[cos_illumination>1] = 1
    cos_illumination[cos_illumination<0] = 0

    # downscaling beam radiation
    beam_radiation_fine_interp[beam_radiation_fine_interp<0] = np.nan
    beam_radiation_fine = shadow_mask*cos_illumination*beam_radiation_fine_interp*optical_length_factor_fine

    # ------------------------------------------------------------
    # 3. downscaling diffuse radiation 
    # ------------------------------------------------------------ 
    # downscaling diffuse radiation
    sky_view_factor_fine[sky_view_factor_fine>1] = 1
    sky_view_factor_fine[sky_view_factor_fine<0] = 0
    diffuse_radiation_fine_interp[diffuse_radiation_fine_interp<0] = np.nan
    diffuse_radiation_fine = diff_rad_adjust(sky_view_factor_fine, 
                                             diffuse_radiation_fine_interp, 
                                             cos_illumination, 
                                             beam_radiation_fine,
                                             toa_in_short_radiation_fine_interp,
                                             slope_fine, 
                                             beam_radiation_fine_interp,
                                             diff_rad_adjust_scheme)
    
    # ------------------------------------------------------------
    # 4. calculate reflected radiation 
    # ------------------------------------------------------------ 
    albedo_fine = black_sky_albedo*(1-diffuse_weight_fine_interp)+white_sky_albedo*(diffuse_weight_fine_interp)
    terrain_factor_fine = ((1+np.cos(slope_fine))/2)-sky_view_factor_fine
    if len(terrain_factor_fine[terrain_factor_fine<0]) != 0:
        terrain_factor_fine = terrain_factor_fine - 1.00001*np.nanmin(terrain_factor_fine[terrain_factor_fine<0])
    reflected_radiation_fine = albedo_fine*terrain_factor_fine* \
        (beam_radiation_fine*np.cos(zenith_angle_fine)+(1-sky_view_factor_fine)*diffuse_radiation_fine)
    
    # for diagnose
    """ 
    np.save('in_short_radiation_coarse_{month}_{day}'.format(month=month, day=day_of_month), np.array(in_short_radiation_coarse))
    np.save('beam_radiation_coarse_{month}_{day}'.format(month=month, day=day_of_month), np.array(beam_radiation_coarse))
    np.save('diffuse_radiation_coarse_{month}_{day}'.format(month=month, day=day_of_month), np.array(diffuse_radiation_coarse))
    np.save('beam_radiation_fine_interp_{month}_{day}'.format(month=month, day=day_of_month), np.array(beam_radiation_fine_interp))
    np.save('diffuse_radiation_fine_interp_{month}_{day}'.format(month=month, day=day_of_month), np.array(diffuse_radiation_fine_interp))
    np.save('cos_illumination_{month}_{day}'.format(month=month, day=day_of_month), np.array(cos_illumination))
    np.save('shadow_mask_{month}_{day}'.format(month=month, day=day_of_month), np.array(shadow_mask))
    np.save('optical_length_factor_fine_{month}_{day}'.format(month=month, day=day_of_month), np.array(optical_length_factor_fine))
    np.save('diffuse_radiation_fine_{month}_{day}'.format(month=month, day=day_of_month), np.array(diffuse_radiation_fine))
    np.save('reflected_radiation_fine_{month}_{day}'.format(month=month, day=day_of_month), np.array(reflected_radiation_fine))
    """

    return np.array(cos_illumination)*np.array(shadow_mask)*np.array(beam_radiation_fine_interp)*np.array(optical_length_factor_fine)+\
        np.array(diffuse_radiation_fine)+np.array(reflected_radiation_fine)


def downscale_precipitation_colm(precipitation_coarse,
                                 elevation_coarse,
                                 elevation_fine):
     # NOTE: only for square inputs
    scale = round(elevation_fine.shape[0]/elevation_coarse.shape[0])
    nt, nlat, nlon = precipitation_coarse.shape
    precipitation_fine = np.full((nt, elevation_fine.shape[0],elevation_fine.shape[1]), np.nan)

    for i in range(nlat):
        for j in range(nlon):
            # handle the boundary problem caused by Not-INT scale 
            if (i != nlat-1) & (j != nlon-1):
                zs = elevation_fine[i*scale:(i+1)*scale, j*scale:(j+1)*scale]
            elif (i == nlat-1) & (j != nlon-1):
                zs = elevation_fine[i*scale:, j*scale:(j+1)*scale]
            elif (i != nlat-1) & (j == nlon-1):
                zs = elevation_fine[i*scale:(i+1)*scale, j*scale:]
            else:
                zs = elevation_fine[i*scale:, j*scale:]

            # calculate precip
            pg, zg = precipitation_coarse[:,i:i+1,j:j+1], elevation_coarse[i,j]

            # Subgrid Precipitation for Use in Earth System Models. Equation (5).
            zs_max = np.nanmax(zs)
            tmp = pg*(1+(zs-zg)/zs_max)

            # rewrite in outfile
            if (i != nlat-1) & (j != nlon-1):
                precipitation_fine[:, i*scale:(i+1)*scale, j*scale:(j+1)*scale] = tmp
            elif (i == nlat-1) & (j != nlon-1):
                precipitation_fine[:, i*scale:, j*scale:(j+1)*scale] = tmp
            elif (i != nlat-1) & (j == nlon-1):
                precipitation_fine[:, i*scale:(i+1)*scale, j*scale:] = tmp
            else:
                precipitation_fine[:, i*scale:, j*scale:] = tmp
    return precipitation_fine
   

def downscale_precipitation(
                            air_temperature_fine,
                            dew_temperature_fine,
                            air_pressure_fine,
                            specific_pressure_fine,
                            in_longwave_radiation_fine,
                            in_shortwave_radiation_fine,
                            wind_speed_fine,
                            lat_fine,
                            lon_fine,
                            elev_fine,
                            year, 
                            month, 
                            day_of_month):
    """downscale precipitation for each day"""
    # get shape
    nt, nx, ny = air_temperature_fine.shape

    # auxillary
    julian_day = []
    jday0 = date2jd(year, month, day_of_month) # day of month must plus 1 to start from 1
    for i in range(24):
        julian_day.append(jday0)
    julian_day = np.array(julian_day)   
    julian_day = np.tile(julian_day[:,np.newaxis,np.newaxis], (1, nx, ny))  
    lat_fine = np.tile(lat_fine[np.newaxis,:,np.newaxis], (julian_day.shape[0], 1, ny))
    lon_fine = np.tile(lon_fine[np.newaxis,np.newaxis], (julian_day.shape[0], nx, 1))
    elev_fine = np.tile(elev_fine[np.newaxis], (julian_day.shape[0],1,1))

    # downscaling
    x = np.stack([air_temperature_fine,
                  dew_temperature_fine,
                  air_pressure_fine,
                  specific_pressure_fine,
                  in_longwave_radiation_fine,
                  in_shortwave_radiation_fine,
                  wind_speed_fine,
                  julian_day,
                  lat_fine,
                  lon_fine, 
                  elev_fine], axis=-1)
    num_feat = x.shape[-1]

    # prepare x
    x = x.reshape(-1, num_feat)
    np.save('x_test_{year}_{month}_{day}.npy'.format(year=year, month=month, day=day_of_month), np.array(x))
