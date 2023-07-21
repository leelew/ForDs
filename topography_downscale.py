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
import topocalc



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


def calc_emissivity(air_temperature):
    """Calculate emissivity by Cosgrove et al. (2003).
    
    Reference:
        Cosgrove et al. (2003): Real-time and retrospective forcing in the 
            North American Land Data Assimilation System (NLDAS) project.
            Journal of Geophysical Research, 108.
    """
    vapor_pressure, _, _, _ = calc_vapor_pressure(air_temperature)
    return 1.08*(1-np.exp(-vapor_pressure**(air_temperature/2016)))
    

def calc_solar_angles(julian_day, hour, latitude):
    """Calculate solar zenith/azimuth angles
    
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
    return cos_azimuth, cos_zenith, sin_zenith


# TODO: Read CastShadow.m and rewrite
def calc_shadow_mask():
    pass


def downscale_air_temperature(air_temperature_coarse, 
                              elevation_coarse, 
                              elevation_fine):
    # calculate lapse rate for air temperature
    laspe_rate = calc_lapse_rate(air_temperature_coarse, elevation_fine)
    # downscaling
    air_temperature_fine = air_temperature_coarse + laspe_rate*(
        elevation_fine-elevation_coarse)
    return air_temperature_fine
    

def downscale_dew_temperature(dew_temperature_coarse, 
                              elevation_coarse, 
                              elevation_fine):
    # calculate lapse rate for dew temperature
    laspe_rate = calc_lapse_rate(dew_temperature_coarse, elevation_fine)
    # downscaling
    dew_temperature_fine = dew_temperature_coarse + laspe_rate*(
        elevation_fine-elevation_coarse)
    return dew_temperature_fine
    

def downscale_air_pressure(air_pressure_coarse,
                           air_temperature_coarse,
                           air_temperature_fine,
                           elevation_coarse, 
                           elevation_fine):
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
    # downscaling
    air_pressure_fine = air_pressure_coarse*np.exp(
        (-G*(elevation_fine-elevation_coarse))/(
            R*(air_temperature_coarse+air_temperature_fine)/2))
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
                                    air_temperature_fine):
    # calculate emissivity of coarse and fine resolution
    emissivity_coarse = calc_emissivity(air_temperature_coarse)
    emissivity_fine = calc_emissivity(air_temperature_fine)
    # downscaling
    in_longwave_radiation_fine = in_longwave_radiation_coarse* \
        (emissivity_fine/emissivity_coarse)* \
            ((air_temperature_fine/air_temperature_coarse)**4)
    return in_longwave_radiation_fine


def downscale_in_shortwave_radiation(in_short_radiation_coarse,
                                     air_pressure_coarse,
                                     elevation_coarse,
                                     air_pressure_fine,
                                     albedo_fine,
                                     julian_day,
                                     hour,
                                     latitude):
    # ------------------------------------------------------------
    # 1. partition short radiation into beam and diffuse radiation
    # ------------------------------------------------------------
    # calculate solar angles
    cos_azimuth, cos_zenith, sin_zenith = calc_solar_angles(julian_day, hour,latitude)
    azimuth_angle = np.arccos(cos_azimuth)
    
    # calculate top-of-atmosphere incident short radiation
    S = 1370 # solar constant [W/m^2]
    toa_in_short_radiation_coarse = S*cos_zenith
    
    # calculate clearness index
    clearness_index = in_short_radiation_coarse/toa_in_short_radiation_coarse
    
    # calculate diffuse weight
    diffuse_weight = 0.952-1.041*np.exp(-np.exp(2.3-4.702*clearness_index))
    
    # calculate diffuse and beam raidation
    diffuse_radiation = in_short_radiation_coarse*diffuse_weight
    beam_radiation = in_short_radiation_coarse*(1-diffuse_weight)
    
    # ------------------------------------------------------------
    # 2. downscaling beam radiation 
    # ------------------------------------------------------------
    # calcualte broadband attenuation coefficient [Pa^-1]
    k = np.log(clearness_index)/air_pressure_coarse
    
    # calculate factor to account for the difference of 
    # optical path length due to pressure difference
    factor_optical_path_length = np.exp(k*(air_pressure_fine-air_pressure_coarse))
    
    # calcualte the cosine of solar illumination angle, cos(θ), 
    # ranging between −1 and 1, indicates if the sun is below or 
    # above the local horizon (note that values lower than 0 are set to 0);
    nx, ny = elevation_coarse.shape
    slope, aspect = topocalc.gradient.gradient_d8(elevation_coarse, nx, ny)
    
    # calculate solar illumination angle
    cos_illumination = cos_zenith*np.cos(slope)+ \
        sin_zenith*np.sin(slope)*np.cos(azimuth_angle-aspect)
    
    # calculate shadow mask
    shadow_mask = calc_shadow_mask()
    
    # downscaling beam radiation
    beam_radiation_fine = shadow_mask*cos_illumination* \
        factor_optical_path_length*beam_radiation
    
    # ------------------------------------------------------------
    # 3. downscaling diffuse radiation 
    # ------------------------------------------------------------ 
    # NOTE: topocalc library only support square input of DEM,
    #       need fix if we apply to the global. or we may generate 
    #       skf, tcf outside of this program.
    sky_view_factor, terrain_configuration_factor = topocalc.viewf.viewf(
        elevation_coarse, nx, sin_slope=np.sin(slope), aspect=aspect)
    
    # downscaling diffuse radiation
    diffuse_radiation_fine = sky_view_factor*diffuse_radiation
    
    # ------------------------------------------------------------
    # 4. calculate reflected radiation 
    # ------------------------------------------------------------ 
    reflected_radiation_fine = albedo_fine*terrain_configuration_factor* \
        (beam_radiation_fine+(1-sky_view_factor)*diffuse_radiation)
    
    return beam_radiation_fine+diffuse_radiation_fine+reflected_radiation_fine


# FIXME: wind speed correction is not useful nowaday
def downscale_wind_speed(wind_speed_coarse,
                         roughness_fine,
                         roughness_coarse):   
    nt = roughness_coarse.shape[-1] 
    # calculate monthly mean of roughness
    # FIXME: Not useful for unchanged roughness
    roughness_coarse_mean = 0
    # temporal disaggregation factor
    roughness_fine = roughness_fine*roughness_coarse/roughness_coarse_mean
    # downscaling wind speed
    wind_speed_fine = wind_speed_coarse*(roughness_fine/roughness_coarse)**0.09
    return wind_speed_fine


def topography_downscale(air_temperature_coarse,
                         dew_temperature_coarse,
                         air_pressure_coarse,
                         in_longwave_radiation_coarse,
                         in_short_radiation_coarse,
                         wind_speed_coarse,
                         roughness_coarse,
                         elevation_coarse,
                         elevation_fine,
                         latitude_fine,
                         longtitude_fine,
                         albedo_fine,
                         NDVI_fine,
                         roughness_fine,
                         time):
    pass







def calc_cos_zenith(julian_day, latitude, longitude):
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
    return orb_coszen