#-----------------------------------------------------------------------------
# The topography downscale methods for forcing except precipitation
#
# Author: Lu Li, Sisi Chen, Zhongwang Wei
# Reference:
#   Mei et al. (2020): A Nonparametric Statistical Technique for Spatial 
#       Downscaling of Precipitation Over High Mountain Asia, 
#       Water Resourse Research, 56, e2020WR027472.
#   Rouf et al. (2020): A Physically Based Atmospheric Variables Downscaling 
#       Technique. Journal of Hydrometeorology, 21, 93-108.
#-----------------------------------------------------------------------------


import os
import netCDF4 as nc
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor



def bilinear_interp_from_cdo(outfil_name,
                             lat_coarse, 
                             lon_coarse, 
                             var_coarse,
                             year,
                             month,
                             begin_hour):
    """bilinear interpolate by cdo"""
    # save intermediate variables into nc file
    f = nc.Dataset('{name}_GD_{year:04}_{month:02}_{begin_hour:02}.nc'.format(
        name=outfil_name, year=year, month=month, begin_hour=begin_hour), 'w', format='NETCDF4')
    f.createDimension('longitude', size=var_coarse.shape[-1])
    f.createDimension('latitude', size=var_coarse.shape[-2])
    f.createDimension('time', size=var_coarse.shape[-3])
    lon0 = f.createVariable('longitude', 'f4', dimensions='longitude')
    lat0 = f.createVariable('latitude', 'f4', dimensions='latitude')
    data = f.createVariable('var', 'f4', dimensions=('time','latitude','longitude'))
    lon0[:], lat0[:], data[:] = lon_coarse, lat_coarse, var_coarse
    f.close()
    # set grid for nc file generated from previous step
    os.system("cdo setgrid,{in_grid_file} {name}_GD_{year:04}_{month:02}_{begin_hour:02}.nc {name}_GD_{year:04}_{month:02}_{begin_hour:02}_tmp.nc".format(
        in_grid_file="/tera06/lilu/ForDs/data/scripts/0p1_GD_cdo.txt",
        name=outfil_name,  year=year, month=month, begin_hour=begin_hour))
    # remap 0.1 degree to 90m in Guangdong based on prepared weight file
    os.system("cdo remap,{out_grid_file},{weight_file} {name}_GD_{year:04}_{month:02}_{begin_hour:02}_tmp.nc {name}_GD_{year:04}_{month:02}_{begin_hour:02}_interp.nc".format(
        out_grid_file="/tera06/lilu/ForDs/data/scripts/90m_GD_cdo.txt",
        weight_file="/tera06/lilu/ForDs/data/scripts/bilinear_90x90_10801x10801_cdo.nc",
        name=outfil_name, year=year, month=month, begin_hour=begin_hour))
    # read var in fine resolution 
    f = nc.Dataset("{name}_GD_{year:04}_{month:02}_{begin_hour:02}_interp.nc".format(name=outfil_name, year=year, month=month, begin_hour=begin_hour),'r')
    var_fine = f["var"][:]
    # remove files
    os.system("rm -rf {name}_GD_{year:04}_{month:02}_{begin_hour:02}.nc".format(name=outfil_name,  year=year, month=month, begin_hour=begin_hour))
    os.system("rm -rf {name}_GD_{year:04}_{month:02}_{begin_hour:02}_tmp.nc".format(name=outfil_name, year=year, month=month, begin_hour=begin_hour))
    os.system("rm -rf {name}_GD_{year:04}_{month:02}_{begin_hour:02}_interp.nc".format(name=outfil_name, year=year, month=month, begin_hour=begin_hour))
    return var_fine


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
    """Calculate lapse rate by regional regression method.

    NOTE: time dimension last
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
    laspe_rate = regression(y, x)
    
    # control the boundary of lapse rate
    for i in range(laspe_rate.shape[-1]):
        tmp = laspe_rate[:,:,i]
        up_bound, low_bound = np.nanquantile(tmp, 0.95), np.nanquantile(tmp, 0.05)
        tmp[tmp>up_bound] = up_bound
        tmp[tmp<low_bound] = low_bound
        laspe_rate[:,:,i] = tmp

    # enlarge the boundary
    laspe_rate_full = np.zeros_like(input_coarse)*np.nan
    laspe_rate_full[1:-1,1:-1] = laspe_rate
    return np.transpose(laspe_rate_full,(2,0,1)), x, y # time first


def calc_clear_sky_emissivity(air_temperature, dew_temperature, case='satt'):
    """Calculate emissivity in clear-sky."""
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
    cos_phi = (np.sin(declination) * np.cos(np.radians(latitude))- \
        np.cos(np.radians(w)) * np.cos(declination) * np.sin(np.radians(latitude)))/ np.cos(sun_elev)
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


def calc_shadow_mask(zenith_angle_fine, 
                     azimuth_angle_fine, 
                     elevation_fine,
                     search_radius,
                     D=90):
    """calculate shadow mask derived from WRF"""
    nlat, nlon = elevation_fine.shape
    shadow_mask = np.zeros_like(zenith_angle_fine) # 2D (lat, lon)
    
    for i in range(search_radius, nlat-search_radius):
        for j in range(search_radius, nlon-search_radius):
            # if zenith angle nearly equal to 0, 
            # shadow mask set to 0
            if np.sin(zenith_angle_fine[i,j])<1e-2:
                shadow_mask[i,j] = 0

            # if azimuth belong to (0, 1/4*pi) and (7/4*pi, pi)
            # search north
            elif (azimuth_angle_fine[i,j]>1.75*np.pi) & \
                (azimuth_angle_fine[i,j]<0.25*np.pi):
                # search N grids along Y-axis 
                for jj in range(j+1,j+search_radius+1):
                    # which grids along X-axis at the direction of azimuth
                    i0 = i+(jj-j)*np.tan(azimuth_angle_fine[i,j]-2*np.pi)
                    # previous/now grid along X-axis
                    i1, i2 = int(np.floor(i0)), int(np.floor(i0)+1)
                    # how long in now grid
                    weight = i0-i1
                    # calculate the angle of delta(z)
                    dz_angle = np.arctan((weight*elevation_fine[i2,jj]+ \
                        (1-weight)*elevation_fine[i1,jj]- \
                                elevation_fine[i,j])/ \
                                    np.sqrt((D*(jj-j))**2+(D*(i0-i))**2))
                    # shadow mask if delta(z) angle less than zenith
                    if np.sin(dz_angle)>np.sin(zenith_angle_fine[i,j]):
                        shadow_mask[i,j] = 1
                        break

            # if azimuth belong to (1/4*pi, 3/4*pi)
            # search east
            elif (azimuth_angle_fine[i,j]<0.75*np.pi) & \
                (azimuth_angle_fine[i,j]>0.25*np.pi):
                for ii in range(i+1, i+search_radius+1):
                    j0 = j-(ii-i)*np.tan(azimuth_angle_fine[i,j]-0.5*np.pi)
                    j1, j2 = int(np.floor(j0)), int(np.floor(j0)+1)
                    weight = j0-j1
                    dz_angle=np.arctan((weight*elevation_fine[ii,j2]+ \
                        (1.-weight)*elevation_fine[ii,j1]- \
                                elevation_fine[i,j])/ \
                                    np.sqrt((D*(ii-i))**2+(D*(j0-j))**2))
                    if np.sin(dz_angle)>np.sin(zenith_angle_fine[i,j]):
                        shadow_mask[i,j] = 1
                        break

            # if azimuth belong to (3/4*pi, 5/4*pi)
            # search south
            elif (azimuth_angle_fine[i,j]<1.25*np.pi) & \
                (azimuth_angle_fine[i,j]>0.75*np.pi):
                for jj in range(j-1,j-search_radius,-1):
                    i0 = i+(jj-j)*np.tan(azimuth_angle_fine[i,j]-np.pi)
                    i1, i2 = int(np.floor(i0)), int(np.floor(i0)+1)
                    weight = i0-i1
                    dz_angle = np.arctan((weight*elevation_fine[i2,jj]+ \
                        (1-weight)*elevation_fine[i1,jj]- \
                            elevation_fine[i,j])/ \
                                    np.sqrt((D*(jj-j))**2+(D*(i0-i))**2))
                    if np.sin(dz_angle)>np.sin(zenith_angle_fine[i,j]):
                        shadow_mask[i,j] = 1
                        break
                    
            # if azimuth belong to (5/4*pi, 7/4*pi)
            # search west
            elif (azimuth_angle_fine[i,j]<1.75*np.pi) & \
                (azimuth_angle_fine[i,j]>1.25*np.pi):
                for ii in range(i-1,i-search_radius,-1):
                    j0 = j-(ii-i)*np.tan(azimuth_angle_fine[i,j]-1.5*np.pi)
                    j1, j2 = int(np.floor(j0)), int(np.floor(j0)+1)
                    weight = j0-j1
                    dz_angle=np.arctan((weight*elevation_fine[ii,j2]+ \
                        (1.-weight)*elevation_fine[ii,j1]- \
                            elevation_fine[i,j])/ \
                                np.sqrt((D*(ii-i))**2+(D*(j0-j))**2))
                    if np.sin(dz_angle)>np.sin(zenith_angle_fine[i,j]):
                        shadow_mask[i,j] = 1
                        break                    
        return shadow_mask


def downscale_air_temperature(air_temperature_coarse,
                              air_temperature_fine_interp, 
                              elevation_coarse,
                              elevation_fine_interp, 
                              elevation_fine,
                              lat_coarse,
                              lon_coarse, year, month, begin_hour):
    # calculate lapse rate for air temperature
    laspe_rate_coarse, x, y = calc_lapse_rate(np.transpose(air_temperature_coarse, (1,2,0)), elevation_coarse)
    # bilinear interpolate of laspe rate 
    laspe_rate_fine_interp = bilinear_interp_from_cdo('laspe_rate', 
                                                      lat_coarse, 
                                                      lon_coarse, 
                                                      laspe_rate_coarse, 
                                                      year,
                                                      month,
                                                      begin_hour)
    # downscaling
    dz = (elevation_fine-elevation_fine_interp)[np.newaxis]
    air_temperature_fine = air_temperature_fine_interp + np.multiply(
            laspe_rate_fine_interp, dz)
    return air_temperature_fine
    

def downscale_dew_temperature(dew_temperature_coarse, 
                              dew_temperature_fine_interp, 
                              elevation_coarse,
                              elevation_fine_interp, 
                              elevation_fine,
                              lat_coarse,
                              lon_coarse, year, month, begin_hour):
    # calculate lapse rate for dew temperature
    laspe_rate_coarse = calc_lapse_rate(dew_temperature_coarse, elevation_coarse)
    # bilinear interpolate of laspe rate
    laspe_rate_fine_interp = bilinear_interp_from_cdo('laspe_rate', 
                                                      lat_coarse, 
                                                      lon_coarse, 
                                                      laspe_rate_coarse,
                                                      year,
                                                      month,
                                                      begin_hour)
    # downscaling
    dz = (elevation_fine-elevation_fine_interp)[:,:,np.newaxis]
    dew_temperature_fine = dew_temperature_fine_interp + np.multiply(
            laspe_rate_fine_interp, dz)
    return dew_temperature_fine
    

def downscale_air_pressure(air_pressure_fine_interp,
                           air_temperature_fine_interp,
                           air_temperature_fine,
                           elevation_coarse, 
                           elevation_fine_interp,
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
                                    in_longwave_radiation_fine_interp,
                                    air_temperature_coarse,
                                    dew_temperature_coarse,
                                    air_temperature_fine,
                                    dew_temperature_fine,
                                    air_temperature_fine_interp,
                                    lat_coarse,
                                    lon_coarse):
    # calculate emissivity of coarse and fine resolution
    emissivity_clear_sky_coarse = calc_clear_sky_emissivity(air_temperature_coarse, dew_temperature_coarse)
    emissivity_clear_sky_fine = calc_clear_sky_emissivity(air_temperature_fine, dew_temperature_fine)

    # calculate all emissivity including both clear-sky and cloudy
    SIGMA = 5.67e-8 # Stefan-Boltzmann constant
    emissivity_all_coarse = in_longwave_radiation_coarse/(SIGMA*air_temperature_coarse**4)

    # calculate cloudy emissivity
    emissivity_cloudy_coarse = emissivity_all_coarse-emissivity_clear_sky_coarse

    # bilinear interpolate cloudy emissivity
    emissivity_cloudy_fine_interp = bilinear_interp_from_cdo('emissivity_cloudy',
                                                             lat_coarse,
                                                             lon_coarse,
                                                             emissivity_cloudy_coarse)
    emissivity_all_fine_interp = bilinear_interp_from_cdo('emissivity_all',
                                                           lat_coarse,
                                                           lon_coarse,
                                                           emissivity_all_coarse)
    # calculate all emissivity in fine
    emissivity_all_fine = emissivity_clear_sky_fine+emissivity_cloudy_fine_interp

    # downscaling
    in_longwave_radiation_fine = in_longwave_radiation_fine_interp* \
        (emissivity_all_fine/emissivity_all_fine_interp)* \
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
                                     latitude_coarse,
                                     latitude_fine):
    # ------------------------------------------------------------
    # 1. partition short radiation into beam and diffuse radiation
    # ------------------------------------------------------------
    #TODO: li, check degree or rad
    # calculate solar angles
    zenith_angle_coarse, azimuth_angle_coarse = calc_solar_angles_chelsa(julian_day,hour,latitude_coarse)
    zenith_angle_fine, azimuth_angle_fine = calc_solar_angles_chelsa(julian_day,hour,latitude_fine)


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

    # calculate factor to account for the difference of 
    # optical path length due to pressure difference
    optical_length_factor_fine = np.exp(k_fine_interp*(
        air_pressure_fine-air_pressure_fine_interp))
    
    # calcualte the cosine of solar illumination angle, cos(θ), 
    # ranging between −1 and 1, indicates if the sun is below or 
    # above the local horizon (note that values lower than 0 are set to 0);
    # TODO: 
    cos_illumination = np.cos(zenith_angle_fine)*np.cos(slope_fine)+ \
        np.sin(zenith_angle_fine)*np.sin(slope_fine)* \
            np.cos(azimuth_angle_fine-aspect_fine)
    
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


def downscale_wind_speed(u_wind_speed_fine_interp,
                         v_wind_speed_fine_interp,
                         slope_fine,
                         aspect_fine,
                         curvature_fine): 
    # calculate wind direction
    wind_direction_fine_interp = np.arctan(v_wind_speed_fine_interp/u_wind_speed_fine_interp)
    wind_speed_fine_interp = np.sqrt(u_wind_speed_fine_interp**2+v_wind_speed_fine_interp**2)

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
    # NOTE: only for square inputs
    scale = round(elevation_fine.shape[0]/elevation_coarse.shape[0])
    nlat, nlon, nt = precipitation_coarse.shape
    precipitation_fine = np.full((elevation_fine.shape[0],elevation_fine.shape[1],nt), np.nan)
    
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
            zs = zs[:,:,np.newaxis] # expand dimension for oprend with time dimensions
            
            # calculate precip
            pg, zg = precipitation_coarse[i:i+1,j:j+1], elevation_coarse[i,j]
            if case == 'tesfa':
                # Tesfa et al, 2020: Exploring Topography-Based Methods for Downscaling
                # Subgrid Precipitation for Use in Earth System Models. Equation (5).
                zs_max = np.nanmax(zs)
                tmp = pg*(zs-zg)/zs_max
            elif case == 'liston':
                # Liston and Elder: A meteorological distribution system for high-resolution 
                # terrestrial modeling (MicroMet), J. Hydrometeorol., 2006. 
                tmp = pg*2.0*0.27e-3*(zs-zg)/(1.0-0.27e-3*(zs-zg))
           
            # rewrite in outfile
            if (i != nlat-1) & (j != nlon-1):
                precipitation_fine[i*scale:(i+1)*scale, j*scale:(j+1)*scale] = tmp
            elif (i == nlat-1) & (j != nlon-1):
                precipitation_fine[i*scale:, j*scale:(j+1)*scale] = tmp
            elif (i != nlat-1) & (j == nlon-1):
                precipitation_fine[i*scale:(i+1)*scale, j*scale:] = tmp
            else:
                precipitation_fine[i*scale:, j*scale:] = tmp
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
                                longtitude_fine,
                                #LAI_coarse,
                                #LAI_fine,
                                RAIN_THRESHOLD=0.01
                                ):
    # calculate precipitation mask in coarse resolution
    precipitation_mask_coarse = np.zeros_like(precipitation_coarse)
    precipitation_mask_coarse[precipitation_coarse>RAIN_THRESHOLD] = 1

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
    precipitation_fine = precipitation_fine.reshape(len(latitude_fine),len(longtitude_fine),-1)
    return precipitation_fine



if __name__ == '__main__':
    import time

    # test interpolate by cdo
    """
    out_file = '/tera06/lilu/ForDs/data/DEM/MERITDEM/MERITDEM_GD_Aspect.nc'
    f = nc.Dataset(out_file)
    lat_out, lon_out = f['lat'][:], f['lon'][:] 
    in_file = '/tera06/lilu/ForDs/data/forcing/ERA5LAND_GD_2018_01_tp.nc'
    f = nc.Dataset(in_file)
    lat_in, lon_in = f['latitude'][:], f['longitude'][:]
    c = bilinear_interp_from_cdo('a', 
                                 lat_out, lon_out, 
                                 np.random.random((2,90,90)))
    print(c.shape)
    """
    
    # test Magnus formula/air pressure/specific humidity/relative humidity
    """ 
    dew_temperature_fine = np.random.random((10801,10801,24))
    vapor_pressure, a, b, c = calc_vapor_pressure(dew_temperature_fine)
    print(vapor_pressure.shape, a.shape, b.shape, c.shape)
    """

    # test air temperature/dew temperature
    """
    elevation_coarse = np.random.random((90,90))
    elevation_fine_interp = np.random.random((10801,10801))
    elevation_fine = np.random.random((10801,10801))
    air_temperature_coarse = np.random.random((90,90,24))
    air_temperature_fine_interp = np.random.random((10801,10801,24))
    in_file = '/tera06/lilu/ForDs/data/forcing/ERA5LAND_GD_2018_01_tp.nc'
    f = nc.Dataset(in_file)
    lat_coarse, lon_coarse = f['latitude'][:], f['longitude'][:]
    
    t1 = time.time()
    air_temperature_fine = downscale_air_temperature(air_temperature_coarse,
                                                     air_temperature_fine_interp,                                                    
                                                     elevation_coarse,
                                                     elevation_fine_interp,
                                                     elevation_fine,
                                                     lat_coarse,
                                                     lon_coarse)
    t2 = time.time()
    print(t2-t1)
    """

    # test longwave radiation
    """
    elevation_coarse = np.random.random((90,90))
    elevation_fine_interp = np.random.random((10801,10801))
    elevation_fine = np.random.random((10801,10801))
    air_temperature_coarse = np.random.random((90,90,24))
    air_temperature_fine = np.random.random((10801,10801,24))
    air_temperature_fine_interp = np.random.random((10801,10801,24))
    dew_temperature_coarse = np.random.random((90,90,24))
    dew_temperature_fine = np.random.random((10801,10801,24))
    in_longwave_radiation_coarse = np.random.random((90,90,24))
    in_longwave_radiation_fine_interp = np.random.random((10801,10801,24))
    in_file = '/tera06/lilu/ForDs/data/forcing/ERA5LAND_GD_2018_01_tp.nc'
    f = nc.Dataset(in_file)
    lat_coarse, lon_coarse = f['latitude'][:], f['longitude'][:]
    t1 = time.time()
    in_longwave_radiation_fine = downscale_in_longwave_radiation(in_longwave_radiation_coarse,
                                                                 in_longwave_radiation_fine_interp,
                                                                 air_temperature_coarse,
                                                                 dew_temperature_coarse,
                                                                 air_temperature_fine,
                                                                 dew_temperature_fine,
                                                                 air_temperature_fine_interp,
                                                                 lat_coarse,
                                                                 lon_coarse)
    t2 = time.time()
    print(t2-t1)
    print(in_longwave_radiation_fine.shape)
    """

    # test wind 

    # test precipitation
    """
    elevation_coarse = np.random.random((90,90))
    elevation_fine = np.random.random((10801,10801))
    precipitation_coarse = np.random.random((90,90,24))
    t1 = time.time()
    precipitation_fine = downscale_precipitation_colm(precipitation_coarse,
                                                      elevation_coarse,
                                                      elevation_fine,
                                                      case='liston')
    t2 = time.time()
    print(t2-t1)
    print(precipitation_fine.shape)
    """

    # test shadow mask
    """
    np.random.seed(0)
    zenith_angle_fine = np.random.random((10801,10801))*2*np.pi
    azimuth_angle_fine = np.random.random((10801,10801))*2*np.pi
    elevation_fine = np.random.random((10801,10801))
    t1 = time.time()
    shadow_mask = calc_shadow_mask(zenith_angle_fine,
                                   azimuth_angle_fine,
                                   elevation_fine,
                                   search_radius=20,
                                   D=90)
    t2 = time.time()
    print(t2-t1)
    print(shadow_mask)
    """
