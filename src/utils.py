import os
import netCDF4 as nc


def save2nc(name, 
            year, 
            month, 
            day_of_month, # Start from 0, thus add 1
            var, 
            lat_out, 
            lon_out): 
    f = nc.Dataset('ERA5_fine_{year:04}_{month:02}_{day_of_month:02}_{var}.nc'.format(
        year=year, month=month, day_of_month=day_of_month, var=name), 'w', format='NETCDF4')
    f.createDimension('longitude', size=var.shape[-1])
    f.createDimension('latitude', size=var.shape[-2])
    f.createDimension('time', size=var.shape[-3])

    times = f.createVariable('time', 'f8', dimensions=('time',))
    times.units = "hours since {year:04}-{month:02}-{day_of_month:02} 00:00:00.0".format(year=year, month=month, day_of_month=day_of_month+1)
    times.calendar = "gregorian"
    times.axis = "T"
    times.long_name = "time"
    times.standard_name = "time"

    lon0 = f.createVariable('longitude', 'f4', dimensions='longitude')
    lon0.units = "degrees_east"
    lon0.long_name = "longitude"
    lon0.standard_name = "longitude"
    lon0.axis = "X"

    lat0 = f.createVariable('latitude', 'f4', dimensions='latitude')
    lat0.units = "degrees_north"
    lat0.axis = "Y"
    lat0.long_name = "latitude"
    lat0.standard_name = "latitude"

    data = f.createVariable(name, 'f4', dimensions=('time','latitude','longitude'))
    lon0[:], lat0[:], data[:] = lon_out, lat_out, var
    f.close()


def bilInterp(outfil_name,
              lat_coarse, 
              lon_coarse, 
              var_coarse,
              year,
              month,
              day_of_month):
    """bilinear interpolate intermediate var by cdo"""        
    # save intermediate variables into nc file
    f = nc.Dataset('{name}_{year:04}_{month:02}_{day_of_month:02}.nc'.format(
        name=outfil_name, year=year, month=month, day_of_month=day_of_month), 'w', format='NETCDF4')
    f.createDimension('longitude', size=var_coarse.shape[-1])
    f.createDimension('latitude', size=var_coarse.shape[-2])
    f.createDimension('time', size=var_coarse.shape[-3])
    lon0 = f.createVariable('longitude', 'f4', dimensions='longitude')
    lat0 = f.createVariable('latitude', 'f4', dimensions='latitude')
    data = f.createVariable('var', 'f4', dimensions=('time','latitude','longitude'))
    lon0[:], lat0[:], data[:] = lon_coarse, lat_coarse, var_coarse
    f.close()
        
    # set grid for nc file generated from previous step
    os.system("cdo setgrid,{in_grid_file} {name}_{year:04}_{month:02}_{day_of_month:02}.nc {name}_{year:04}_{month:02}_{day_of_month:02}_tmp.nc".format(
        in_grid_file='grid_coarse_{year}_{month}_{day}.txt'.format(year=year, month=month, day=day_of_month),
        name=outfil_name,   
        year=year, 
        month=month, 
        day_of_month=day_of_month))
    
    # remap 0.1 degree to 90m in Guangdong based on prepared weight file
    os.system("cdo remap,{out_grid_file},{weight_file} {name}_{year:04}_{month:02}_{day_of_month:02}_tmp.nc {name}_{year:04}_{month:02}_{day_of_month:02}_interp.nc".format(
        out_grid_file='grid_fine_{year}_{month}_{day}.txt'.format(year=year, month=month, day=day_of_month),
        weight_file='weight_{year}_{month}_{day}.nc'.format(year=year, month=month, day=day_of_month),
        name=outfil_name, 
        year=year, 
        month=month, 
        day_of_month=day_of_month))
    
    # read var in fine resolution 
    f = nc.Dataset("{name}_{year:04}_{month:02}_{day_of_month:02}_interp.nc".format(
        name=outfil_name, year=year, month=month, day_of_month=day_of_month),'r')
    var_fine = f["var"][:]
    
    # remove files
    os.system("rm -rf {name}_{year:04}_{month:02}_{day_of_month:02}.nc".format(
        name=outfil_name,  year=year, month=month, day_of_month=day_of_month))
    os.system("rm -rf {name}_{year:04}_{month:02}_{day_of_month:02}_tmp.nc".format(
        name=outfil_name, year=year, month=month, day_of_month=day_of_month))
    os.system("rm -rf {name}_{year:04}_{month:02}_{day_of_month:02}_interp.nc".format(
        name=outfil_name, year=year, month=month, day_of_month=day_of_month))
    return var_fine
    