import os

def postprocess(name, year, month):
    os.system("cdo mergetime ERA5LAND_GD_fine_{year:04}_{month:02}_*_{var}.nc ERA5LAND_GD_fine_{year:04}_{month:02}_{var}.nc".format(
        year=year, month=month, var=name))
    os.system("rm -rf ERA5LAND_GD_fine_{year:04}_{month:02}_*_{var}.nc".format(year=year, month=month, var=name))


if __name__ == '__main__':
    postprocess('t2m', 2018, 1)
