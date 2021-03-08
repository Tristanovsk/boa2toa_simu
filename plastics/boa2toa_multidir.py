import os, copy
import glob
import numpy as np
import pandas as pd
import xarray as xr
from multiprocessing.dummy import Pool
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.ioff()
plt.rcParams.update({'font.family': 'Times New Roman',
                     'font.size': 16, 'axes.labelsize': 20,
                     'xtick.minor.visible': True,
                     'xtick.major.size': 5,
                     'ytick.minor.visible': True,
                     'ytick.major.size': 5})

from RTxploitation import lutplot
lp = lutplot.plot()

from plastics import *

SixSexe_path = os.path.abspath('6SV1.1/sixsV1.1')
from Py6Sperso import *

from plastics import data_utils as du

plot = False

# -----------------------------------
#      Data formatting
# -----------------------------------

files = glob.glob(opj(idir, 'data', 'Level_2B', '*dat'))
xdata = du.data(files).load()

# -----------------------------------
# PLOTTING
# -----------------------------------

conds = ['dry', 'wet']
cond = conds[1]
dup = du.plot(xdata, cond)
if plot:

    dup.spectra()

    for vza in [0, 15, 30, 45]:
        dup.concentration(vza)

    for concentration in xdata.concentration.values:
        dup.viewing_angle(concentration)

# -----------------------------
# get spectra of reference for VRT calculations
# -----------------------------

wls = np.arange(450, 2300, 10)
# remove wl when atmo transmittance ~ 0 or low quality data
wls = wls[wls > 450]

for wl_ in ([1304, 1465], [1763, 1982]):
    wls = wls[(wls < wl_[0]) | (wls > wl_[1])]

hdrf_ref = xdata.sel(wl=wls, vza=0, concentration=100, condition='wet').hdrf_avg

# ---------------------------------------
#     6S VRT computation
# ---------------------------------------
sza = 5 #30  # 10 #40
wl = 1600 #865
wind_speed, wind_azimuth, salinity, pigment_concentration = 2, 0, 34, 0.3
xfoams = [0, 0.2, 0.4, 0.6, 0.8, 1]

vza = np.linspace(0, 80, 11)
azi = np.linspace(0, 360, 31)
geom = []
for vza_ in vza:
    for azi_ in azi:
        geom.append([vza_, azi_])
geom_ = pd.DataFrame(geom)
vzas, azis = geom_.values.T

aot = 0.2
aerosols = [['maritime', AeroProfile.Maritime],
            ['continental', AeroProfile.Continental],
            ['desert', AeroProfile.Desert]]
aerosol = aerosols[1]
amodel = aerosol[0]

def process(sza=30,wl=865):
    resfile = 'plastic_dir_impact_boa_toa_amodel_' + amodel + '_aot' + str(aot) + \
              '_sza' + str(sza) + '_wl' + str(wl)

    datafile = opj(idir, 'data', resfile + '.csv')

    wl_mic = wl / 1000  # convert nm -> microns
    print(wl)

    if os.path.exists(datafile):
        dff = pd.read_csv(datafile)
    else:
        s = SixS(SixSexe_path)
        s.geometry.solar_z = sza
        s.geometry.solar_a = 0
        # s.geometry.view_z = vza
        # s.geometry.view_a = azi
        s.altitudes.set_sensor_satellite_level()
        s.aot550 = aot
        s.aero_profile = AeroProfile.PredefinedType(aerosol[1])
        # xfoam = 1
        # ground_reflec = 0.1
        # s.wavelength = Wavelength(0.67)
        # s.ground_reflectance = GroundReflectance.HomogeneousOcean(
        #     wind_speed, wind_azimuth, salinity, pigment_concentration,
        #     ground_reflec, xfoam)
        # SixSHelpers.Angles.run_and_plot_360(s, 'view', 'pixel_reflectance')
        df_ = []
        for type, group in hdrf_ref.groupby('type'):
            print(type)
            geom_['hdrf'] = group.interp(wl=wl)
            g = geom_.__array__()  # group.to_dataframe().reset_index().loc[:, ('wl', 'hdrf_avg')]
            # wl = g.wl
            for xfoam in xfoams:

                def proc(p):
                    vza, azi, ground_reflec = p

                    s.outputs = None
                    a = copy.deepcopy(s)
                    a.wavelength = Wavelength(wl_mic)
                    a.geometry.view_z = vza
                    a.geometry.view_a = azi
                    a.ground_reflectance = GroundReflectance.HomogeneousOcean(
                        wind_speed, wind_azimuth, salinity, pigment_concentration,
                        ground_reflec, xfoam)
                    a.run()

                    return a.outputs


                pool = Pool()

                results = pool.map(proc, g)
                pool.close()
                pool.join()

                F0, trans_gas, trans_scat, irradiance, = [], [], [], []
                toa_refl, intrinsic_refl, toa_rad = [], [], []

                for res in results:
                    print(res.atmospheric_intrinsic_reflectance)
                    toa_refl = np.append(toa_refl, res.apparent_reflectance)
                    toa_rad = np.append(toa_rad, res.apparent_radiance)

                df = pd.DataFrame({'type': type, 'xfoam': xfoam, 'wl': wl, 'sza': sza,
                                   'vza': vzas, 'azi':azis, 'toa_rad': toa_rad, 'toa_refl': toa_refl})
                df_.append(df)

        dff = pd.concat(df_)
        dff.to_csv(datafile, index=False)

for sza in (5,30,60):
    for wl in (865,1600):
        process(sza,wl)

#
# xtoa = dff.set_index(['type', 'xfoam', 'wl','sza','vza','azi']).to_xarray()
#
#
# # ------------------------------
# # Plotting section
# # ------------------------------
# # ----------------------------
# # plot rho factor for given SZA
# # ----------------------------
# # construct raster dimensions
# type='B'
#
# type_label = {'B': 'Sheets', 'Foam': 'Foam', 'HDPE': 'HDPE',
#               'Hard': 'Hard Fragments', 'LDPE': 'LDPE', 'P': 'Pellets',
#               'PP': 'PP', 'PS': 'PS', 'Rope': 'Rope'}
# vmax={'B': 0.2, 'Foam': 0.2, 'HDPE': 0.2,
#               'Hard': 0.1, 'LDPE': 0.1, 'P': 0.1,
#               'PP': 0.2, 'PS': 0.2, 'Rope': 0.1}
# cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["darkblue","cornflowerblue","white","gold","red",'firebrick'])#"firebrick","darkgoldenrod",,'purple'
# vzamax=80
# xtoa_ = xtoa.interp(azi=np.linspace(0,360,100),method='cubic').interp(vza=np.arange(0,80,1), method='cubic')
# r, theta = np.meshgrid(xtoa_.vza, np.radians(xtoa_.azi))
#
# for type, group in xtoa_.toa_refl.isel(wl=0,sza=0).groupby('type'):
#
#     fig, axs = plt.subplots(1,6, figsize=(26, 8), subplot_kw=dict(projection='polar'))
#
#     for i, xfoam in  enumerate([0,0.2,0.4,0.6,0.8,1.]):
#         arr = group.sel(xfoam=xfoam)
#         cax = lp.add_polplot(axs[i], r, theta, arr.T,
#                              title=str(xfoam*100)+' %',cmap=cmap,scale=False,
#                              vmin=0.0,vmax=0.2) #vmax[type])
#     plt.tight_layout()
#     fig.subplots_adjust(top=.85)
#     cb = fig.colorbar(cax, ax=axs.ravel().tolist(),
#                       shrink=0.6, location='top',pad=0.25,label=type_label[type])
#     cb.ax.tick_params(labelsize=19)
#     #plt.suptitle(type,fontsize=24)
#     plt.savefig(opj(idir, 'fig', resfile + '_'+type+'_polar_plot.png'), dpi=300)
