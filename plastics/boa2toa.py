
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
plt.rcParams.update({'font.family':'Times New Roman',
                    'font.size': 16,'axes.labelsize':20,
                     'xtick.minor.visible':True,
                     'xtick.major.size':5,
                     'ytick.minor.visible':True,
                     'ytick.major.size':5})


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

wind_speed, wind_azimuth, salinity, pigment_concentration = 2, 0, 34, 0.3
xfoams = [0, 0.2, 0.4, 0.6, 0.8, 1]
sza = 40 #10 #40
vza, azi =  5, 90
aot = 0.2
aerosols = [['maritime', AeroProfile.Maritime],
            ['continental', AeroProfile.Continental],
            ['desert', AeroProfile.Desert]]
aerosol = aerosols[1]
amodel = aerosol[0]

resfile = 'plastic_impact_boa_toa_amodel_' + amodel + '_aot' + str(aot) + '_sza' + str(sza) + '_vza' + str(
    vza) + '_azi' + str(azi)
datafile = opj(idir, 'data', resfile + '.csv')


if os.path.exists(datafile):
    dff = pd.read_csv(datafile)
else:
    s = SixS(SixSexe_path)
    s.geometry.solar_z = sza
    s.geometry.solar_a = 0
    s.geometry.view_z = vza
    s.geometry.view_a = azi
    s.altitudes.set_sensor_satellite_level()
    s.aot550 = aot
    s.aero_profile = AeroProfile.PredefinedType(aerosol[1])
    # xfoam = 1
    # ground_reflec = 0.1
    # s.wavelength = Wavelength(0.67)
    # s.ground_reflectance = GroundReflectance.HomogeneousOcean(
    #     wind_speed, wind_azimuth, salinity, pigment_concentration,
    #     ground_reflec, xfoam)
    #SixSHelpers.Angles.run_and_plot_360(s, 'view', 'pixel_reflectance')
    df_ = []
    for type, group in hdrf_ref.groupby('type'):
        print(type)
        g = group.to_dataframe().reset_index().loc[:, ('wl', 'hdrf_avg')]
        wl = g.wl
        for xfoam in xfoams:

            def proc(p):
                wl, ground_reflec = p
                wl = wl / 1000  # convert nm -> microns
                print(wl)
                s.outputs = None
                a = copy.deepcopy(s)
                a.wavelength = Wavelength(wl)
                a.ground_reflectance = GroundReflectance.HomogeneousOcean(
                    wind_speed, wind_azimuth, salinity, pigment_concentration,
                    ground_reflec, xfoam)
                a.run()

                return a.outputs


            pool = Pool()
            refl_spectrum = g.__array__()
            results = pool.map(proc, refl_spectrum)
            pool.close()
            pool.join()

            F0, trans_gas, trans_scat, irradiance, = [], [], [], []
            toa_refl, intrinsic_refl, toa_rad = [], [], []

            for res in results:
                print(res.atmospheric_intrinsic_reflectance)
                toa_refl = np.append(toa_refl, res.apparent_reflectance)
                toa_rad = np.append(toa_rad, res.apparent_radiance)

            df = pd.DataFrame({'type': type, 'xfoam': xfoam, 'wl': wl, 'toa_rad': toa_rad, 'toa_refl': toa_refl})
            df_.append(df)

    dff = pd.concat(df_)
    dff.to_csv(datafile,index=False)

xtoa = dff.set_index(['type', 'xfoam', 'wl']).to_xarray()


# ------------------------------
# Plotting section
# ------------------------------
type_order = [2,4,6,7,8,0,1,5,3]
xtoa = xtoa.isel(type=type_order)
type_label = {'B':'Sheets', 'Foam':'Foam', 'HDPE':'HDPE',
              'Hard':'Hard Fragments', 'LDPE':'LDPE', 'P':'Pellets',
              'PP':'PP', 'PS':'PS', 'Rope':'Rope'}

cmap = plt.cm.RdYlGn_r
nrows = 9
fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=(13, 30))
fig.subplots_adjust(bottom=0.05, top=0.99, left=0.1, right=0.98,
                    hspace=0.6, wspace=0.3)
# axs = axs.ravel()
# axs[0].plot(wl, trans_tot, '--k')

for i, type in enumerate(xtoa.type.values):
    g = dff[dff.type == type]
    print(i,type)
    # axs[0].plot(wl, y, color=cmap(norm(xfoam)), lw=1.5, markersize=2, alpha=0.5)
    p = axs[i, 0].scatter(g.wl, g.toa_refl, c=g.xfoam * 100, cmap=cmap, lw=1.5, alpha=0.5)
    axs[i, 1].scatter(g.wl, g.toa_rad, c=g.xfoam * 100, cmap=cmap, lw=1.5, alpha=0.5)
    axs[i, 0].set_ylabel('TOA Reflectance $(-)$',fontsize=15)
    axs[i, 1].set_ylabel('TOA Radiance\n $(W\ m^{-2}\ sr^{-1}\ \mu m^{-1})$',fontsize=15)
    axs[i, 0].set_title(type_label[type])
    axs[i, 1].set_title(type_label[type])
    i += 1
axs[i - 1, 0].set_xlabel('Wavelength (nm)')
axs[i - 1, 1].set_xlabel('Wavelength (nm)')

cb = fig.colorbar(p, ax=axs, shrink=0.6, location='top')
cb.set_label('Plastic coverage (%)')

plt.suptitle(resfile)
plt.savefig(opj(idir, 'fig', resfile + 'test.png'), dpi=200)
plt.close()

cmap = plt.cm.Spectral_r
wls = [550, 670, 870, 1020, 1600, 2200]
param = 'toa_refl'
fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=(13, 30))
fig.subplots_adjust(bottom=0.05, top=0.99, left=0.1, right=0.98,
                    hspace=0.6, wspace=0.3)
# axs = axs.ravel()
# axs[0].plot(wl, trans_tot, '--k')
i = 0
for type, g in xtoa.sel(wl=wls).groupby('type'):
    # axs[0].plot(wl, y, color=cmap(norm(xfoam)), lw=1.5, markersize=2, alpha=0.5)
    p = g.plot.scatter('xfoam', param, ax=axs[i, 0], hue='wl', cmap=cmap, add_guide=False, )
    g.plot.scatter('xfoam', 'toa_rad', ax=axs[i, 1], hue='wl', cmap=cmap, add_guide=False)
    axs[i, 0].xaxis.label.set_visible(False)
    axs[i, 1].xaxis.label.set_visible(False)

    axs[i, 0].set_ylabel('TOA Reflectance $(-)$',fontsize=15)
    axs[i, 1].set_ylabel('TOA Radiance\n $(W\ m^{-2}\ sr^{-1}\ \mu m^{-1})$',fontsize=15)
    axs[i, 0].set_title(type)
    axs[i, 1].set_title(type)
    i += 1
axs[i - 1, 0].set_xlabel('Pixel coverage (%)')
axs[i - 1, 1].set_xlabel('Pixel coverage (%)')

cb = fig.colorbar(p, ax=axs, shrink=0.6, location='top')
cb.set_label('Wavelengths (%)')


cmap = plt.cm.Spectral_r
wls = [550, 670, 870, 1020, 1600, 2200]
symbol={550:'*',670:'D', 870:'s', 1020:'o', 1600:'p', 2200:'h'}
param = 'toa_refl'
norm = mpl.colors.Normalize(vmin=min(wls), vmax=max(wls))
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18,12))
fig.subplots_adjust(bottom=0.15, top=0.92, left=0.1, right=0.9,
                    hspace=0.3, wspace=0.2)
axs = axs.ravel()

for i,type in enumerate(xtoa.type.values):
    group=xtoa.sel(type=type,wl=wls)
    print(i, type)
    for wl,g in group.groupby('wl'):
        axs[i].plot(g.xfoam, g[param], color=cmap(norm(wl)),zorder=0)
        p = axs[i].scatter(g.xfoam, g[param], color=cmap(norm(wl)),marker=symbol[wl],alpha=0.95,label=str(wl)+" nm") #, edgecolors='black')
        # axs[i].scatter(g.concentration, g.hdrf_med, c=c, cmap=cmap, alpha=0.5)

    axs[i].set_title(type_label[type])
    i += 1
#axs[0].set_ylabel('TOA Reflectance')
axs[3].set_ylabel('TOA Reflectance',fontsize=22)
#axs[6].set_ylabel('TOA Reflectance')
#axs[6].set_xlabel('Pixel coverage (%)')
axs[7].set_xlabel('Percentage Pixel Coverage',fontsize=22)
#axs[8].set_xlabel('Pixel coverage (%)')

axs[7].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
          fancybox=True, shadow=True, ncol=6,handletextpad=0.,fontsize=20)
#cb = fig.colorbar(sm, ax=axs, shrink=0.6, aspect=30, fraction=.1,pad=0.08, location='bottom')
#cb.set_label('Wavelengths (nm)',labelpad=5)

plt.suptitle(resfile)
plt.savefig(opj(idir, 'fig', 'toa_vs_concentration_' + resfile + '_final2.pdf'))#, dpi=200)
plt.close()
