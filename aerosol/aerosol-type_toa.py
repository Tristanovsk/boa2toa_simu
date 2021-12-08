import os, copy
import glob
import numpy as np
import scipy.optimize as so
import pandas as pd
import xarray as xr
from multiprocessing.dummy import Pool
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.ioff()

plt.rcParams.update({'font.family': 'serif',
                     'font.size': 16, 'axes.labelsize': 20,
                     'mathtext.fontset': 'stix',
                     })
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

SixSexe_path = os.path.abspath('6SV1.1/sixsV1.1')
from Py6Sperso import *

opj = os.path.join
# idir = '/DATA/projet/VRTC/aerosol/'

# ---------------------------------------
#     6S VRT computation
# ---------------------------------------

wind_speed, wind_azimuth, salinity, pigment_concentration = 2, 0, 34, 0.3
xfoams = [0, 0.2, 0.4, 0.6, 0.8, 1]
xfoam = 0
ground_reflec = 0
sza =  10 # 40 #
vza, azi = 5, 90
aot = 0.2
aerosols = [['maritime', AeroProfile.Maritime],
            ['continental', AeroProfile.Continental],
            ['desert', AeroProfile.Desert]]
aerosol = aerosols[1]
amodel = aerosol[0]

resfile = 'toa_amodel_' + amodel + '_aot' + str(aot) + '_sza' + str(sza) + '_vza' + str(
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
    # SixSHelpers.Angles.run_and_plot_360(s, 'view', 'pixel_reflectance')
    df_ = []
    for type in rho_w.Sample_Name.values:
        for depth in rho_w.Depth.values:
            for spm in rho_w.SPM.values:
                print(type, depth, spm)
                group = rho_w.sel(Sample_Name=type, Depth=depth, SPM=spm)
                g = group.to_dataframe().reset_index().loc[:, ['wavelength', 'Rrs_Mean']]
                wl = g.wavelength


                def proc(p):
                    wl, Rw = p
                    wl = wl / 1000  # convert nm -> microns
                    print(wl)
                    s.outputs = None
                    a = copy.deepcopy(s)
                    a.wavelength = Wavelength(wl)
                    a.ground_reflectance = GroundReflectance.HomogeneousOcean(
                        wind_speed, wind_azimuth, salinity, Rw,
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

                df = pd.DataFrame(
                    {'type': type, 'depth': depth, 'spm': spm, 'wl': wl, 'toa_rad': toa_rad, 'toa_refl': toa_refl})
                df_.append(df)

    dff = pd.concat(df_)
    dff.to_csv(datafile, index=False)

xtoa = dff.set_index(['type', 'depth', 'spm', 'wl']).to_xarray()

# ------------------------------
# Plotting section
# ------------------------------
#cmap = plt.cm.RdYlGn_r

# axs = axs.ravel()
# axs[0].plot(wl, trans_tot, '--k')
norm = mpl.colors.Normalize(vmin=min(xtoa.depth), vmax=max(xtoa.depth))

sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

for param in ('toa_rad','toa_refl'):
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(18, 12), sharey=True, sharex=True)
    fig.subplots_adjust(bottom=0.15, top=0.92, left=0.1, right=0.9,
                        hspace=0.3, wspace=0.2)
    # axs = axs.ravel()

    for i, (name, x_) in enumerate(
            xtoa.sel(depth=[0.025, 0.05, 0.09, 0.12, 0.16, 0.32]).groupby('type')):
        for depth, x__ in x_.groupby('depth'):
            print(name)
            for ii, (spm, x___) in enumerate(x__.groupby('spm')):
                p = axs[i, ii].plot(x___.wl, x___[param] / np.pi, color=cmap(norm(depth)), label=str(depth) + ' m',
                                    zorder=0)
    for irow, sample in enumerate(xdata.Sample_Name.values):
        axs[irow][0].text(0.95, 0.95, sample, size=15,
                          transform=axs[irow][0].transAxes, ha="right", va="top",
                          bbox=dict(boxstyle="round",
                                    ec=(0.1, 0.1, 0.1),
                                    fc=plt.matplotlib.colors.to_rgba('blue', 0.15)))
        if param ==  'toa_rad':
            axs[irow][0].set_ylabel('TOA Radiance\n $(W\ m^{-2}\ sr^{-1}\ \mu m^{-1})$', fontsize=15)
        else:
            axs[irow][0].set_ylabel('TOA Reflectance $(-)$', fontsize=15)
    for ii, spm in enumerate(xdata.SPM.values):
        axs[0, ii].set_title('SPM = ' + str(spm) + ' mg/L')
        axs[-1, ii].set_xlabel('Wavelength (nm)')
    axs[-1, -2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
                       fancybox=True, shadow=True, ncol=7, handletextpad=0.1, fontsize=20)

    figname=param+'_from_submerged_plastics_amodel_' + amodel + '_aot' + str(aot) + '_sza' + str(
        sza) + '_vza' + str(vza) + '_azi' + str(azi)+'.png'

    plt.savefig(opj(idir, 'fig', figname), dpi=300)



#######
##
#
# END
#
##
#######