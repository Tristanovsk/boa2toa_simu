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

from OP3 import *

SixSexe_path = os.path.abspath('6SV1.1/sixsV1.1')
from Py6Sperso import *

from OP3 import data_utils as du

import RTxploitation as RT

RTp = RT.parameterization
ad = RT.auxdata

plot = False

idir = '/DATA/projet/VRTC/aerosol/' #'/DATA/projet/garaba/boa_to_toa/Submerged_to_TOA/'

# ---------------------------------------
#     6S VRT computation
# ---------------------------------------

wind_speed, wind_azimuth, salinity, pigment_concentration = 2, 0, 34, 0.3
xfoams = [0, 0.2, 0.4, 0.6, 0.8, 1]
xfoam = 0
ground_reflec = 0.0
sza = 40  # 40 #
vza, azi = 5, 90
suffix='visnir'
wl = np.linspace(350, 1050, 101)
suffix='visswir'
wl = np.linspace(400, 2500, 501)

aerosols = [['maritime', AeroProfile.Maritime],
            ['continental', AeroProfile.Continental],
            ['desert', AeroProfile.Desert],
            ['urban', AeroProfile.Urban],
            ['biomass', AeroProfile.BiomassBurning], ]

Rw = 0.

refl_spectrum = np.zeros((len(wl), 2))
refl_spectrum[:, 0] = wl
refl_spectrum[:, 1] = Rw
resfile = 'toa_aerosols_sza' + str(sza) + '_vza' + str(
    vza) + '_azi' + str(azi)+'_Rw'+str(Rw)+'_'+suffix
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
    s.altitudes.set_target_sea_level()

    df_ = []
    for aerosol in aerosols:
        # AOT uncertainty from Levy et al, 2013 Collection 6

        amodel = aerosol[0]
        s.aero_profile = AeroProfile.PredefinedType(aerosol[1])

        for aot in [0.05, 0.1, 0.2, 0.5, 0.7]:
            aot = int(aot * 1000) / 1000


            def proc(p):
                wl, Rw = p
                wl = wl / 1000  # convert nm -> microns
                print(wl)
                # s.outputs = None
                s = SixS(SixSexe_path)
                s.geometry.solar_z = sza
                s.geometry.solar_a = 0
                s.geometry.view_z = vza
                s.geometry.view_a = azi
                s.aot550 = aot
                s.altitudes.set_sensor_satellite_level()
                # s.altitudes.set_target_sea_level()
                s.aero_profile = AeroProfile.PredefinedType(aerosol[1])
                a = copy.deepcopy(s)
                a.wavelength = Wavelength(wl)
                a.ground_reflectance = GroundReflectance.HomogeneousOcean(
                    wind_speed, wind_azimuth, salinity, Rw,
                    ground_reflec, xfoam)
                a.run()
                return a.outputs


            pool = Pool()

            results = pool.map(proc, refl_spectrum)
            pool.close()
            pool.join()

            F0, trans_gas, trans_scat, irradiance, = [], [], [], []
            toa_refl, intrinsic_refl, toa_rad = [], [], []
            bg_refl, atmo_refl, pix_refl = [], [], []
            for res in results:
                print('aot, TT:', aot, res.transmittance_total_scattering.total)
                toa_refl = np.append(toa_refl, res.apparent_reflectance)
                toa_rad = np.append(toa_rad, res.apparent_radiance)
                bg_refl = np.append(bg_refl, res.background_reflectance)
                atmo_refl = np.append(atmo_refl, res.atmospheric_intrinsic_reflectance)
                pix_refl = np.append(pix_refl, res.pixel_reflectance)

            df = pd.DataFrame(
                {'type': amodel, 'aot': aot, 'wl': wl, 'toa_rad': toa_rad, 'toa_refl': toa_refl, 'bg_refl': bg_refl,
                 'atmo_refl': atmo_refl, 'pix_refl': pix_refl})
            df_.append(df)

    dff = pd.concat(df_)
    dff.to_csv(datafile, index=False)

xdata = dff.set_index(['type', 'aot', 'wl']).to_xarray()

cmap = plt.cm.Spectral_r
norm = mpl.colors.Normalize(vmin=0.05, vmax=0.7)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
colors = ['forestgreen', "darkgoldenrod", 'orangered', 'lightskyblue', 'black']

# -------------------------------------------
# Plot TOA reflectance for each AOT
# -------------------------------------------

fig, axs = plt.subplots(2, 3, figsize=(20, 10), sharey=True)  # , sharex=True)# ,
axs = axs.ravel()

fig.subplots_adjust(bottom=0.12, top=0.95, left=0.08, right=0.92,
                    hspace=0.38, wspace=0.05)
params = ['atmo_refl', 'pix_refl', 'bg_refl', 'toa_refl']
ls = ['o:', 'o-', 'o:']
for i, (aot, xdata_) in enumerate(xdata.groupby('aot')):
    print(aot)
    # xdata_ = xdata_.dropna('aot')

    for itype, (type, x_) in enumerate(xdata_.groupby('type')):
        axs[i].plot(x_.wl, x_['toa_refl'], '-', color=colors[itype], label=type)
    axs[i].set_title('aot = ' + str(aot))
for i in range(5):
    axs[i].set_xlabel('Wavelength (nm)')

axs[0].set_ylabel('$R_{apparent}$')
axs[3].set_ylabel('$R_{apparent}$')
axs[-1].set_axis_off()
axs[-2].legend(loc='upper center', bbox_to_anchor=(1.55, .95),
               fancybox=True, shadow=True, ncol=2, handletextpad=0.1, fontsize=20)

# plt.suptitle(amodel)

plt.savefig(opj(idir, 'fig', 'toa_reflectances_vs_aerosol_type_Rw'+str(Rw)+suffix+'.png'), dpi=300)

for i in range(5):
    axs[i].semilogy()
axs[0].set_ylim([0.0001, 0.3])
plt.savefig(opj(idir, 'fig', 'toa_reflectances_vs_aerosol_type_Rw'+str(Rw)+'_log'+suffix+'.png'), dpi=300)

plt.close()

# -------------------------------------------
# Plot TOA reflectance for each aerosol type
# -------------------------------------------

fig, axs = plt.subplots(2, 3, figsize=(20, 10), sharey=True)  # , sharex=True)# ,
axs = axs.ravel()

fig.subplots_adjust(bottom=0.12, top=0.92, left=0.08, right=0.92,
                    hspace=0.38, wspace=0.05)
for itype, (type, xdata_) in enumerate(xdata.groupby('type')):
    for i, (aot, x_) in enumerate(xdata_.groupby('aot')):
        axs[itype].plot(x_.wl, x_['toa_refl'], '-', color=cmap(norm(aot)))
    axs[itype].set_title(type)
for i in range(5):
    axs[i].set_xlabel('Wavelength (nm)')

axs[0].set_ylabel('$R_{apparent}$')
axs[3].set_ylabel('$R_{apparent}$')
axs[-1].set_axis_off()

cb = fig.colorbar(sm,ax=axs, shrink=0.6, aspect=30, location='top')
cb.set_label('AOT (-)')
# plt.suptitle(amodel)

plt.savefig(opj(idir, 'fig', 'toa_reflectances_vs_aot_Rw'+str(Rw)+suffix+'.png'), dpi=300)

for i in range(5):
    axs[i].semilogy()
axs[0].set_ylim([0.0001, 0.3])
plt.savefig(opj(idir, 'fig', 'toa_reflectances_vs_aot_Rw'+str(Rw)+'_log'+suffix+'.png'), dpi=300)

# -------------------------------------------
# Plot all in one
# -------------------------------------------
log=True #False
fig, axs = plt.subplots(1, 1, figsize=(16, 8), sharey=True)  # , sharex=True)# ,

fig.subplots_adjust(bottom=0.12, top=0.95, left=0.12, right=0.92,
                    hspace=0.38, wspace=0.05)
params = ['atmo_refl', 'pix_refl', 'bg_refl', 'toa_refl']
ls = [':', '-', '--', '-.', 'o-']
lw=[2,2,2,2,3]
for i, (aot, xdata_) in enumerate(xdata.isel(aot=[0,2,4]).groupby('aot')):
    print(aot)
    # xdata_ = xdata_.dropna('aot')

    for itype, (type, x_) in enumerate(xdata_.groupby('type')):

        axs.plot(x_.wl, x_['toa_refl'], ls[itype], ms=lw[itype], color=cmap(norm(aot)),alpha=1.
                 )


axs.set_xlabel('Wavelength (nm)')
axs.set_ylabel('$R_{apparent}$')
lines = [plt.Line2D([0], [0], color='grey', linewidth=2, ls=l) for l in ls[:-1]]
lines.append(
    plt.Line2D([0], [0], color='grey', linewidth=2, ls='-', marker='o'))
if log :
    axs.semilogy()
    axs.set_ylim([0.0001, 0.2])
axs.legend(lines, xdata.type.values, loc='best', fancybox=True, shadow=True, ncol=2, handletextpad=0.1,
           fontsize=20)


cb = fig.colorbar(sm, ax=axs, shrink=0.6, aspect=30)
cb.set_label('AOT (-)')

# plt.suptitle(amodel)
if log :
    plt.savefig(opj(idir, 'fig', 'toa_reflectances_vs_aerosol_type_allinone_Rw'+str(Rw)+'_log'+suffix+'.png'), dpi=300)
else:
    plt.savefig(opj(idir, 'fig', 'toa_reflectances_aerosol_type_allinone_Rw'+str(Rw)+suffix+'.png'), dpi=300)


#######
##
#
# END
#
##
#######
