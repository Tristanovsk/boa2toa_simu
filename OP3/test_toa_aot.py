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

idir = '/DATA/projet/garaba/boa_to_toa/Submerged_to_TOA/'

# ---------------------------------------
#     6S VRT computation
# ---------------------------------------

wind_speed, wind_azimuth, salinity, pigment_concentration = 2, 0, 34, 0.3
xfoams = [0, 0.2, 0.4, 0.6, 0.8, 1]
xfoam = 0
ground_reflec = 0.0
sza = 40  # 40 #
vza, azi = 10, 90
wl = np.linspace(450, 1100, 21)
wl = np.linspace(450, 2500, 21)

df_ = []
for aot_ref in 0.05, 0.2, 0.5:
    # AOT uncertainty from Levy et al, 2013 Collection 6
    sig_aot = aot_ref * 0.25 + 0.05
    aerosols = [['maritime', AeroProfile.Maritime],
                ['continental', AeroProfile.Continental],
                ['desert', AeroProfile.Desert],
                ['urban', AeroProfile.Urban],
                ['biomass', AeroProfile.BiomassBurning], ]
    aerosol = aerosols[-2]
    amodel = aerosol[0]
    Rw = 0.3

    refl_spectrum = np.zeros((len(wl), 2))
    refl_spectrum[:, 0] = wl
    refl_spectrum[:, 1] = Rw

    s = SixS(SixSexe_path)
    s.geometry.solar_z = sza
    s.geometry.solar_a = 0
    s.geometry.view_z = vza
    s.geometry.view_a = azi
    s.altitudes.set_sensor_satellite_level()
    s.altitudes.set_target_sea_level()
    s.aero_profile = AeroProfile.PredefinedType(aerosol[1])

    for aot in [aot_ref, aot_ref + sig_aot, np.max([0., aot_ref - sig_aot])]:
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
            {'aot_ref': aot_ref, 'aot': aot, 'type': amodel, 'wl': wl, 'toa_rad': toa_rad, 'toa_refl': toa_refl,
             'bg_refl': bg_refl,
             'atmo_refl': atmo_refl, 'pix_refl': pix_refl})
        df_.append(df)

dff = pd.concat(df_)
xdata = dff.set_index(['aot_ref', 'aot', 'wl']).to_xarray()

cmap = plt.cm.Spectral_r
cmap = mpl.colors.LinearSegmentedColormap.from_list("",
                                                    ['royalblue',
                                                     "grey",  'gold',
                                                     "khaki", "gold",
                                                     'slategrey'])
norm = mpl.colors.Normalize(vmin=0.05, vmax=0.5)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])


fig, axs = plt.subplots(1, 4, figsize=(25, 6), sharey=True, sharex=True)
fig.subplots_adjust(bottom=0.15, top=0.82, left=0.08, right=0.95,
                    hspace=0.3, wspace=0.2)
params = ['atmo_refl', 'pix_refl', 'bg_refl', 'toa_refl']
ls = ['o:', 'o-', 'o:']
for aot_ref, xdata_ in xdata.groupby('aot_ref'):
    print(aot_ref)
    xdata_ = xdata_.dropna('aot')

    for i in range(4):
        # for iaot in range(3):
        #     axs[i].plot(xdata_.wl, xdata_.isel(aot=iaot)[params[i]],ls[iaot],color=cmap(norm(xdata_.aot.values[iaot])))
        axs[i].plot(xdata_.wl, xdata_.isel(aot=1)[params[i]], 'o-', color=cmap(norm(aot_ref)),label=str(aot_ref))
        axs[i].fill_between(xdata_.wl, xdata_.isel(aot=0)[params[i]], xdata_.isel(aot=2)[params[i]],
                            color=cmap(norm(aot_ref)), alpha=0.3)
for i in range(4):
    axs[i].set_xlabel('Wavelength (nm)')
axs[0].set_ylabel('$R_{atmo}$')
axs[1].set_ylabel('$R_{pixel}$')
axs[2].set_ylabel('$R_{background}$')
axs[3].set_ylabel('$R_{apparent}$')

# plt.suptitle(amodel)
axs[0].legend( loc='upper center', bbox_to_anchor=(0.5, 0.97),title='AOT (550 nm)',
           fancybox=True, shadow=True, ncol=3, handletextpad=0.1, fontsize=18)

# cb = fig.colorbar(sm, ax=axs, shrink=0.6, aspect=30, location='top')
# cb.set_label('AOT (-)')
plt.suptitle(amodel.capitalize()+' aerosols',fontsize=32)
plt.savefig(opj(idir, 'fig', 'test_toa_reflectances_' + amodel + '_Rw0.3.png'), dpi=300)
plt.close()

#######
##
#
# END
#
##
#######
