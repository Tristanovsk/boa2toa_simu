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

# -----------------------------------
#      Data formatting
# -----------------------------------
idir = '/DATA/projet/garaba/boa_to_toa/Submerged_to_TOA/'
figdir = opj(idir, 'fig')

# -- load blank measurements

res = []
spm = {'LowTurb': 0., 'MedTurb': 75., 'HiTurb': 321.33}
NIR_offset = {'LowTurb': 0.004, 'MedTurb': 0.002, 'HiTurb': 0.0024}

for file in glob.glob(opj(idir, 'data', 'water', 'Water*.dat')):
    name = os.path.basename(file)
    type = name.split('_')[1]
    print(type)
    blank = pd.read_csv(file, sep='\s+')
    blank.columns = ['wl', 'median', 'mean', 'stdev']

    blank['median'] = blank['median'] - NIR_offset[type]
    blank['mean'] = blank['mean'] - NIR_offset[type]

    blank['spm'] = spm[type]
    blank['file'] = name
    res.append(blank)
blank = pd.concat(res)
blank_xr = blank.set_index(['wl', 'spm', 'file']).to_xarray()
blank_mean_xr = blank.groupby(['spm', 'wl']).mean().to_xarray()
blank_mean_xr.to_dataframe().to_csv('OP3/data/blanks_processed.csv')

fig, axs = plt.subplots(1, 1, figsize=(10, 4), sharey=True, sharex=True)
fig.subplots_adjust(bottom=0.15, top=0.92, left=0.1, right=0.95,
                    hspace=0.17, wspace=0.17)
raw = blank[(blank.wl > 350) & (blank.wl < 1350)]

raw.plot.scatter(x='wl', y='mean', c='spm', cmap=plt.cm.Spectral_r, ax=axs)

# -- load full measurements
df = pd.read_excel(opj(idir, 'data', 'Garaba_Dec2019_v2.xlsx'), sheet_name=1,
                   index_col=[0, 1, 2, 3, 4, 5, 6, 7, 8])
df.columns = df.columns.str.replace(' \[nm]', '').astype('float')
df.columns.name = 'wavelength'
dff = (df
       .dropna(how='all')
       .stack()
       .rename('Rrs')
       .reset_index()
       )
dff.columns = dff.columns.str.replace(' ', '_')
dff.columns = dff.columns.str.replace('Total_Suspended_Material_\[mg/L\]', 'SPM')
dff.columns = dff.columns.str.replace('Depth_\[m\]', 'Depth')

# dff=dff.set_index(['Sample_Name', 'Sample_ID', 'Sample_Condition', 'Date/Time',
#        'Latitude', 'Longitude', 'SPM',
#        'Depth', 'wavelength','Product']).unstack()
dff = dff.set_index(['Sample_Name', 'Sample_Condition',
                     'SPM',
                     'Depth', 'wavelength', 'Product']).unstack()
dff.columns = ['_'.join(col) for col in dff.columns]
dff.columns = [col.replace('Standard Deviation', 'SD') for col in dff.columns]

xdata_full = dff.to_xarray()

wls = xdata_full.wavelength
xdata = xdata_full.sel(wavelength=wls[(wls >= 350) & (wls < 1300)])

aw, bbw = ad.iopw().get_iopw(wls)

# -----------------------------
# get spectra of reference for VRT calculations
# -----------------------------

rho_w = xdata.sel(Sample_Condition='Wet').sel(Depth=[0.025, 0.05, 0.09, 0.12, 0.16, 0.32]).sel(
    wavelength=wls[(wls >= 350) & (wls <= 1200)]).Rrs_Mean
# reproject on new wavelength wls
wls = np.arange(350, 1200, 5)
rho_w = rho_w.interp(wavelength=wls)

# ---------------------------------------
#     6S VRT computation
# ---------------------------------------

wind_speed, wind_azimuth, salinity, pigment_concentration = 2, 0, 34, 0.3
xfoams = [0, 0.2, 0.4, 0.6, 0.8, 1]
xfoam = 0
ground_reflec = 0
sza = 40  # 40 #
vza, azi = 5, 90
aot_ref = 0.1

aerosols = [['maritime', AeroProfile.Maritime],
            ['continental', AeroProfile.Continental],
            ['desert', AeroProfile.Desert],
            ['urban', AeroProfile.Urban],
            ['biomass', AeroProfile.BiomassBurning], ]
aerosol = aerosols[0]
amodel = aerosol[0]

resfile = 'submerged_plastic_boa_toa_tot_amodel_' + amodel + '_aot'+str(
    aot_ref)+'_sza' + str(sza) + '_vza' + str(vza) + '_azi' + str(azi)
datafile = opj(idir, 'data', resfile + '.csv')

if os.path.exists(datafile):
    df = pd.read_csv(datafile)
    # df['aot'] = aot
    # df['sza'] = sza
    # df['vza'] = vza
    # df['azi'] = azi
    print(resfile)

else:

    df_ = []

    aot = int(aot_ref * 1000) / 1000

    s = SixS(SixSexe_path)
    s.geometry.solar_z = sza
    s.geometry.solar_a = 0
    s.geometry.view_z = vza
    s.geometry.view_a = azi
    s.altitudes.set_sensor_satellite_level()
    s.aot550 = aot
    s.aero_profile = AeroProfile.PredefinedType(aerosol[1])

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


                pool = Pool(processes=38)
                refl_spectrum = g.__array__()
                results = pool.map(proc, refl_spectrum)
                pool.close()
                pool.join()

                F0, trans_gas, trans_scat, irradiance, = [], [], [], []
                toa_refl, intrinsic_refl, toa_rad = [], [], []
                bg_refl, atmo_refl, pix_refl = [], [], []

                for res in results:
                    print(res.atmospheric_intrinsic_reflectance)
                    toa_refl = np.append(toa_refl,
                                         res.apparent_reflectance)
                    toa_rad = np.append(toa_rad,
                                        res.apparent_radiance)
                    bg_refl = np.append(bg_refl, res.background_reflectance)
                    atmo_refl = np.append(atmo_refl, res.atmospheric_intrinsic_reflectance)
                    pix_refl = np.append(pix_refl, res.pixel_reflectance)

                df = pd.DataFrame(
                    {'aerosol': amodel, 'aot_ref': aot_ref, 'aot': aot_ref, 'sza': sza, 'vza': vza,
                     'azi': azi,
                     'type': type, 'depth': depth, 'spm': spm, 'wl': wl, 'toa_rad': toa_rad,
                     'toa_refl': toa_refl, 'bg_refl': bg_refl,
                     'atmo_refl': atmo_refl, 'pix_refl': pix_refl})
                df_.append(df)

df = pd.concat(df_)
df.to_csv(datafile, index=False)

xtoa = df.set_index(['aerosol', 'aot_ref', 'aot', 'sza', 'vza', 'azi', 'type', 'depth', 'spm', 'wl']).to_xarray()
#xtoa.to_netcdf('./OP3/data/sixs_simu_aerosol_plastics.nc')

# ------------------------------
# Plotting section
# ------------------------------
cmap = mpl.colors.LinearSegmentedColormap.from_list("",
                                                    ['navy', "blue", 'lightskyblue',
                                                      'forestgreen', 'yellowgreen',
                                                      "gold", "darkgoldenrod",'orangered','orangered',
                                                     'orangered', "firebrick", ])


norm = mpl.colors.Normalize(vmin=0.05, vmax=0.5)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

colors = ['blue', 'sienna', 'orange', 'grey']
# cmap = plt.cm.RdYlGn_r
all_depths = [0.025, 0.05, 0.09, 0.12, 0.16, 0.32]
xtoa_ = xtoa.sel(aerosol='maritime',aot_ref=0.1,aot=0.1)
# axs = axs.ravel()
# axs[0].plot(wl, trans_tot, '--k')
norm = mpl.colors.Normalize(vmin=min(xtoa.depth), vmax=max(xtoa.depth))

sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
for param in ('toa_rad', 'toa_refl'):
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(16, 12), sharey=True, sharex=True)
    fig.subplots_adjust(bottom=0.115, top=0.962, left=0.086, right=0.98,
                        hspace=0.05, wspace=0.05)
    # axs = axs.ravel()

    for i, (name, x_) in enumerate(
            xtoa_.sel(sza=sza, vza=vza, azi=azi).sel(
                depth=all_depths).groupby('type')):
        for depth, x__ in x_.groupby('depth'):
            print(name)
            for ii, (spm, x___) in enumerate(x__.groupby('spm')):
                print(x___.aot)
                if np.isnan(x___[param].values).all():
                    axs[i, ii].set_axis_off()
                    continue
                axs[i, ii].minorticks_on()
                p = axs[i, ii].plot(x___.wl, x___[param], color=cmap(norm(depth)),
                                    label=str(depth) + ' m',
                                    zorder=0)
                # axs[i, ii].fill_between(x___.wl, x___.isel(aot=0)[param], x___.isel(aot=2)[param],
                #                         color=cmap(norm(depth)), alpha=0.3)
    for irow, sample in enumerate(xdata.Sample_Name.values):
        axs[irow][0].text(0.98, 0.96, sample, size=15,
                          transform=axs[irow][0].transAxes, ha="right", va="top",
                          bbox=dict(boxstyle="round",
                                    ec=(0.1, 0.1, 0.1),
                                    fc=plt.matplotlib.colors.to_rgba(colors[irow], 0.2)))
        if param == 'toa_rad':
            axs[irow][0].set_ylabel('TOA Radiance\n $(W\ m^{-2}\ sr^{-1}\ \mu m^{-1})$', fontsize=15)
        else:
            axs[irow][0].set_ylabel('TOA Reflectance $(-)$', fontsize=15)
    for ii, spm in enumerate(xdata.SPM.values):
        axs[0, ii].set_title('SPM = ' + str(spm) + ' mg/L')
        axs[-1, ii].set_xlabel('$Wavelength\ (nm)$')
    axs[-1, -2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
                       fancybox=True, shadow=True, ncol=7, handletextpad=0.1, fontsize=20)

    figname = param + '_from_submerged_plastics_amodel_' + str(xtoa_.aerosol.values) + '_aot' + str(
        xtoa_.aot.values) + '_sza' + str(
        sza) + '_vza' + str(vza) + '_azi' + str(azi) + '.png'

    plt.savefig(opj(idir, 'fig', 'toa', figname), dpi=300)

#######
##
#
# END
#
##
#######
