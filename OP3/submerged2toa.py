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
vza, azi = 15, 90
aot_ref = 0.2
# AOT uncertainty from Levy et al, 2013 Collection 6
sig_aot = aot_ref * 0.2 + 0.05

aerosols = [['maritime', AeroProfile.Maritime],
            ['continental', AeroProfile.Continental],
            ['desert', AeroProfile.Desert],
            ['urban', AeroProfile.Urban],
            ['biomass', AeroProfile.BiomassBurning], ]

depths = [0.025, 0.12, 0.32]
szas = [15, 30, 60]
vzas = [5, 15, 30]
azis = [45, 75]
df__ = []
for iaerosol in [0,1,3]:
    aerosol = aerosols[iaerosol]

    amodel = aerosol[0]
    for sza in szas:
        for vza in vzas:
            for azi in azis:
                resfile = 'submerged_plastic_boa_toa_tot_amodel_' + amodel + '_sza' + str(
                    sza) + '_vza' + str(
                    vza) + '_azi' + str(azi)
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
                    for aot_ref in 0.05, 0.2, 0.5:
                        sig_aot = aot_ref * 0.25 + 0.05
                        for aot in [aot_ref, aot_ref + sig_aot, np.max([0., aot_ref - sig_aot])]:
                            aot = int(aot * 1000) / 1000

                            s = SixS(SixSexe_path)
                            s.geometry.solar_z = sza
                            s.geometry.solar_a = 0
                            s.geometry.view_z = vza
                            s.geometry.view_a = azi
                            s.altitudes.set_sensor_satellite_level()
                            s.aot550 = aot
                            s.aero_profile = AeroProfile.PredefinedType(aerosol[1])

                            for type in rho_w.Sample_Name.values:
                                for depth in depths:  # rho_w.Depth.values:
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


                                        pool = Pool(processes=34)
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
                                            {'aerosol': amodel, 'aot_ref': aot_ref, 'aot': aot, 'sza': sza, 'vza': vza,
                                             'azi': azi,
                                             'type': type, 'depth': depth, 'spm': spm, 'wl': wl, 'toa_rad': toa_rad,
                                             'toa_refl': toa_refl, 'bg_refl': bg_refl,
                                             'atmo_refl': atmo_refl, 'pix_refl': pix_refl})
                                        df_.append(df)

                        df = pd.concat(df_)
                        df.to_csv(datafile, index=False)
                df__.append(df)

dff_ = pd.concat(df__)
xtoa = dff_.set_index(['aerosol', 'aot_ref', 'aot', 'sza', 'vza', 'azi', 'type', 'depth', 'spm', 'wl']).to_xarray()
#xtoa.to_netcdf('./OP3/data/sixs_simu_aerosol_plastics.nc')

# ------------------------------
# Plotting section
# ------------------------------

depths = [0.025, 0.12, 0.32]
depth = depths[0]
cmap = mpl.colors.LinearSegmentedColormap.from_list("",
                                                    ['royalblue',
                                                     "grey", 'gold',
                                                     "khaki", "gold",
                                                     'slategrey'])

norm = mpl.colors.Normalize(vmin=0.05, vmax=0.5)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

# -------------------------
# plot 6SV all component for two aerosol types (e.g., maritime and urban)
# -------------------------
spm=0. # in 0.0 75.0 321.33
sza, vza, azi = (30, 5, 75)
fig, axs = plt.subplots(3, 4, figsize=(17, 12), sharey=True, sharex=True)
fig.subplots_adjust(bottom=0.1, top=0.875, left=0.08, right=0.985,
                    hspace=0.05, wspace=0.05)
params = ['atmo_refl', 'pix_refl', 'bg_refl', 'toa_refl']
ls = ['o:', 'o-', 'o:']
for imodel, aerosol in enumerate(['maritime','continental','urban']):
    _xdata = xtoa.sel(aerosol=aerosol).sel(sza=sza, vza=vza, azi=azi, depth=depth, spm=spm).isel(type=3)
    for aot_ref, xdata_ in _xdata.groupby('aot_ref'):
        print(aot_ref)
        xdata_ = xdata_.squeeze()
        xdata_ = xdata_.dropna('aot')
        if len(xdata_.aot) == 0:
            print(type, spm)
            plt.close()
            continue
        for i in range(4):
            # for iaot in range(3):
            #     axs[i].plot(xdata_.wl, xdata_.isel(aot=iaot)[params[i]],ls[iaot],color=cmap(norm(xdata_.aot.values[iaot])))
            axs[imodel, i].plot(xdata_.wl, xdata_.isel(aot=1)[params[i]], '-', lw=2, color=cmap(norm(aot_ref)),
                                label=str(aot_ref))
            axs[imodel, i].fill_between(xdata_.wl, xdata_.isel(aot=0)[params[i]], xdata_.isel(aot=2)[params[i]],
                                        color=cmap(norm(aot_ref)), alpha=0.3)
            axs[imodel, i].minorticks_on()

    # plt.suptitle(amodel)
    axs[imodel, 0].legend(loc='upper center', bbox_to_anchor=(0.5, 0.97), title='$aot(550 nm),\ $'+aerosol.capitalize(),
                     fancybox=True, shadow=True, ncol=3, handletextpad=0.1, fontsize=14)
for i in range(4):
    axs[-1, i].set_xlabel('$Wavelength\ (nm)$')

axs[0, 0].set_ylabel('$R^{TOA} (-)$')
axs[1, 0].set_ylabel('$R^{TOA} (-)$')
axs[2, 0].set_ylabel('$R^{TOA} (-)$')
axs[0, 0].set_title('$R_{atm}^{TOA}$')
axs[0, 1].set_title('$R_{w}^{TOA}$')
axs[0, 2].set_title('$R_{env}^{TOA}$')
axs[0, 3].set_title('$R_{tot}^{TOA}$')
# cb = fig.colorbar(sm, ax=axs, shrink=0.6, aspect=30, location='top')
# cb.set_label('AOT (-)')
plt.suptitle(' Distinct aerosol types, depth=' + str(depth) + 'm, spm=' + str(
    spm) + 'mg/L, sza=' + str(sza) + ', vza=' + str(vza) + ', azi=' + str(azi), fontsize=18)
plt.savefig(opj(idir, 'fig', 'toa',
                'toa_reflectances_maritime_cont_urban_' + type + '_spm' + str(spm) + '_depth' + str(
                    depth) + '_sza' + str(sza) + '_vza' + str(vza) + '_azi' + str(azi) + '.png'), dpi=300)


# -------------------------
# plot 6SV Rtot - Renv (no adjacency)
# -------------------------
spm=321.33 #0.0 # in 0.0 75.0 321.33
sza, vza, azi = (30, 5, 75)
fig, axs = plt.subplots(3, 4, figsize=(17, 12), sharey=True, sharex=True)
fig.subplots_adjust(bottom=0.1, top=0.875, left=0.08, right=0.985,
                    hspace=0.05, wspace=0.05)
params = ['atmo_refl', 'pix_refl', 'bg_refl', 'toa_refl']
param = 'toa_noadj'
ls = ['o:', 'o-', 'o:']
for imodel, aerosol in enumerate(['maritime','continental','urban']):
    xtoa_ = xtoa.sel(aerosol=aerosol).sel(sza=sza, vza=vza, azi=azi, depth=depth, spm=spm)
    for itype, (type, xdata) in enumerate(xtoa_.groupby('type')):
        for aot_ref, xdata_ in xdata.groupby('aot_ref'):

            xdata_ = xdata_.squeeze()
            xdata_ = xdata_.dropna('aot')
            xdata_['toa_noadj']=xdata_['toa_refl']-xdata_['bg_refl']
            if len(xdata_.aot) == 0:
                print(type, spm)

                continue

            axs[imodel, itype].plot(xdata_.wl, xdata_.isel(aot=1)[param], '-', lw=2, color=cmap(norm(aot_ref)),
                                label=str(aot_ref))
            axs[imodel, itype].fill_between(xdata_.wl, xdata_.isel(aot=0)[param], xdata_.isel(aot=2)[param],
                                        color=cmap(norm(aot_ref)), alpha=0.3)
            axs[imodel, itype].minorticks_on()
        if imodel == 0:
            axs[imodel, itype].set_title(type)

    # plt.suptitle(amodel)
    axs[imodel, 0].legend(loc='upper center', bbox_to_anchor=(0.5, 0.97), title='$aot(550 nm),\ $'+aerosol.capitalize(),
                     fancybox=True, shadow=True, ncol=3, handletextpad=0.1, fontsize=14)
for i in range(4):
    axs[-1, i].set_xlabel('$Wavelength\ (nm)$')

axs[0, 0].set_ylabel('$R^{TOA} (-)$')
axs[1, 0].set_ylabel('$R^{TOA} (-)$')
axs[2, 0].set_ylabel('$R^{TOA} (-)$')

# cb = fig.colorbar(sm, ax=axs, shrink=0.6, aspect=30, location='top')
# cb.set_label('AOT (-)')
plt.suptitle(' Distinct aerosol types, depth=' + str(depth) + 'm, spm=' + str(
    spm) + 'mg/L, sza=' + str(sza) + ', vza=' + str(vza) + ', azi=' + str(azi), fontsize=18)
plt.savefig(opj(idir, 'fig', 'toa',
                'toa_reflectances_maritime_cont_urban_all_plastics_spm' + str(spm) + '_depth' + str(
                    depth) + '_sza' + str(sza) + '_vza' + str(vza) + '_azi' + str(azi) + '.png'), dpi=300)

if False:
    for sza in szas:
        for vza in vzas:
            for azi in azis:
                xtoa_ = xtoa.sel(sza=sza, vza=vza, azi=azi).sel(depth=depth)
                for type, xdata in xtoa_.groupby('type'):
                    for spm, _xdata in xdata.groupby('spm'):

                        fig, axs = plt.subplots(1, 4, figsize=(17, 5), sharey=True, sharex=True)
                        fig.subplots_adjust(bottom=0.15, top=0.85, left=0.08, right=0.985,
                                            hspace=0.3, wspace=0.05)
                        params = ['atmo_refl', 'pix_refl', 'bg_refl', 'toa_refl']
                        ls = ['o:', 'o-', 'o:']

                        for aot_ref, xdata_ in _xdata.groupby('aot_ref'):
                            print(aot_ref)
                            xdata_ = xdata_.squeeze()
                            xdata_ = xdata_.dropna('aot')
                            if len(xdata_.aot) == 0:
                                print(type, spm)
                                plt.close()
                                continue
                            for i in range(4):
                                # for iaot in range(3):
                                #     axs[i].plot(xdata_.wl, xdata_.isel(aot=iaot)[params[i]],ls[iaot],color=cmap(norm(xdata_.aot.values[iaot])))
                                axs[i].plot(xdata_.wl, xdata_.isel(aot=1)[params[i]], '-', lw=2, color=cmap(norm(aot_ref)),
                                            label=str(aot_ref))
                                axs[i].fill_between(xdata_.wl, xdata_.isel(aot=0)[params[i]], xdata_.isel(aot=2)[params[i]],
                                                    color=cmap(norm(aot_ref)), alpha=0.3)
                        for i in range(4):
                            axs[i].set_xlabel('$Wavelength\ (nm)$')
                            axs[i].minorticks_on()
                        axs[0].set_ylabel('$R^{TOA} (-)$')
                        axs[0].set_title('$R_{atm}^{TOA}$')
                        axs[1].set_title('$R_{w}^{TOA}$')
                        axs[2].set_title('$R_{env}^{TOA}$')
                        axs[3].set_title('$R_{tot}^{TOA}$')

                        # plt.suptitle(amodel)
                        axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 0.97), title='$aot(550 nm)$',
                                      fancybox=True, shadow=True, ncol=3, handletextpad=0.1, fontsize=14)

                        # cb = fig.colorbar(sm, ax=axs, shrink=0.6, aspect=30, location='top')
                        # cb.set_label('AOT (-)')
                        plt.suptitle(amodel.capitalize() + ' aerosols, ' + type + ', depth=' + str(depth) + 'm, spm=' + str(
                            spm) + 'mg/L, sza=' + str(
                            sza) + ', vza=' + str(vza) + ', azi=' + str(azi), fontsize=18)
                        plt.savefig(opj(idir, 'fig', 'toa',
                                        'toa_reflectances_' + amodel + '_' + type + '_spm' + str(spm) + '_depth' + str(
                                            depth) + '_sza' + str(
                                            sza) + '_vza' + str(vza) + '_azi' + str(azi) + '.png'), dpi=300)
                        plt.close()


plt.show()

colors = ['blue', 'sienna', 'orange', 'grey']
# cmap = plt.cm.RdYlGn_r
all_depths = [0.025, 0.05, 0.09, 0.12, 0.16, 0.32]
xtoa_ = xtoa.sel(aerosol='maritime',aot_ref=0.2,aot=0.1)
# axs = axs.ravel()
# axs[0].plot(wl, trans_tot, '--k')
norm = mpl.colors.Normalize(vmin=min(xtoa.depth), vmax=max(xtoa.depth))

sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
for sza in [15, 30, 60]:
    for vza in [5, 15, 30]:
        for azi in [45]:
            for param in ('toa_rad', 'toa_refl'):
                fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(18, 14), sharey=True, sharex=True)
                fig.subplots_adjust(bottom=0.15, top=0.92, left=0.1, right=0.9,
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
                    axs[-1, ii].set_xlabel('Wavelength (nm)')
                axs[-1, -2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
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
