import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sys.path.extend([__file__+'/..'])

class rot:
    '''
    Rayleigh Optical Thickness for
    P=1013.25mb,
    T=288.15K,
    CO2=360ppm
    from
    Bodhaine, B.A., Wood, N.B, Dutton, E.G., Slusser, J.R. (1999). On Rayleigh
    Optical Depth Calculations, J. Atmos. Ocean Tech., 16, 1854-1861.
    '''

    def __init__(self):
        file = os.path.join(os.path.dirname(__file__),'rayleigh_bodhaine.txt')
        data = pd.read_csv(file, skiprows=16, sep=' ', header=None)
        data.columns = ('wl', 'rot', 'dpol')
        self.arr = data.set_index('wl').to_xarray()

    def get_data(self, wl):
        '''
        provide interpolated data for wavelength given in wl
        :param wl: scalar or array like values in nm
        :return:
        '''

        return self.arr.interp(wl=wl).rot.values

    def plot_data(self, wl):
        '''
        plot interpolated data for wavelength given in wl
        :param wl: scalar or array like values in nm
        :return:
        '''

        return plt.plot(wl, self.get_data(wl=wl))


class water_refractive_index:
    '''
    REFRACTIVE INDICES CALCULATION

    all the input wavelength in nm in vacuum
    '''

    def __init__(self):

        self.n_mc09 = pd.read_csv(os.path.join(os.path.dirname(__file__),"Max_Chapados_2009_data.txt"), sep=' ').set_index('wl_micron').to_xarray()
        self.n_hq73 = pd.read_csv(os.path.join(os.path.dirname(__file__),"RefractiveIndex_hale_querry_1973.csv"), sep='\s+').set_index(
            'wl_micron').to_xarray()

    def n_K2012(self, wl):
        '''
        Kedenburg et al, 2012 (applicability 0.5 to 1.6 µm)
        :param wl:
        :return:
        '''

        wl_sq = (wl / 1000) ** 2
        n = np.sqrt(1 + (0.75831 * wl_sq) / (wl_sq - 0.01007) + (0.08495 * wl_sq) / (wl_sq - 8.91377))
        return n

    def n_li2015(self, wl, salinity):
        '''
        Li et al, 2015 (applicability 300 to 2500nm)
        :param wl:
        :param salinity:
        :return:
        '''

        wlnm_sq = wl ** 2
        a = [0.385, 1.32, 1, 0.0244, 2.07e-5, 1.75e-7]
        l_sq = [8.79e3, 1.1e8, 6.09e4]
        n = np.sqrt(a[0] + a[1] * wlnm_sq / (wlnm_sq - l_sq[0]) +
                    a[2] * wlnm_sq / (wlnm_sq - l_sq[1]) +
                    a[3] * wlnm_sq / (wlnm_sq - l_sq[2])) + a[4] * salinity - a[5] * salinity ** 2
        return n

    def n_QF1997(self, wl, temp_C, salinity):
        '''
        parameterization from Quan and Fry, 1997 (applicability from 0.3 to 1 µm)
        :param wl:
        :param temp_C:
        :param salinity:
        :return:
        '''

        a = [1.31405e0, 1.779e-4, -1.05e-6, 1.6e-8, -2.02e-6, 1.5868e1, 1.155e-2, -4.23e-3, -4.382e3, 1.1455e6]
        n = a[0] + (a[1] + a[2] * temp_C + a[3] * temp_C ** 2) * salinity + \
            a[4] * temp_C ** 2 + (a[5] + a[6] * salinity + a[7] * temp_C) / wl + \
            a[8] * wl ** (-2) + a[9] * wl ** (-3)
        return (n)

    def n_MC2009(self, wl):
        '''
        tabulated values from Max Chapados 2009
        :param wl:
        :return:
        '''

        wl_micron = wl / 1000
        return self.n_mc09.interp(wl_micron=wl_micron).nw.values

    def n_HQ1973(self, wl):
        '''
        tabulated values from Hale and Querry 1973
        :param wl:
        :return:
        '''

        wl_micron = wl / 1000
        return self.n_hq73.interp(wl_micron=wl_micron).nw.values

    def nair(self, wl):
        '''
        refractive index of air
        :param wl:
        :return:
        '''

        k = [238.0185, 5792105, 57.362, 167917]
        nu = 1 / (wl / 1000)
        nair = 1 + 10 ** -8 * (k[2] / (k[1] - nu ** 2) + k[4] / (k[3] - nu ** 2))
        return nair

    def n_harmel(self, wl, temp_C=20, salinity=33):
        '''
        composite by Harmel et al., RSE, 2018
        :param wl: np.array in nm
        :param temp_C: array ni degC
        :param salinity: array in PSU
        :return:
        '''
        n = np.empty(wl.shape)
        for iwl, wl_ in enumerate(wl):
            if wl_ < 800:
                n[iwl] = self.n_QF1997(wl_, temp_C, salinity)
            elif wl_ < 1667:
                n[iwl] = self.n_K2012(wl_) \
                                   + (self.n_QF1997(800, temp_C, salinity) - self.n_K2012(800))
            else:
                n[iwl] = self.n_MC2009(wl_) + ( self.n_K2012(1667) \
                            + (self.n_QF1997(800, temp_C, salinity) - self.n_K2012(800)) \
                              - self.n_MC2009(1667))

        # n=n/self.nair(wl)
        return n

