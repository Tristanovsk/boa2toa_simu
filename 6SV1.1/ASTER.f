      subroutine aster(iwa)
      common /sixs_ffu/ s(1501),wlinf,wlsup
      real sr(10,1501),wli(10),wls(10)
      real wlinf,wlsup,s
      integer iwa,l,i
   
c
c     aster vnir band 1
c
      data (sr(1,l),l=1,1501) / 
     & 94*0.,
     & .0075, .0078, .0082, .0066, .0062, .0253, .0580, .0906,
     & .1598, .2692, .3714, .4814, .6180, .7436, .8392, .9074,
     & .9552, .9864,1.0003, .9986, .9804, .9372, .8900, .8734,
     & .8807, .8941, .8986, .9056, .9226, .9246, .9184, .9292,
     & .9434, .9495, .9520, .9547, .9590, .9619, .9528, .9234,
     & .8976, .8760, .8254, .7622, .7057, .6308, .5206, .3792,
     & .2503, .1634, .1155, .0862, .0598, .0448, .0355, .0187,
     & .0065, .0046, .0040, .0038, .0053, .0053, .0037, .0000,
     & 1343*0./
c
c     aster vnir band 2
c
      data (sr(2,l),l=1,1501) / 
     & 136*0.,
     & .0038, .0036, .0008, .0004, .0028, .0027, .0006, .0041,
     & .0063, .0008, .0084, .0087, .0000, .1218, .2733, .4701,
     & .6735, .8347, .9709,1.0023, .9838, .9693, .9727, .9727,
     & .9430, .9089, .9271, .9467, .9365, .9197, .9100, .8986,
     & .8613, .8188, .7982, .7887, .7702, .7558, .6962, .6088,
     & .5179, .4509, .4258, .3819, .3240, .2597, .1855, .1112,
     & .0623, .0461, .0342, .0185, .0105, .0100, .0054, .0043,
     & .0020,
     & 1308*0./
c
c     aster vnir band 3n
c
      data (sr(3,l),l=1,1501) / 
     & 188*0.,
     & .0022, .0000, .0014, .0078, .0162, .0289, .0444, .0487,
     & .0509, .0738, .1178, .1774, .2504, .3628, .5441, .7950,
     &1.0000,1.0182, .9516, .9033, .8953, .9222, .9639, .9906,
     & .9942, .9847, .9752, .9658, .9444, .9372, .9617, .9783,
     & .9820, .9924, .9984, .9918, .9795, .9759, .9855, .9840,
     & .9694, .9574, .9540, .9514, .9326, .9269, .9528, .9619,
     & .9489, .9419, .9429, .9351, .8916, .8217, .7367, .6191,
     & .4813, .3574, .2609, .1892, .1278, .0831, .0610, .0451,
     & .0318, .0267, .0252, .0217, .0134, .0068, .0070, .0071,
     & .0049, .0030, .0024, .0025,
     & 1237*0./
c
c     aster vnir band 3b
c
      data (sr(4,l),l=1,1501) / 
     & 188*0.,
     & .0099, .0080, .0107, .0187, .0287, .0412, .0579, .0797,
     & .1218, .1996, .2872, .3827, .5204, .6754, .8016, .8701,
     & .8973, .9251, .9616, .9957,1.0056, .9967, .9808, .9566,
     & .9279, .9045, .8983, .9049, .9030, .9019, .9129, .9200,
     & .9138, .8983, .9004, .9186, .9144, .9022, .9032, .8988,
     & .8883, .8852, .8883, .8928, .8960, .9056, .9202, .9102,
     & .8863, .8811, .8794, .8617, .8219, .7809, .7470, .6763,
     & .5576, .4183, .3163, .2605, .1963, .1353, .1021, .0795,
     & .0594, .0436, .0282, .0148, .0125, .0148, .0140, .0156,
     & .0146, .0038, .0042, .0171, .0145, .0070, .0095, .0069,
     & .0000, .0013,
     & 1231*0./
c
c     aster swir band 4
c
      data (sr(5,l),l=1,1501) / 
     & 528*0.,
     & .0040, .0038, .0040, .0052, .0080, .0128, .0190, .0260,
     & .0330, .0450, .0740, .0990, .1520, .2140, .3050, .4200,
     & .5970, .6750, .7900, .8270, .8400, .8480, .9010, .9050,
     & .9100, .9140, .9050, .9260, .9470, .9670, .9760, .9840,
     & .9710, .9790, .9880,1.0000, .9920, .9840, .9780, .9710,
     & .9470, .9510, .9450, .9380, .9340, .9300, .9360, .9420,
     & .9140, .8970, .7980, .7000, .5970, .4610, .3700, .2630,
     & .1730, .1130, .0767, .0565, .0450, .0360, .0281, .0215,
     & .0160, .0117, .0083, .0058, .0040, .0027, .0019, .0014,
     & .0010, .0007, .0006, .0004, .0003, .0003, .0002, .0002,
     & 893*0./
c
c     aster swir band 5
c
      data (sr(6,l),l=1,1501) / 
     & 748*0.,
     & .0080, .0086, .0100, .0134, .0200, .0290, .0410, .0780,
     & .1310, .2050, .3030, .5410, .7050, .7790, .7910, .8030,
     & .8220, .8400, .9180, .9590,1.0000, .9750, .9020, .7790,
     & .6890, .5900, .4180, .3030, .2300, .1720, .1070, .0700,
     & .0610, .0498, .0385, .0273, .0160, .0140, .0120, .0100,
     & .0080, .0070, .0060, .0050, .0040, .0037, .0037, .0035,
     & .0032, .0031, .0029, .0027, .0026, .0024, .0023, .0021,
     & .0020, .0019, .0017, .0016, .0014, .0013, .0012, .0010,
     & .0009, .0008,
     & 687*0./
c
c     aster swir band 6
c
      data (sr(7,l),l=1,1501) / 
     & 760*0.,
     & .0080, .0097, .0117, .0138, .0160, .0186, .0223, .0281,
     & .0370, .0490, .0660, .1070, .1720, .2540, .3520, .5000,
     & .6020, .7420, .7620, .7790, .8520, .8690, .8860, .9020,
     & .9290, .9550, .9840,1.0000, .9340, .8200, .7540, .5160,
     & .3280, .2380, .1640, .1070, .0570, .0468, .0365, .0263,
     & .0160, .0140, .0120, .0100, .0080, .0070, .0060, .0050,
     & .0040, .0034, .0031, .0029, .0026, .0022, .0018, .0015,
     & .0011, .0007, .0004,
     & 682*0./
c
c     aster swir band 7
c
      data (sr(8,l),l=1,1501) / 
     & 784*0.,
     & .0080, .0113, .0152, .0197, .0250, .0330, .0490, .0700,
     & .1150, .1760, .2500, .3850, .5080, .6560, .7950, .8690,
     & .8480, .9100, .9100, .9260, .9260, .9430, .9590, .9750,
     & .9750,1.0000, .9590, .8690, .7990, .7050, .6230, .5000,
     & .3930, .3030, .2420, .1760, .1270, .1060, .0840, .0625,
     & .0410, .0370, .0330, .0290, .0250, .0226, .0214, .0200,
     & .0183, .0164, .0148, .0131, .0112, .0095, .0081, .0073,
     & .0069, .0067, .0066, .0064, .0060, .0055, .0050, .0045,
     & .0040, .0035, .0031, .0027, .0022, .0017, .0013, .0008,
     & .0004,
     & 644*0./
c
c     aster swir band 8
c
      data (sr(9,l),l=1,1501) / 
     & 800*0.,
     & .0023, .0051, .0079, .0103, .0120, .0129, .0134, .0142,
     & .0160, .0193, .0249, .0332, .0450, .0610, .0820, .1060,
     & .1390, .2040, .2860, .4490, .6040, .7020, .8330, .9710,
     & .9880, .9550, .9800, .9770, .9750, .9720, .9700, .9670,
     & .9470, .9620, .9770, .9920,1.0000, .9800, .9960, .9920,
     & .9960, .9550, .9630, .9060, .8370, .7840, .7020, .5800,
     & .4410, .3430, .2780, .2200, .1670, .1257, .0953, .0734,
     & .0570, .0440, .0336, .0257, .0200, .0163, .0141, .0129,
     & .0120, .0111, .0101, .0091, .0080, .0069, .0059, .0049,
     & .0040, .0032, .0024, .0018, .0012,
     & 624*0./
c
c     aster swir band 9
c
      data (sr(10,l),l=1,1501) / 
     & 819*0.,
     & .0004, .0009, .0015, .0021, .0029, .0037, .0047, .0057,
     & .0068, .0080, .0093, .0105, .0114, .0120, .0122, .0129,
     & .0151, .0200, .0290, .0450, .0650, .0780, .1100, .1550,
     & .2290, .3270, .4240, .5390, .7270, .7840, .9060, .9270,
     & .8980, .9000, .9010, .9030, .9040, .9060, .9310, .9270,
     & .9220, .9610,1.0000, .9800, .9590, .9270, .8940, .8690,
     & .8330, .8160, .7670, .7020, .6610, .5630, .4240, .3430,
     & .2610, .1920, .1392, .0992, .0698, .0490, .0347, .0253,
     & .0196, .0160, .0134, .0113, .0096, .0080, .0064, .0050,
     & .0041, .0040, .0050, .0066, .0085,
     & 605*0./

      wli(1)=.485
      wls(1)=.6425
      wli(2)=.590
      wls(2)=.730
      wli(3)=.720
      wls(3)=.9075
      wli(4)=.720
      wls(4)=.9225
      wli(5)=1.570
      wls(5)=1.7675
      wli(6)=2.120
      wls(6)=2.2825
      wli(7)=2.150
      wls(7)=2.295
      wli(8)=2.210
      wls(8)=2.390
      wli(9)=2.250
      wls(9)=2.440
      wli(10)=2.2975
      wls(10)=2.4875
      do 1 i=1,1501
      s(i)=sr(iwa,i)
    1 continue
      wlinf=wli(iwa)
      wlsup=wls(iwa)
      return
      end
