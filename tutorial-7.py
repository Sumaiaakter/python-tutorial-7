#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 13:58:35 2022

@author: sarmg
"""

import numpy as np
from astropy.io import ascii
from astropy.table import Table, Column
from astropy.io import fits
import matplotlib.pyplot as plt
import photutils
from scipy import optimize as optimise


import imexam


data='/Users/sarmg/Downloads/Tutorial_VII/data/'




image_data=fits.open(data+'MOO_1506+5136_gmos_r.fits')

image_get_data=image_data[0].data
#print(image_get_data)

r_image_header=image_data[0].header
#print(r_image_header)
#-------------------865,2512----------------
r_x_values=1647*np.random.random_sample((10**6))+865
#print(r_x_values)
#[2511.95751432 1453.74471929 2128.92560434 ... 1313.88092878  896.64642415
 #1338.16959369]

#-------------350,1805----------------------
r_y_values=1455*np.random.random_sample((10**6))+350
#print(r_y_values)
#[ 892.73487368  962.55267113 1786.7103062  ... 1446.59472944 1683.55099057
 #1717.57723677]

r_image_shape=np.array([r_x_values,r_y_values])

r_image_shape.shape #r_image_shape.shape  Out[8]: (2, 1000000)



r_im=r_image_shape.T

r_im.shape  #Out[11]: (1000000, 2)
#print(len(image_shape))

z_image_data=fits.open(data+'MOO_1506+5136_gmos_Z.fits')
image_get_data=image_data[0].data
#print(image_get_data)

z_image_header=image_data[0].header
#print(z_image_header)

z_x_values=1494*np.random.random_sample((10**6))+919    #[1154.09685149 2234.78736275 1616.0290553  ... 2340.65057588 1937.17746287
 #1790.62833372]

#-------------350,1805----------------------
z_y_values=1998*np.random.random_sample((10**6))+63    #[1583.13122519 2026.73654727 1727.24920498 ... 1696.37151207  548.78033423
  #392.76433436]

z_image_shape=np.array([z_x_values,z_y_values])

z_image_shape.shape     #(2, 1000000)

z_im=z_image_shape.T

z_im.shape       #(1000000, 2)


r_header = r_image_header['CD2_2']
r_pixel = r_header*3600
r_arc = 2/r_pixel   #print(r_arc)  12.343204098534102


header=z_image_header['CD2_2']
#print(header)

pixel=header*3600
#print(pixel)

arcsec=2/pixel   #print(arcsec)   12.343204098534102  this is diameter

#print(arcsec)




#------------------ch1--calculate radius-------------

image_ch1=fits.open(data+'MOO_1506+5136_irac_ch1.fits')
image_header_ch1=image_ch1[0].header

#print(image_header_ch1)
header_ch1=image_header_ch1['CD2_2']
#print(header_ch1)

pixel_ch1=header_ch1*3600
#print(pixel_ch1)

diameter_ch1=4/pixel_ch1  #6.665333599946676
#print(diameter_ch1)
#radius_ch1=6.665333599946676/2
#Out[69]: 3.332666799973338


#----------------for ch1 images bound x(509.232,1048.53 and y(1048.98,1626.71)
ch1_x_location=539.298*np.random.random_sample((10**6,))+509.232
ch1_y_location= 577.73*np.random.random_sample((10**6,))+1048.98
spitzer_ch1_location = np.array([ch1_x_location, ch1_y_location])
spitzer_ch1_location.shape #Out[75]: (2, 1000000)




#------------------ch2---calculate radius------------------

image_ch2=fits.open(data+'MOO_1506+5136_irac_ch2.fits')
image_header_ch2=image_ch2[0].header
#print(image_header_ch2)
header_ch2=image_header_ch2['CD2_2']
#print(header_ch2)

pixel_ch2=header_ch2*3600
#print(pixel_ch2)

diameter_ch2=4/pixel_ch2
#print(diameter_ch2)  #6.665333599946676
#6.665333599946676/2
#Out[71]: 3.332666799973338


#------------------ch2 images bound x(879.582,1399.99) and y(532.462, 1036.16)
ch2_x_location=520.408*np.random.random_sample((10**6,))+879.582
ch2_y_location= 503.698*np.random.random_sample((10**6,))+532.462
spitzer_ch2_location = np.array([ch2_x_location, ch2_y_location])
spitzer_ch2_location.shape   #Out[2]: (2, 1000000)


#--------transpose--array-----------------------

ch1_array = spitzer_ch1_location.T
ch2_array = spitzer_ch2_location.T
ch1_array.shape  #Out[5]: (1000000, 2)
ch2_array.shape

#-------------aperture  array--------------------

#----------------------------------------------

r_aperture=photutils.aperture.CircularAperture(r_im,r=6.665333599946676)

#print(r_aperture)

r_gemini_data=fits.open('/Users/sarmg/Downloads/Tutorial_VII/data/MOO_1506+5136_gmos_r.fits')
gemini_r_data=fits.getdata('/Users/sarmg/Downloads/Tutorial_VII/data/MOO_1506+5136_gmos_r.fits')
#print(gemini_r_data)

phot_rcat=photutils.aperture.aperture_photometry(gemini_r_data, r_aperture)
#print(phot_rcat)



z_aperture=photutils.aperture.CircularAperture(z_im,r=6.665333599946676)

#print(z_aperture)

z_gemini_data=fits.open('/Users/sarmg/Downloads/Tutorial_VII/data/MOO_1506+5136_gmos_z.fits')
gemini_z_data=fits.getdata('/Users/sarmg/Downloads/Tutorial_VII/data/MOO_1506+5136_gmos_z.fits')
#print(gemini_z_data)

phot_zcat=photutils.aperture.aperture_photometry(gemini_z_data, z_aperture)
#print(phot_zcat)
#---------------step-12---------------------

r_aperture_sum=phot_rcat['aperture_sum']
print(r_aperture_sum)

z_aperture_sum=phot_zcat['aperture_sum']
#print(z_aperture_sum)

ch1_aperture = photutils.aperture.CircularAperture(ch1_array, r=3.332666799973338)
ch2_aperture = photutils.aperture.CircularAperture(ch2_array, r=3.332666799973338)


#-----------------load image data-----------------------------
ch1_gemini_data=fits.open('/Users/sarmg/Downloads/Tutorial_VII/data/MOO_1506+5136_irac_ch1.fits')
ch2_gemini_data=fits.open('/Users/sarmg/Downloads/Tutorial_VII/data/MOO_1506+5136_irac_ch2.fits')

gemini_ch1_data =fits.getdata('/Users/sarmg/Downloads/Tutorial_VII/data/MOO_1506+5136_irac_ch1.fits')
gemini_ch2_data =fits.getdata('/Users/sarmg/Downloads/Tutorial_VII/data/MOO_1506+5136_irac_ch2.fits')
#print(gemini_ch2_data)
ch1_photometry = photutils.aperture.aperture_photometry(gemini_ch1_data, ch1_aperture)
ch2_photometry = photutils.aperture.aperture_photometry(gemini_ch2_data, ch2_aperture)
#print(ch1_photometry)


#---------convert image flux to real image-----------------------


def convert_to_mj(image_flux,ZP):
    '''
    convert image flux to microJanskys
    
    INPUT
    ------------
    image_flux:float
        image flux
        
    ZP: float
       zeropoint
    
    
    OUTPUT
    -------------
    real_flux:float
    
    '''
    
    real_flux = image_flux*10**(-0.4*(ZP-23.9))
    return real_flux



zflux = convert_to_mj(phot_zcat['aperture_sum'],32.7)
rflux = convert_to_mj(phot_rcat['aperture_sum'],32.0)

phot_zcat.add_column(Column(zflux, name='real_flux'))
phot_rcat.add_column(Column(rflux, name='real_flux'))




rflux_histogram=plt.axes()
rflux_histogram.set_xlabel('r_flux(uly)',fontsize=17.5)
rflux_histogram.hist(phot_rcat['real_flux'], bins=100, range=(-1,1))
r_hist=np.histogram(phot_rcat['real_flux'], bins=100,range=(-1,1))

rflux_histogram.figure
#plt.savefig('r-flux-histogram.pdf')
#rflux_histogram.close()

zflux_histogram=plt.axes()
zflux_histogram.set_xlabel('z_flux(uly)',fontsize=17.5)
zflux_histogram.hist(phot_zcat['real_flux'], bins=200, range=(-1,1))
z_hist=np.histogram(phot_zcat['real_flux'], bins=200,range=(-1,1))

zflux_histogram.figure
#plt.savefig('z-histogram.pdf')

ch1_flux = convert_to_mj(ch1_photometry['aperture_sum'],21.58)
ch2_flux = convert_to_mj(ch2_photometry['aperture_sum'],21.58)

#---------added to table
ch1_photometry.add_column(Column(ch1_flux, name= 'real_flux'))
ch2_photometry.add_column(Column(ch2_flux, name= 'real_flux'))

#ch1 ch2--------------------
#--------------plot ch1 histogram--------------------

#ch1_histogram = plt.axes()
#ch1_histogram.set_xlabel('flux (uJy)', fontsize=17.5)
#ch1_histogram.hist(ch1_photometry['real_flux'], bins=50, range=(-2,10))
ch1_hist = np.histogram(ch1_photometry['real_flux'], bins=50, range=(-2,10))
#ch1_histogram.figure

#plt.savefig('ch1_new_histogram.pdf')

#del(ch1_histogram)

ch2_hist = np.histogram(ch2_photometry['real_flux'], bins=50, range=(6,20))


#---------------docstring documents----------------------
    
def gaussian(xs,a,mu,sigma):
    '''
    Defines a Gaussian curve
    
    INPUT
    -----
    xs: NumPy array
        The range of values over which the Gaussian is to be calculated
    a: flaot
        The scale factor for the Gaussian
    mu: float
        The mean of the Gaussian
    sigma: float
        The standard deviation of the Gaussian
    
    OUTPUT
    ------
    ys: NumPy array
        The y-values of the Gaussian, corresponding to the input x-values
    '''

    ys = a/np.sqrt(2*np.pi)*np.exp(-(xs-mu)**2/(2*sigma**2)) #Equation for a Gaussian
    #Note that because NumPy can do arithmatic on arrays as though they were single
    #numbers, we only need to put one line in here, and the output will be an array
    #of the same dimension as the input.

    return ys

def makeGauss(np_hist):
    '''
    Takes the input histogram from NumPy and returns a symmetric Gaussian \
    based off the negative side of the distribution.
    
    INPUT
    -----
    np_hist: tuple of arrays
        This should be the direct output of np.histogram, a tuple of two arrays. 
        The first array gives the heights of the bins, the second the bin edges.
    
    OUTPUT
    ------
    symmetric_xs: NumPy array
        The x-values of a symmetric Gaussian, matching the lower half of the input
    
    symmetric_ys: NumPy array
        The y-values of a symmetric Gaussian, matching the lower half of the input
    '''
    xs = []
    for i in range(len(np_hist[1])-1): #Iterate through the bin edges, up to the penultimate one
        xs.append(np.mean(np_hist[1][i:i+2])) #Add the midpoint of the bin to the array
    xs = np.array(xs)

    ys = np_hist[0] #xs and ys are now the midpoints and heights of the bins, respectively

    peak = np.max(ys)
    mean = xs[np.where(ys == peak)] #Define a mean of the new distribution as the peak of the input

    lower_ys = ys[np.where(xs < mean)] #Lower half of the Gaussian
    lower_xs = xs[np.where(xs < mean)]
    
    upper_ys = lower_ys[::-1] #Upper Gaussian
    upper_xs = xs[np.where(xs > mean)]
    upper_xs = upper_xs[:len(upper_ys)]

    symmetric_ys = np.concatenate([lower_ys,peak,upper_ys],axis=None) #Cat all the values into one array
    symmetric_ys = symmetric_ys.astype(np.float)
    symmetric_xs = np.concatenate([lower_xs,mean,upper_xs],axis=None)

    return symmetric_xs,symmetric_ys

def fitGauss(np_hist):
    '''
    Takes the input histogram from NumPy and returns a symmetric Gaussian \
    based off the negative side of the distribution.
    
    INPUT
    -----
    np_hist: tuple of arrays
        This should be the direct output of np.histogram, a tuple of two arrays. 
        The first array gives the heights of the bins, the second the bin edges.
    
    OUTPUT
    ------
    fit: tuple
        The scale, mean, and standard deviation of the best-fit Gaussian
    '''
    symmetric_xs,symmetric_ys = makeGauss(np_hist) #Get x and y for the Gaussian
    
    fit = optimise.curve_fit(gaussian,symmetric_xs,symmetric_ys,p0=(np.max(symmetric_ys),0.0,1.0)) #Fit parameters to the Gaussian

    return fit[0][0],fit[0][1],np.abs(fit[0][2])


ch1_fit = fitGauss(ch1_hist)

#Out[80]: (177280.06304937662, 1.7200000000609474, 0.8032408158246305)

ch2_fit = fitGauss(ch2_hist)
#Out[81]: (433114.27693803015, 0.37840273842423877, 1.0682548983836042)

ch1_xs = np.arange(-2.0,10,0.05)
ch1_a = 177280.06304937662
ch1_mu = 1.7200000000609474
ch1_sigma= 0.8032408158246305
#ch1_ys = ch1_a/np.sqrt(2*np.pi)*np.exp(-(ch1_xs-ch1_mu)**2/(2*ch1_sigma**2))
ch1_ys = gaussian(ch1_xs,ch1_fit[0],ch1_fit[1],ch1_fit[2])

ch1_histogram=plt.axes()
ch1_histogram.hist(ch1_photometry['real_flux'], bins=50, range=(-2,10))
ch1_histogram.set_xlabel('ch1_flux(uly)',fontsize=17.5)
ch1_histogram.plot(ch1_xs,ch1_ys, c='red')

#ch1_histogram.hist(ch1_photometry['real_flux'], bins=50, range=(-2,10))
#ch1_histogram.plot(ch1_xs,ch1_ys, c='red')

#plt.savefig('new_ch1_gaussian.pdf')


#--------------plot ch1 histogram--------------------

ch2_histogram = plt.axes()
ch2_histogram.set_xlabel('ch2_flux (uJy)', fontsize=17.5)
ch2_histogram.hist(ch2_photometry['real_flux'], bins=50, range=(6,20))
#ch2_hist = np.histogram(ch2_photometry['real_flux'], bins=50, range=(6,20))
#plt.savefig('ch2_histogram.pdf')

#del(ch2_histogram)

#--------------------------------------------

ch2_xs = np.arange(-2.0,10,0.30)
ch2_a = 433114.27693803015
ch2_mu = 0.37840273842423877
ch2_sigma = 1.0682548983836042

ch2_ys = gaussian(ch2_xs,ch2_fit[0],ch2_fit[1],ch2_fit[2])
ch1_histogram.set_xlabel('r_flux(uly)',fontsize=17.5)
ch2_histogram=plt.axes()
ch2_histogram.hist(ch2_photometry['real_flux'], bins=50, range=(-2,20))
ch2_histogram.plot(ch2_xs,ch2_ys, c='red')
plt.savefig('ch2_gaussian.pdf')
ch2_xs = np.arange(-2.0,10,0.05)
ch2_ys = gaussian(ch2_xs,ch2_fit[0],ch2_fit[1],ch2_fit[2])



#-------------------taskIII---------------------------

import numpy as np
from astropy.io import ascii
from astropy.table import Table, Column
from astropy.io import fits
import matplotlib.pyplot as plt
import photutils
from scipy import optimize as optimise
import os


#---------------------------------------------------

def create_r_cat(measuring_image,detection_image,config_file,cat_name,ZP):
    cat = f'sex {measuring_image}.fits,{detection_image}.fits -c {config_file}.sex -CATALOG_NAME {cat_name}.cat -MAG_ZEROPOINT {ZP}'
    create_r_cat = os.system(cat)
    return create_r_cat


#--------------create r_band catalog
create_r_cat('/Users/sarmg/Downloads/Tutorial_VII/data/MOO_1506+5136_gmos_r','/Users/sarmg/Downloads/Tutorial_VII/data/MOO_1506+5136_gmos_r','/Users/sarmg/Downloads/Tutorial_VII/data/T3_default','/Users/sarmg/Downloads/Tutorial_VII/data/r_band_cat',32.0)
r_band_cat = ascii.read('/Users/sarmg/Downloads/Tutorial_VII/data/r_band_cat.cat')
print(r_band_cat)

#-------------create z_band catalog ----------------------------------
create_r_cat('/Users/sarmg/Downloads/Tutorial_VII/data/MOO_1506+5136_gmos_r','/Users/sarmg/Downloads/Tutorial_VII/data/MOO_1506+5136_gmos_z','/Users/sarmg/Downloads/Tutorial_VII/data/T3_default','/Users/sarmg/Downloads/Tutorial_VII/data/z_band_cat',32.7)
z_band_cat = ascii.read('/Users/sarmg/Downloads/Tutorial_VII/data/z_band_cat.cat')
print(z_band_cat)

#--------------create ch1 catalog-------------------------------------

create_r_cat('/Users/sarmg/Downloads/Tutorial_VII/data/MOO_1506+5136_irac_ch1','/Users/sarmg/Downloads/Tutorial_VII/data/MOO_1506+5136_irac_ch1','/Users/sarmg/Downloads/Tutorial_VII/data/ch1_SE_VII','/Users/sarmg/Downloads/Tutorial_VII/data/ch1_band_cat',21.58)
ch1_band_cat = ascii.read('/Users/sarmg/Downloads/Tutorial_VII/data/ch1_band_cat.cat')
print(ch1_band_cat)

#-----------rename of r_band_cat------------------------------------
# r_band_cat.info('stats')
r_band_cat.rename_column('NUMBER', 'ID')
r_band_cat.rename_column('MAG_APER', 'mag_r')
r_band_cat.rename_column('ALPHA_J2000', 'RA')
r_band_cat.rename_column('DELTA_J2000', 'DEC')
r_band_cat.rename_column('FLUX_APER', 'Flux_r')
r_band_cat.rename_column('FLAGS', 'flags_r')

#-----------rename of z_band_cat------------------------------------

z_band_cat.rename_column('NUMBER', 'ID')
z_band_cat.rename_column('MAG_APER', 'mag_z')
z_band_cat.rename_column('ALPHA_J2000', 'RA')
z_band_cat.rename_column('DELTA_J2000', 'DEC')
z_band_cat.rename_column('FLUX_APER', 'Flux_z')
z_band_cat.rename_column('FLAGS', 'flags_z')

#-----------rename of ch1_band_cat------------------------------------

ch1_band_cat.rename_column('NUMBER', 'ID')
ch1_band_cat.rename_column('MAG_APER', 'mag_ch1')
ch1_band_cat.rename_column('ALPHA_J2000', 'RA')
ch1_band_cat.rename_column('DELTA_J2000', 'DEC')
ch1_band_cat.rename_column('FLUX_APER', 'Flux_ch1')
ch1_band_cat.rename_column('FLAGS', 'Flags_ch1')
ch1_band_cat.add_column(ch1_band_cat['Flux_ch1']*1.41, name='newFlux_ch1')
ch1_band_cat.remove_column('Flux_ch1')
ch1_band_cat.rename_column('newFlux_ch1', 'Flux_ch1')



#----------------------step-9-----------------------------------
r_sigma= 0.07806646782173232


r_flux_error_array = np.ones(len(r_band_cat))

r_flux_error_array = r_flux_error_array*0.07806646782173232

#r_flux_error_array = r_flux_error_array*0.07806646782173232

#print(r_flux_error_array)
#[0.07806647 0.07806647 0.07806647 ... 0.07806647 0.07806647 0.07806647]


z_sigma= 0.30071045809626423
z_flux_error_array = np.ones(len(z_band_cat))
z_flux_eror_array = z_flux_error_array*0.30071045809626423

#print(z_flux_eror_array)
#[0.30071046 0.30071046 0.30071046 ... 0.30071046 0.30071046 0.30071046]


ch1_sigma= 0.8032408158246305

ch1_flux_error_array = np.ones(len(ch1_band_cat))
ch1_flux_error_array = ch1_flux_error_array*0.8032408158246305

#print(ch1_flux_error_array)
#[0.80324082 0.80324082 0.80324082 ... 0.80324082 0.80324082 0.80324082]

ch1_flux_error_array = ch1_flux_error_array*0.8032408158246305*1.41

#print(ch1_flux_error_array)
#[0.90972609 0.90972609 0.90972609 ... 0.90972609 0.90972609 0.90972609]

#------------column error----------r_band------------------------------

r_flux_error_column = Column(r_flux_error_array, name='FluxErr_r')
r_band_cat.add_column(r_flux_error_column)

#------------column error----------r_band------------------------------

z_flux_error_column = Column(z_flux_error_array, name='FluxErr_z')
z_band_cat.add_column(z_flux_error_column)

#------------column error----------r_band------------------------------

ch1_flux_error_column = Column(ch1_flux_error_array, name='FluxErr_ch1')
ch1_band_cat.add_column(ch1_flux_error_column)


#--------flux--------------------

r_flux = r_band_cat['Flux_r']
mag_error_r = ((1.09)*((r_flux_error_column)/(r_flux)))

r_mag_error_column = Column(mag_error_r, name='magerr_r')
r_band_cat.add_column(r_mag_error_column)

z_flux = z_band_cat['Flux_z']
mag_error_z = ((1.09)*((z_flux_error_column)/(z_flux)))

z_mag_error_column = Column(mag_error_z, name='magerr_z')
z_band_cat.add_column(z_mag_error_column)

ch1_flux = ch1_band_cat['Flux_ch1']
mag_error_ch1 = ((1.09)*((ch1_flux_error_column)/(ch1_flux)))

ch1_mag_error_column = Column(mag_error_ch1, name='magerr_ch1')
ch1_band_cat.add_column(ch1_mag_error_column)



#---------------------task-IV----------------------------------

#------------------step-1--------------------------------------

r_and_z_cat = Table()

r_and_z_cat.add_columns([r_band_cat['ID'],r_band_cat['RA'],r_band_cat['DEC']])

r_and_z_cat.add_columns([r_band_cat['Flux_r'],r_band_cat['mag_r'],z_band_cat['Flux_z'],z_band_cat['mag_z']])
r_and_z_cat.add_columns([r_band_cat['magerr_r'],r_band_cat['FluxErr_r'],z_band_cat['magerr_z'],z_band_cat['FluxErr_z']])
r_and_z_cat.add_columns([r_band_cat['flags_r'],z_band_cat['flags_z']])


ch1_table = Table()
ch1_table.add_columns([ch1_band_cat['ID'],ch1_band_cat['RA'],ch1_band_cat['DEC'],ch1_band_cat['mag_ch1'],ch1_band_cat['Flux_ch1'],ch1_band_cat['magerr_ch1'],ch1_band_cat['FluxErr_ch1'],ch1_band_cat['Flags_ch1']])


#--------------step-7--------------------------------
ch1_table.rename_column ('ID','ID_ch1')
ch1_table.rename_column ('RA','RA_ch1')
ch1_table.rename_column ('DEC','DEC_ch1')

#matched_table = Table()

def distance_function(RA_1,RA_2,DEC_1,DEC_2):
    #calculate the distance
    '''
    INPUT
    ---------------
    RA_1 : array-like
       Right ascention
    RA_2 : aaray-like
       second right ascention
       
    DEC_1 : array-like
       First declination from east to west
    DEC_2 : array_like
        second delination from east to west
        
        
    OUTPUT
    -------------------
    distance : array-like
       distance value would be in arcsec
       
    '''
    
    
    RA_1_arcsec = RA_1*3600
    RA_2_arcsec = RA_2*3600
    DEC_1_arcsec = DEC_1*3600
    DEC_1_rad = DEC_1* np.pi / 180
    DEC_2_arcsec = DEC_2*3600
    distance = np.sqrt((RA_2_arcsec - RA_1_arcsec)**2 * np.cos(DEC_1_rad)**2 + (DEC_2_arcsec - DEC_1_arcsec)**2)
    return distance

for row in ch1_table:
    ra = row['RA_ch1']
    dec = row['DEC_ch1']
    distances = distance_function(ra,r_and_z_cat['RA'].data,dec,r_and_z_cat['DEC'].data)
    match_row = r_and_z_cat[np.where(distances == np.min(distances))]
    try:
        matched_table.add_row(match_row[0])
    except NameError:
        matched_table = Table(match_row[0])
    
matched_table.add_column(ch1_table['ID_ch1'])  
matched_table.add_column(ch1_table['mag_ch1'])  
matched_table.add_column(ch1_table['RA_ch1'])  
matched_table.add_column(ch1_table['DEC_ch1']) 

matched_table.add_column(ch1_table['Flags_ch1'])
matched_table.add_column(ch1_table['Flux_ch1'])
matched_table.add_column(ch1_table['FluxErr_ch1'])
matched_table.add_column(ch1_table['magerr_ch1'])    
    
    
Gemini_ch1 = matched_table[np.where((matched_table['flags_r']<3)|(matched_table['flags_z']<3)|(matched_table['Flags_ch1']<3))]   
    
Gemini_ch1.write('Gemini_ch1.cat', format='ascii.commented_header')   
    








#---------------------step-13---------------------------------------

'''
ZP=32
flux_image=r_aperture_sum


aperture1=(10**(-.4*(32.0-23.9)))*flux_image
print(aperture1)

aperture_histogram=plt.axes()


aperture_histogram.set_xlabel('Aperture (A)', fontsize=17.5)


aperture_histogram.hist(aperture1,bins=50,range=(-0.5,0.5),alpha=0.5)
    
aperture_histogram.figure
'''