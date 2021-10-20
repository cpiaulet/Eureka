# NIRISS specific rountines go here
#
# Written by: Adina Feinstein
# Last updated by: Adina Feinstein
# Last updated date: October 12, 2021
#
####################################

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage import filters, feature
from scipy.ndimage import gaussian_filter

from .background import fitbg3


__all__ = ['NIRISS_S3']


class NIRIRSS_S3(object):
    """
    Class that handles the S3 data reduction for NIRISS.
    """

    def __init__(self, filename, f277_filename):
        """
        Initializes the S3 data reduction for NIRISS.

        Parameters
        ----------
        filename : str
           Single filename to read. Should be a `.fits` file.
        f277_filename : str
           Single F277 filtered filename to read. Should be
           `.fits` file.

        Attributes
        ----------
        filename : str
        f277_filename : str
        """
        self.filename = filename
        self.f277_filename = f277_filename

        self.read()


    def read(self):
        """
        Reads a single FITS file from JWST's NIRISS instrument.
        This takes in the Stage 2 processed files.
        
        Attributes
        -------
        hdr : astropy.io.header.Header
           Filename main header.
        shdr : astropy.io.header.Header
           Science extension header for the main NIRISS data.
        intend : np.array
        bjdtbd : np.array
           Time array from the header.
        data : np.ndarray
           Data frames.
        err : np.ndarray
           Error frames.
        dq : np.ndarray
           Data quality frames.
        f277 : np.ndarray
           F277 filter data.
        var : np.ndarray
           Poisson variance information.
        v0 : np.ndarray
           Variance noise information.
        meta : np.ndarray
           `ASDF_METADATA` information.
        """
        assert(self.filename, str)
        
        hdu = fits.open(self.filename)
        f277= fits.open(self.f277_filename)
        
        # loads in all the header data
        self.hdr = hdu[0].header
        self.shdr = hdu['SCI'][1].header
        
        self.intend = hdu[0].header['NINTS'] + 0.0
        self.bjdtbd = np.linspace(self.hdr['EXPSTART'], 
                                  self.hdr['EXPEND'], 
                                  self.intend)
        
        # loads all the data into the data object
        self.data = hdu['SCI',1].data + 0.0
        self.err  = hdu['ERR',1].data + 0.0
        self.dq   = hdu['DQ' ,1].data + 0.0
        
        self.f277 = f277[1].data + 0.0
        
        self.var  = hdu['VAR_POISSON',1].data + 0.0
        self.v0   = hdu['VAR_RNOISE' ,1].data + 0.0
        
        self.meta = hdu['ASDF_METADATA',1].data
        
        # removes NaNs from the data & error arrays
        self.data[np.isnan(self.data)==True] = 0
        self.err[ np.isnan(self.err) ==True] = 0
        
        return 


    def image_filtering(self, img, radius=1, gf=4):
        """
        Does some simple image processing to isolate where the
        spectra are located on the detector. This routine is 
        optimized for NIRISS S2 processed data and the F277W filter.
        
        Parameters
        ----------
        img : np.ndarray
           2D image array. 
        radius : np.float, optional
           Default is 1.
        gf : np.float, optional
           The standard deviation by which to Gaussian
           smooth the image. Default is 4.
        
        Returns
        -------
        img_mask : np.ndarray
           A mask for the image that isolates where the spectral 
           orders are.
        """
        mask = filters.rank.maximum(img/np.nanmax(img),
                                    disk(radius=radius))
        mask = np.array(mask, dtype=bool)
        
        # applies the mask to the main frame
        data = img*~mask
        g = gaussian_filter(data, gf)
        g[g>6] = 200
        edges = filters.sobel(g)
        edges[edges>0] = 1
    
        # turns edge array into a boolean array
        edges = (edges-np.nanmax(edges)) * -1
        z = feature.canny(edges)
    
        return z, g


    def f277_mask(self, img):
        """        
        Marks the overlap region in the f277w filter image.                                                       
                
        Returns
        -------
        mask : np.ndarray
           2D mask for the f277w filter.
        mid : np.ndarray
           (x,y) anchors for where the overlap region is located.
        """
        mask, _ = self.image_filtering(img[:150,:500])
        mid = np.zeros((mask.shape[1], 2),dtype=int)
        new_mask = np.zeros(img.shape)
        
        for i in range(mask.shape[1]):
            inds = np.where(mask[:,i]==True)[0]
            if len(inds) > 1:
                new_mask[inds[1]:inds[-2], i] = True
                mid[i] = np.array([i, (inds[1]+inds[-2])/2])

        q = ((mid[:,0]<420) & (mid[:,1]>0) & (mid[:,0] > 0))
        return new_mask, mid[q]


    def create_niriss_mask(self, order_width=14, plot=False):
        """
        This routine takes the output S2 processed images and creates
        a mask for each order. This routine creates a single image from
        all 2D images, applies a Gaussian filter to smooth the image, 
        and a Sobel edge detection method to identify the outlines of
        each order. The orders are then fit with 2nd degree polynomials.
        
        Parameters
        ----------
        order_width : int, optional
           The width around the order that should be extracted.
           Default is 14.
        plot : bool, optional
           An option to plot the data and intermediate steps to 
           retrieve the mask per each order. Default is False.

        Attributes
        -------
        order_mask : np.ndarray
           A mask for the 2D images that marks the first and second
           orders for NIRISS observations. The first order is marked
           with value = 1; the second order is marked with value = 2.
           Overlap regions are marked with value = 3.
        bkg_mask : np.ndarray
           A mask for the 2D images that marks where the background
           is for NIRISS observations. Background regions are given
           value = 1. Regions to ignore are given value = 0.

        """
        def poly_fit(x,y,deg):
            poly = np.polyfit(x,y,deg=deg)
            return np.poly1d(poly)
        
        perc  = np.nanmax(self.data, axis=0)
        fperc = np.nanmax(self.f277, axis=(0,1))

        # creates data img mask
        z,g = self.image_filtering(perc)

        # creates mask for f277w image and anchors
        fmask, fmid = self.f277_mask(fperc)

        # Identify the center of the 1st and 2nd
        # spectral orders
        zmask = np.zeros(z.shape)
        start = 800
        mid1 = np.zeros((z[:,start:].shape[1],2),dtype=int)
        mid2 = np.zeros((z[:,start:].shape[1],2),dtype=int)
        
        for y in np.arange(start,z.shape[1]-1,1,dtype=int):
            inds = np.where(z[:,y]==True)[0]
        
            if len(inds)>=4:
                zmask[inds[0]:inds[1],y] = True
                zmask[inds[2]:inds[-1],y] = True
                
                mid1[y-start] = np.array([y, (inds[0]+inds[1])/2])
                mid2[y-start] = np.array([y, (inds[2]+inds[-1])/2])
                
            if y > 1900:
                zmask[inds[0]:inds[-1],y] = True
                mid1[y-start] = np.array([y, (inds[0]+inds[-1])/2])

        # Clean 1st order of outliers
        mid1 = mid1[np.argsort(mid1[:,0])]
        tempfit = poly_fit(mid1[:,0], mid1[:,1], 3)
        q1 = ((np.abs(tempfit(mid1[:,0])-mid1[:,1]) <2) &
              (mid1[:,0] > start))
        mid1 = mid1[q1]
        
        # Clean 2nd order of outliers
        mid2 = mid2[np.argsort(mid2[:,0])]
        tempfit = poly_fit(mid2[:,0], mid2[:,1], 3)
        q2 = (( np.abs(tempfit(mid2[:,0])-mid2[:,1]) <2) &
              (mid2[:,0] > start) )
        mid2 = mid2[q2]

        # Append overlap region to non-overlap regions
        x1, y1 = np.append(fmid[:,0], mid1[:,0]), np.append(fmid[:,1], mid1[:,1])
        x2, y2 = np.append(fmid[:,0], mid2[:,0]), np.append(fmid[:,1], mid2[:,1])

        fit1 = poly_fit(x1,y1,4)
        fit2 = poly_fit(x2,y2,4)

        img_mask = np.zeros(perc.shape)
        bkg_mask = np.ones(perc.shape)

        bkg_width = 30
        for i in range(perc.shape[1]):
            img_mask[int(fit1(i)-order_width):
                         int(fit1(i)+order_width),i] += 1
            bkg_mask[int(fit1(i)-bkg_width):
                         int(fit1(i)+bkg_width),i] = np.nan
            
            if i < x2[-1]:
                img_mask[int(fit2(i)-order_width):
                             int(fit2(i)+order_width),i] += 2
                bkg_mask[int(fit2(i)-bkg_width):
                             int(fit2(i)+bkg_width),i] = np.nan

                
        # plots some of the intermediate and final steps
        if plot:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, 
                                                     figsize=(14,10))
            ax1.imshow(g)
            ax1.set_title('Gaussian smoothed data')
            ax2.imshow(z)
            ax2.set_title('Canny edge detector')
            ax3.imshow(img_mask, vmin=0, vmax=3)
            ax3.set_title('Final mask')
            ax4.imshow(bkg_mask, vmin=0, vmax=1)
            ax4.set_title('Background mask')
            plt.show()

        self.order_mask = img_mask
        self.bkg_mask  = bkg_mask


    def fit_bg(self, deg=2, thresh=5):
        """
        Subtracts background from non-spectral regions.
        
        Parameters
        ----------
        deg : int, optional
           What degree polynomial should be fit to the 
           background. Default is 2. Deg <= 0 will return
           a median frame.
        thresh : float, optional
           The sigma threshold to identify outliers by.
           Default is 5.

        Attributes
        ----------
        bkg : np.ndarray
           Background model.
        """
        bg = fitbg3(self.data, self.order_mask, 
                    self.bkg_mask,
                    deg=deg, threshold=thresh)
        
        self.bkg = bg
