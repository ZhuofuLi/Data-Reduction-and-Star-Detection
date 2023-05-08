import os
import numpy as np
import astropy.io.fits as fits
import astroalign as aa
from astropy.stats import sigma_clipped_stats
from astropy.visualization import astropy_mpl_style, ImageNormalize
from auto_stretch.stretch import Stretch
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
from astroquery.astrometry_net import AstrometryNet
from photutils import DAOStarFinder
from photutils.aperture import CircularAperture
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.simbad import Simbad

###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################

def align_frames(directory):
    # Get a list of all FITS files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.fit')]

    # Load the first frame as the reference image
    reference_file = os.path.join(directory, files[0])
    reference_image = fits.getdata(reference_file)

    # Align all subsequent frames to the reference image
    aligned_frames = [reference_image]
    for file in files[1:]:
        # Load the image data
        image_file = os.path.join(directory, file)
        image_data = fits.getdata(image_file)

        # Align the image to the reference image using astroalign
        aligned_image, _ = aa.register(image_data, reference_image)

        # Append the aligned image to the list of aligned frames
        aligned_frames.append(aligned_image)

    # Return the list of aligned frames
    return aligned_frames

###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################

def sigma_clip_and_stack_images(fits_data, threshold=3, num_iterations=5):
    """
    Perform sigma clipping on all FITS images and stack the resulting clipped images.
    
    Args:
        fits_data (ndarray): data of FITS images.
        threshold (float): Number of standard deviations from the mean beyond which a pixel will be considered an outlier.
        num_iterations (int): Maximum number of iterations to perform.
        output_dir (str): Optional output directory for saving sigma-clipped images and stacked image.
    Returns:
        final_image (numpy.ndarray): The final stacked image.
    """
    
    # Initialize a list to store the sigma-clipped images
    clipped_images = []

    # Iterate over all FITS files in the directory
    for data in fits_data:
        mean, median, std = sigma_clipped_stats(data, sigma=threshold, maxiters=num_iterations)
        
        # Remove outliers from the image
        data[data > mean + threshold*std] = median
        data[data < mean - threshold*std] = median
        
        # Append the clipped image to the list
        clipped_images.append(data)

    # Stack the clipped images into a final image
    final_image = np.median(clipped_images, axis=0)
    
    # Return the final stacked image
    return final_image


###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################

def calibrate_images(light_dir, bias_dir, dark_dir, flat_dir=None, output_dir=None):
    # Load the bias images
    bias_files = os.listdir(bias_dir)
    bias_images = [fits.getdata(os.path.join(bias_dir, f)) for f in bias_files]

    # Load the dark images
    dark_files = os.listdir(dark_dir)
    dark_images = [fits.getdata(os.path.join(dark_dir, f)) for f in dark_files]

    # Load the flat images if flat_dir is provided
    if flat_dir is not None:
        flat_files = os.listdir(flat_dir)
        flat_images = [fits.getdata(os.path.join(flat_dir, f)) for f in flat_files]
    else:
        flat_images = [np.ones_like(dark_images[0])]

    # Align the light images and median stack them
    aligned = align_frames(light_dir)
    median_light = np.median(aligned, axis=0)
    #median_light = sigma_clip_and_stack_images(aligned)

    # Calibrate the median light image using median bias, dark, and flat images
    median_bias = np.median(bias_images, axis=0)
    median_dark = np.median(dark_images, axis=0)
    median_flat = np.median(flat_images, axis=0)

    if flat_dir is not None:
        calibrated_image = (median_light - median_bias - median_dark) / median_flat
    #calibrated_image, footprint = aa.register(calibrated_image, aligned[0], max_control_points=3)
    else:
        calibrated_image = (median_light - median_bias - median_dark)

    # Save the calibrated image to the specified output directory or the current working directory
    if output_dir is None:
        output_dir = os.getcwd()
    output_file = os.path.join(output_dir, 'calibrated.fits')
    fits.writeto(output_file, calibrated_image, overwrite=True)

    # Display the first original light frame, median dark, median bias, and median flat
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].imshow(Stretch().stretch(aligned[0]), cmap='gray', origin='lower')
    axs[0, 0].set_title('First Light Frame')
    axs[0, 0].set_xlabel('x position [pixels]')
    axs[0, 0].set_ylabel('y position [pixels]')

    axs[0, 1].imshow(Stretch().stretch(median_dark), cmap='gray', origin='lower')
    axs[0, 1].set_title('Median Dark Frame')
    axs[0, 1].set_xlabel('x position [pixels]')
    axs[0, 1].set_ylabel('y position [pixels]')

    axs[1, 0].imshow(Stretch().stretch(median_bias), cmap='gray', origin='lower')
    axs[1, 0].set_title('Median Bias Frame')
    axs[1, 0].set_xlabel('x position [pixels]')
    axs[1, 0].set_ylabel('y position [pixels]')

    if flat_dir is not None:
        axs[1, 1].imshow(Stretch().stretch(median_flat), cmap='gray', origin='lower')
        axs[1, 1].set_title('Median Flat Frame')
    else:
        axs[1, 1].text(0.5, 0.5, 'No flats found', horizontalalignment='center',
                        verticalalignment='center', transform=axs[1, 1].transAxes)
        axs[1, 1].set_title('Median Flat Frame')
    axs[1, 1].set_xlabel('x position [pixels]')
    axs[1, 1].set_ylabel('y position [pixels]')
    
    plt.show()

    # Display the calibrated fits
    '''
    stretched_image = Stretch().stretch(calibrated_image)
    plt.figure()
    plt.title("Calibrated Frame")
    plt.imshow(stretched_image, cmap='gray')
    plt.xlabel("x Position [pixels]")
    plt.ylabel("y Position [pixels]")
    '''

    # Return the path of the calibrated image
    return output_file

###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################

def find_faintest_star(astrometry_api_key, fit_image_path, detection_fwhm = 4.0, detection_sigma_threshold = 5, search_radius = '0d5m0s'):

    # Load the FITS file
    hdul = fits.open(fit_image_path)
    data = hdul[0].data

    # Calculate the image statistics
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)

    # Detect the stars using DAOStarFinder
    daofind = DAOStarFinder(fwhm=detection_fwhm, threshold=detection_sigma_threshold*std)
    sources = daofind(data - median)

    # Use Astrometry.net to obtain a WCS solution
    ast = AstrometryNet()
    ast.api_key = astrometry_api_key
    wcs_header = ast.solve_from_image(fit_image_path, force_image_upload=True)

    # Convert pixel coordinates to celestial coordinates
    wcs = WCS(wcs_header)
    ra_arr, dec_arr = wcs.all_pix2world(sources['xcentroid'], sources['ycentroid'], 0)

    # Set up the Simbad query
    simbad_query = Simbad()
    simbad_query.add_votable_fields('flux(V)')

    # Create an empty list to store the query results
    results = []

    # Loop over each RA and Dec value and query SIMBAD for each object
    for ra, dec in zip(ra_arr, dec_arr):
        # Create a SkyCoord object for the query coordinates
        query_coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
        
        # Query SIMBAD for objects within a radius
        result_table = simbad_query.query_region(query_coord, radius=search_radius)
        
        # Check if the query returned any results
        if result_table is not None:
            # Calculate the separation between the query coordinates and each object in the result table
            result_coord = SkyCoord(ra=result_table['RA'], dec=result_table['DEC'], unit=(u.hourangle, u.deg))
            sep = result_coord.separation(query_coord)
            
            # Add the separation as a new column to the result table
            result_table['Separation'] = sep
            
            # Sort the result table by distance from the requested position
            result_table.sort('Separation')
            
            # Find the first object with magnitude information
            mag_idx = 0
            while mag_idx < len(result_table) and not result_table['FLUX_V'][mag_idx]:
                mag_idx += 1
            
            # Extract the object's magnitude from the query result if it exists
            if mag_idx < len(result_table):
                mag = result_table['FLUX_V'][mag_idx]
                
                # Append the closest result and magnitude to the results list
                closest_result = result_table[mag_idx]
                results.append((closest_result, mag))
            else:
                # Append None to the results list if no object with magnitude information was found
                results.append(None)
        else:
            # Append None to the results list if no object was found
            results.append(None)

    # Count the number of non-None results
    num_results = sum(1 for result in results if result is not None)

    # Print the number of non-None results
    #print(f"Number of results: {len(results)}")

    # Print the number of non-None results
    #print(f"Number of non-None results: {num_results}")

    max_mag_index = None
    max_mag = None

    for i, result in enumerate(results):
        if result is not None:
            mag = result[1]
            if max_mag is None or mag > max_mag:
                max_mag = mag
                max_mag_index = i

    # Plot the original image and the image with the detected stars
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    axs[0].imshow(Stretch().stretch(data), cmap='gray', origin='lower')
    axs[0].set_title('Calibrated Stacked Frame')
    axs[0].set_xlabel('x position [pixels]')
    axs[0].set_ylabel('y position [pixels]')

    axs[1].imshow(Stretch().stretch(data), cmap='gray', origin='lower')
    axs[1].scatter(sources['xcentroid'], sources['ycentroid'], facecolors='none', edgecolors='b', s=20)
    axs[1].scatter(sources['xcentroid'][max_mag_index], sources['ycentroid'][max_mag_index], color='red', s=20)
    axs[1].set_title('Calibrated Frame with Detected Stars')
    axs[1].set_xlabel('x position [pixels]')
    axs[1].set_ylabel('y position [pixels]')

    axs[2].imshow(Stretch().stretch(data), cmap='gray', origin='lower')
    axs[2].scatter(sources['xcentroid'][max_mag_index], sources['ycentroid'][max_mag_index], facecolors='none', edgecolors='r', s = 300)
    axs[2].set_xlim(sources['xcentroid'][max_mag_index]-100, sources['xcentroid'][max_mag_index]+100)
    axs[2].set_ylim(sources['ycentroid'][max_mag_index]-100, sources['ycentroid'][max_mag_index]+100)
    axs[2].set_title('Zoomed-In View of the Faintest Star')
    axs[2].set_xlabel('x position [pixels]')
    axs[2].set_ylabel('y position [pixels]')

    # Show the plots
    plt.show()

    # Return the SIMBAD info of the faintest star
    return results[max_mag_index]