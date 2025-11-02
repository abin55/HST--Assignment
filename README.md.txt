Overview:
 	This project performs astronomical image combination and photometric analysis on HST FITS data. It detects stars, measures thier fluxes 
	in two filters as F336W and F555W, as produce an HR Diagram.
 
First install the required dependencies
	numpy
	pandas
	matplotlib
	astropy
	photutils
	scipy

  The FITS images are in data/F336W & F555W

  To run the code give command as python hst_assignment.py

  The output of this code will be catalog.csv and catalog.fits are photometric results, HR_Diagram.png and FWHM_histogram.png.
				
Note:
	This code uses the DAOStarFinder for detection and Gaussion fitting for centroid refinement.
	Magnitudes are relative and detection threshold, FWHM range, aperture size as adjustable settings.


	To open the FITS file use this code;

		from astropy.table import Table   # Read the FITS catalog                                        

		catalog = Table.read("outputs/catalog.fits")

		print(catalog[:])   # Show the first few rows                                                          

		print(catalog.colnames)   # See all column names
