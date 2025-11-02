import os
import sys
import warnings
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from astropy.visualization import simple_norm

from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.psf import IntegratedGaussianPRF

from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

# Utility: 2D Gaussian fit

def two_d_gaussian(coords, amp, x0, y0, sigma_x, sigma_y, theta, offset):
    """
    2D elliptical Gaussian used by curve_fit.
    coords: (x.ravel(), y.ravel())
    """
    x, y = coords
    x0 = float(x0)
    y0 = float(y0)
    a = (np.cos(theta)**2) / (2*sigma_x**2) + (np.sin(theta)**2) / (2*sigma_y**2)
    b = -(np.sin(2*theta)) / (4 * sigma_x**2) + (np.sin(2*theta)) / (4 * sigma_y**2)
    c = (np.sin(theta)**2) / (2*sigma_x**2) + (np.cos(theta)**2) / (2*sigma_y**2)
    return offset + amp * np.exp(-(a * (x - x0)**2 + 2*b*(x - x0)*(y - y0) + c*(y - y0)**2))


def fit_2d_gaussian(cutout: np.ndarray):
    """
    Fit a 2D elliptical Gaussian to a small cutout image.
    Returns dictionary with fit params and success flag.
    """
    h, w = cutout.shape
    y, x = np.mgrid[0:h, 0:w]
    x_flat = x.ravel()
    y_flat = y.ravel()
    z_flat = cutout.ravel()

    # initial guesses
    amp0 = np.nanmax(cutout) - np.nanmedian(cutout)
    if amp0 <= 0 or not np.isfinite(amp0):
        return {'success': False}
    x0_0 = w / 2.0
    y0_0 = h / 2.0
    sigma0 = max(1.0, min(w, h) / 4.0)
    theta0 = 0.0
    offset0 = np.nanmedian(cutout)

    p0 = [amp0, x0_0, y0_0, sigma0, sigma0, theta0, offset0]
    bounds = (
        [0, 0, 0, 0.3, 0.3, -np.pi/2, -np.inf],
        [np.inf, w, h, max(w,h), max(w,h), np.pi/2, np.inf]
    )

    # mask NaNs
    finite_mask = np.isfinite(z_flat)
    try:
        popt, pcov = curve_fit(
            two_d_gaussian,
            (x_flat[finite_mask], y_flat[finite_mask]),
            z_flat[finite_mask],
            p0=p0,
            bounds=bounds,
            maxfev=10000
        )
    except Exception:
        return {'success': False}

    amp, x0, y0, sx, sy, theta, offset = popt
    # convert gaussian sigma to FWHM: FWHM = 2.355 * sigma (for 1D gaussian)
    fwhm_x = 2.355 * sx
    fwhm_y = 2.355 * sy

    return {
        'success': True,
        'amp': float(amp),
        'x0': float(x0),
        'y0': float(y0),
        'sigma_x': float(sx),
        'sigma_y': float(sy),
        'theta': float(theta),
        'offset': float(offset),
        'fwhm_x': float(fwhm_x),
        'fwhm_y': float(fwhm_y),
    }


# Combine exposures

def combine_images(folder: str, out_name="combined.fits") -> Tuple[np.ndarray, dict]:
    """
    Median combine all .fits exposures in folder (excluding combined.fits).
    Returns (median_image, header_info)
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    fits_files = [f for f in os.listdir(folder) if f.endswith('.fits') and f != out_name]
    if len(fits_files) == 0:
        raise FileNotFoundError(f"No FITS exposures found in {folder}")

    data_stack = []
    header_example = None

    for fname in sorted(fits_files):
        path = os.path.join(folder, fname)
        try:
            with fits.open(path) as hdul:
                # Attempt to take image data intelligently
                data = None
                if len(hdul) > 1 and hdul[1].data is not None:
                    data = hdul[1].data
                    header_example = hdul[1].header
                else:
                    data = hdul[0].data
                    header_example = hdul[0].header

                if data is None:
                    warnings.warn(f"No image data in {fname} (skipping).")
                    continue

                # convert to float and mask non-finite
                arr = np.array(data, dtype=float)
                # Replace NaNs with median of finite values in that image to avoid propagation
                if not np.isfinite(arr).all():
                    med = np.nanmedian(arr)
                    arr = np.nan_to_num(arr, nan=med, posinf=med, neginf=med)
                if arr.ndim != 2:
                    warnings.warn(f"Data in {fname} not 2D (skipping).")
                    continue

                data_stack.append(arr)
        except Exception as e:
            warnings.warn(f"Error reading {fname}: {e}. Skipping.")

    if len(data_stack) == 0:
        raise RuntimeError(f"No valid images found in {folder} after reading files.")

    stack = np.array(data_stack)
    median_image = np.median(stack, axis=0)

    # Save combined
    out_path = os.path.join(folder, out_name)
    hdu = fits.PrimaryHDU(median_image)
    if header_example is not None:
        # preserve some header keys if present
        for key in ('INSTRUME', 'FILTER', 'EXPTIME'):
            if key in header_example:
                hdu.header[key] = header_example.get(key)
    hdu.writeto(out_path, overwrite=True)
    print(f"Saved combined image to {out_path}")
    return median_image, dict(header_example or {})



# Initial detection

def detect_sources(image: np.ndarray, fwhm=3.0, threshold_sigma=5.0):
    """
    Detect candidate sources using DAOStarFinder.
    Returns astropy table-like object (or None).
    """
    # compute background statistics
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    if not np.isfinite(std) or std <= 0:
        raise RuntimeError("Image standard deviation invalid; cannot detect sources.")

    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std)
    sources = daofind(image - median)
    return sources  # can be None



# Candidate refinement & checks

def refine_and_filter_sources(image: np.ndarray, sources, box_size=15,
                              ellipticity_tol=0.5, fwhm_range=(1.0, 8.0),
                              min_flux=5.0):
    """
    For each source, extract a small cutout, fit a 2D Gaussian to refine centroid and measure shape.
    Apply candidacy checks:
      - fit success
      - flux > min_flux
      - ellipticity <= ellipticity_tol (f = 1 - min(sx,sy)/max(sx,sy))
      - FWHM within fwhm_range (both axes)
    Returns list of dicts with refined params (x_global, y_global, fwhm_x, fwhm_y, amp, flag, etc.)
    """
    if sources is None:
        return []

    image_h, image_w = image.shape
    refined = []

    for row in sources:
        try:
            x0 = float(row['xcentroid'])
            y0 = float(row['ycentroid'])
        except Exception:
            continue

        # cutout bounds
        half = box_size // 2
        x_min = int(max(0, np.round(x0) - half))
        x_max = int(min(image_w, np.round(x0) + half + 1))
        y_min = int(max(0, np.round(y0) - half))
        y_max = int(min(image_h, np.round(y0) + half + 1))

        cutout = image[y_min:y_max, x_min:x_max].astype(float)
        if cutout.size == 0:
            continue

        # subtract median background in cutout to help fit
        cut_med = np.nanmedian(cutout)
        cutout_sub = cutout - cut_med

        fit = fit_2d_gaussian(cutout_sub)
        if not fit.get('success', False):
            flag = 'fit_fail'
            refined.append({
                'x': x0, 'y': y0, 'flag': flag, 'fit': None
            })
            continue

        # convert local centroid to global coords
        x_local = fit['x0']
        y_local = fit['y0']
        x_global = x_min + x_local
        y_global = y_min + y_local

        amp = fit['amp']
        sx = fit['sigma_x']
        sy = fit['sigma_y']
        fwhm_x = fit['fwhm_x']
        fwhm_y = fit['fwhm_y']

        # flux proxy: sum in small circle around refined centroid
        rtest = max(1, int(round(max(sx, sy) * 1.5)))
        ygrid, xgrid = np.mgrid[y_min:y_max, x_min:x_max]
        rmap = np.hypot((xgrid - x_global), (ygrid - y_global))
        flux_proxy = np.sum(cutout[rmap <= rtest])

        # ellipticity f = 1 - min/max
        f_ell = 1.0 - (min(sx, sy) / max(sx, sy)) if max(sx, sy) > 0 else 1.0

        # candidacy checks
        pass_flux = (flux_proxy > min_flux)
        pass_ell = (f_ell <= ellipticity_tol)
        pass_fwhm = (fwhm_range[0] <= fwhm_x <= fwhm_range[1]) and (fwhm_range[0] <= fwhm_y <= fwhm_range[1])

        flags = []
        if not pass_flux:
            flags.append('low_flux')
        if not pass_ell:
            flags.append('elliptical')
        if not pass_fwhm:
            flags.append('bad_fwhm')

        if len(flags) == 0:
            flag = 'ok'
        else:
            flag = ';'.join(flags)

        refined.append({
            'x': x_global, 'y': y_global, 'flag': flag,
            'fit': fit, 'flux_proxy': float(flux_proxy), 'ellipticity': float(f_ell)
        })

    return refined



# Photometry with local annulus

def aperture_photometry_local_bkg(image: np.ndarray, positions: np.ndarray,
                                  r_ap=4.0, r_in=8.0, r_out=12.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute aperture photometry and subtract local background using an annulus.
    positions: N x 2 array of (x, y)
    Returns (net_fluxes, mags) arrays (np.nan where invalid).
    """
    if len(positions) == 0:
        return np.array([]), np.array([])

    apertures = CircularAperture(positions, r=r_ap)
    annuli = CircularAnnulus(positions, r_in=r_in, r_out=r_out)
    ap_tbl = aperture_photometry(image, apertures)
    an_tbl = aperture_photometry(image, annuli)

    ap_sum = np.array(ap_tbl['aperture_sum'], dtype=float)
    an_sum = np.array(an_tbl['aperture_sum'], dtype=float)
    an_area = annuli.area
    ap_area = apertures.area

    # background per pixel and total background in aperture
    bkg_per_pix = an_sum / an_area
    bkg_total = bkg_per_pix * ap_area
    net_flux = ap_sum - bkg_total

    # convert to magnitude; no zero point applied (relative mags)
    mags = np.full_like(net_flux, np.nan, dtype=float)
    valid = net_flux > 0
    mags[valid] = -2.5 * np.log10(net_flux[valid])

    return net_flux, mags



# Matching catalogs between filters

def match_catalogs(coords_ref: np.ndarray, coords_other: np.ndarray, max_sep=2.0):
    """
    Match coords_ref (Nx2) to coords_other (M x 2) using KDTree.
    Returns indices in other for each ref (or -1 if no match) and distances.
    """
    tree = cKDTree(coords_other)
    dists, idxs = tree.query(coords_ref, k=1)
    idxs_out = np.where(dists <= max_sep, idxs, -1)
    dists_out = np.where(dists <= max_sep, dists, np.inf)
    return idxs_out, dists_out



# Save catalog helpers

def save_catalog_table(df: pd.DataFrame, out_csv='catalog.csv', out_fits='catalog.fits'):
    df.to_csv(out_csv, index=False)
    print(f"Saved CSV catalog: {out_csv}")

    # also save as FITS table
    t = Table.from_pandas(df)
    t.write(out_fits, overwrite=True)
    print(f"Saved FITS catalog: {out_fits}")


# Plotting
def plot_hr_diagram(color_index: np.ndarray, mag: np.ndarray, outname='HR_Diagram.png'):
    # filter finite
    mask = np.isfinite(color_index) & np.isfinite(mag)
    if np.sum(mask) == 0:
        print("No valid points for HR diagram.")
        return

    plt.figure(figsize=(7, 6))
    plt.scatter(color_index[mask], mag[mask], s=10)
    plt.gca().invert_yaxis()
    plt.xlabel('Color (F336W - F555W)')
    plt.ylabel('F555W Magnitude')
    plt.title('Hertzsprung–Russell Diagram')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outname)
    plt.close()
    print(f"Saved HR diagram: {outname}")


def plot_fwhm_histogram(fwhm_vals: np.ndarray, outname='FWHM_histogram.png'):
    fwhm_vals = np.array(fwhm_vals)
    finite = fwhm_vals[np.isfinite(fwhm_vals)]
    if finite.size == 0:
        print("No FWHM values to plot.")
        return
    plt.figure(figsize=(6,4))
    plt.hist(finite, bins=30)
    plt.xlabel('FWHM (pixels)')
    plt.ylabel('Number of sources')
    plt.title('FWHM distribution')
    plt.tight_layout()
    plt.savefig(outname)
    plt.close()
    print(f"Saved FWHM histogram: {outname}")


# Main pipeline
def run_pipeline(data_dir='data', out_dir='outputs',
                 r_ap=4.0, ann_r_in=8.0, ann_r_out=12.0,
                 detect_fwhm=3.0, detect_sigma=5.0,
                 max_match_sep=2.0):
    """
    Execute the full pipeline. Saves outputs in out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)

    f336_dir = os.path.join(data_dir, 'F336W')
    f555_dir = os.path.join(data_dir, 'F555W')

    # 1. Combine exposures
    try:
        f336_img, hdr336 = combine_images(f336_dir)
        f555_img, hdr555 = combine_images(f555_dir)
    except Exception as e:
        print(f"Error combining images: {e}")
        sys.exit(1)

    # quick check shapes
    if f336_img.shape != f555_img.shape:
        print(f"Combined images shapes mismatch: {f336_img.shape} vs {f555_img.shape}")
        # proceed if user wants but better to exit
        print("Please align images or ensure same chip; exiting.")
        sys.exit(1)

    # 2. Detect sources (use F555W as detection image; typical)
    sources = detect_sources(f555_img, fwhm=detect_fwhm, threshold_sigma=detect_sigma)
    
    if sources is None or len(sources) == 0:
        print("No sources detected. Exiting.")
        sys.exit(1)
    print(f"DAOStarFinder found {len(sources)} candidates.")

    sources = sources[:500]
    print(f"Processing only first {len(sources)} sources for speed.")
    print("Loading......")

    # 3. Refine and filter candidates
    refined = refine_and_filter_sources(f555_img, sources,
                                       box_size=15, ellipticity_tol=0.5,
                                       fwhm_range=(1.0, 8.0), min_flux=5.0)
    # keep only 'ok' candidates as primary list, but we will also keep others with flag noted
    positions = []
    ids = []
    flags = []
    fits_list = []

    for i, r in enumerate(refined):
        ids.append(i+1)
        flags.append(r.get('flag', 'unknown'))
        fits_list.append(r.get('fit'))
        positions.append((r['x'], r['y']))

    positions = np.array(positions)  # N x 2

    # 4. Photometry on both filters using same positions (we already refined on F555)
    flux_f555, mag_f555 = aperture_photometry_local_bkg(f555_img, positions, r_ap=r_ap, r_in=ann_r_in, r_out=ann_r_out)
    flux_f336, mag_f336 = aperture_photometry_local_bkg(f336_img, positions, r_ap=r_ap, r_in=ann_r_in, r_out=ann_r_out)

    # 5. If you'd rather detect on both and match, you can; here we used same positions for both bands.
    # Build catalog DataFrame
    df = pd.DataFrame({
        'id': ids,
        'x_center': positions[:,0],
        'y_center': positions[:,1],
        'flag': flags,
        'flux_F555W': flux_f555,
        'mag_F555W': mag_f555,
        'flux_F336W': flux_f336,
        'mag_F336W': mag_f336,
    })

    # attach fit params where available
    fwhm_x_list = []
    fwhm_y_list = []
    sigma_x_list = []
    sigma_y_list = []
    amp_list = []
    ellipticity_list = []
    for r in refined:
        fit = r.get('fit')
        if fit and fit.get('success', False):
            fwhm_x_list.append(fit['fwhm_x'])
            fwhm_y_list.append(fit['fwhm_y'])
            sigma_x_list.append(fit['sigma_x'])
            sigma_y_list.append(fit['sigma_y'])
            amp_list.append(fit['amp'])
        else:
            fwhm_x_list.append(np.nan)
            fwhm_y_list.append(np.nan)
            sigma_x_list.append(np.nan)
            sigma_y_list.append(np.nan)
            amp_list.append(np.nan)
        ellipticity_list.append(r.get('ellipticity', np.nan))

    df['fwhm_x'] = fwhm_x_list
    df['fwhm_y'] = fwhm_y_list
    df['sigma_x'] = sigma_x_list
    df['sigma_y'] = sigma_y_list
    df['amp'] = amp_list
    df['ellipticity'] = ellipticity_list
    df['aperture_radius'] = r_ap

    # Save outputs
    csv_out = os.path.join(out_dir, 'catalog.csv')
    fits_out = os.path.join(out_dir, 'catalog.fits')
    save_catalog_table(df, out_csv=csv_out, out_fits=fits_out)

    # HR diagram: color = F336W - F555W, mag = F555W
    color = df['mag_F336W'].values - df['mag_F555W'].values
    plot_hr_diagram(color, df['mag_F555W'].values, outname=os.path.join(out_dir, 'HR_Diagram.png'))

    # FWHM histogram
    all_fwhm = np.hstack([df['fwhm_x'].values, df['fwhm_y'].values])
    plot_fwhm_histogram(all_fwhm, outname=os.path.join(out_dir, 'FWHM_histogram.png'))

    print(f"Pipeline finished. Outputs in {out_dir}")


if __name__=='__main__':
    run_pipeline()

        
        
    
