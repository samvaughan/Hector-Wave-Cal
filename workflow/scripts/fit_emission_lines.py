from matplotlib import pyplot as plt
import numpy as np
from astropy.io import fits
import pandas as pd
import scipy.optimize as opt
from astropy.wcs import WCS
from tqdm import tqdm
from pathlib import Path
import sqlite3


def gaussian(x, offset, amplitude, mean, sigma):
    return offset + amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma**2))


def make_xvalues_for_cut_spectrum(spectrum_around_line):
    xvalues = np.arange(len(spectrum_around_line))

    return xvalues


def fit_gaussian(spectrum_around_line):
    xvalues = make_xvalues_for_cut_spectrum(spectrum_around_line)
    guess_offset = 0.0
    guess_amplitude = np.max(spectrum_around_line)
    guess_mean = np.mean(xvalues)
    guess_sigma = 2

    popt, pcov = opt.curve_fit(
        gaussian,
        xvalues,
        spectrum_around_line,
        p0=[guess_offset, guess_amplitude, guess_mean, guess_sigma],
    )

    return popt, pcov


def find_line_centre(spectrum_around_line):
    popt, pcov = fit_gaussian(spectrum_around_line)

    xmean = popt[2]
    xmean_error = pcov[2, 2]

    return xmean, xmean_error


def add_fit(ax, xvalues, popt, **kwargs):
    fine_xvalues = np.linspace(xvalues.min(), xvalues.max(), 100)
    ax.plot(fine_xvalues, gaussian(fine_xvalues, *popt), **kwargs)
    return ax


def QC_plot(hdu, fibre_number, selected_line_list):
    spectrum = hdu[0].data[fibre_number, :]

    fig, ax = plt.subplots()
    ax.plot(spectrum, c="k")

    for j, (index, row) in enumerate(selected_line_list.iterrows()):
        line_centre = row.wave
        mask = (wave > line_centre - box_width) & (wave < line_centre + box_width)
        spectrum_around_line = spectrum[mask]
        starting_pixel = np.where(mask)[0][0]
        # Fit the Gaussian
        popt, pcov = fit_gaussian(spectrum_around_line)
        xvals = make_xvalues_for_cut_spectrum(spectrum_around_line) + starting_pixel

        # Adjust the mean to the starting pixel
        popt[2] += starting_pixel
        ax = add_fit(ax, xvals, popt, c="r")

    return fig, ax


if __name__ == "__main__":
    smk = snakemake  # noqa

    # Input filenames
    con = sqlite3.connect(smk.input.database)
    filename = Path(smk.input.arc_frame)
    line_list = smk.input.line_list

    # output filenames for plots and flags
    flag_file = Path(smk.output.flag)
    filename_stem = filename.stem
    hdu = fits.open(filename)

    # Some constants
    # This is the width of spectrum we cut either side of a given line
    box_width = 5.0
    # This is the minimum average flux in a fibre
    # Any fibres below this are ignored
    min_mean_flux = 5
    plot = False

    # Quick sanity check
    assert (
        hdu[0].header["NDFCLASS"] == "MFARC"
    ), f"This code is only for running on arc frames! The NDFCLASS for this file is {hdu[0].header['NDFCLASS']}"

    # Get the reduced arc wavelength solution along a central fibre
    wcs = WCS(hdu[0].header)
    wave, _ = wcs.pixel_to_world_values(np.arange(hdu[0].header["NAXIS1"]), 425)

    # Load our line list
    selected_line_list = pd.read_csv(line_list)

    # Some useful constants
    N_fibres = hdu[0].data.shape[0]
    N_pixels = hdu[0].data.shape[1]
    N_lines = len(selected_line_list)

    # Make our "y" values, the fibre numbers
    fibre_numbers = np.arange(N_fibres).repeat(N_lines).reshape(N_fibres, N_lines)
    # Make our long list of wavelengths
    # Note the transpose at the end is very important!
    wavelengths = np.tile(selected_line_list, N_fibres).T

    # Empty arrays we'll fill
    xmeans = np.zeros((N_fibres, N_lines))
    xmean_errors = np.zeros((N_fibres, N_lines))
    starting_pixels = np.zeros((N_fibres, N_lines))

    # Find the mean flux in each fibre
    # This helps us decide whether it's worth fitting it or not
    flux_means = np.nanmean(hdu[0].data, axis=1)

    for i in tqdm(range(N_fibres)):
        spectrum = hdu[0].data[i, :]

        if flux_means[i] > min_mean_flux:
            for j, (index, row) in enumerate(selected_line_list.iterrows()):
                line_centre = row.wave

                mask = (wave > line_centre - box_width) & (
                    wave < line_centre + box_width
                )
                spectrum_around_line = spectrum[mask]

                starting_pixel = np.where(mask)[0][0]
                starting_pixels[i, j] = starting_pixel
                # Fit the Gaussian
                try:
                    popt, pcov = fit_gaussian(spectrum_around_line)
                    xmean = popt[2]
                    xmean_error = pcov[2, 2]
                except RuntimeError:
                    xmean = np.nan
                    xmean_error = np.nan
                except ValueError:
                    xmean = np.nan
                    xmean_error = np.nan

                xmeans[i, j] = xmean + starting_pixel
                xmean_errors[i, j] = xmean_error

        else:
            xmeans[i, ...] = np.nan
            xmean_errors[i, ...] = np.nan

    # Save our results into a nice dataframe
    results_df = pd.DataFrame(
        dict(
            x=xmeans.ravel(),
            x_error=xmean_errors.ravel(),
            y=fibre_numbers.ravel(),
            wavelength=wavelengths.ravel(),
            filename=np.full(N_fibres * N_lines, filename_stem),
        )
    )

    # Now append it to a database
    # Make sure we overwrite if this arc frame has already been measured
    cur = con.cursor()
    cur.executescript(
        f"""
        DELETE FROM arc_data
        WHERE arc_data.filename='{filename_stem}';
        """
    )
    results_df.to_sql("arc_data", con, index=False, if_exists="append")

    # And touch our flag file
    Path(flag_file).touch()

    if plot:
        # Plot 1D histograms of each residual
        fig, axs = plt.subplots(ncols=5, nrows=5, sharex=True, constrained_layout=True)

        for ax, xmean_values in zip(axs.ravel(), xmeans.T):
            ax.hist(xmean_values - np.nanmean(xmean_values), bins="fd")
            ax.set_xlim(-1, 1)
            ax.set_yticks([])
        fig.savefig(
            f"arc_{filename_stem}_pixel_residuals_histogram.pdf",
            bbox_inches="tight",
        )

        # Plot residuals against fibre number
        fig, axs = plt.subplots(ncols=5, nrows=5, sharex=True, constrained_layout=True)
        fibnum = np.arange(N_fibres)
        fig, axs = plt.subplots(ncols=5, nrows=5, sharex=True, constrained_layout=True)
        for ax, xmean_values in zip(axs.ravel(), xmeans.T):
            ax.scatter(fibnum, xmean_values - np.nanmean(xmean_values), marker="o", s=5)
            ax.set_ylim(-1, 1)
        fig.savefig(
            f"arc_{filename_stem}_pixel_residual_vs_fibnum.pdf",
            bbox_inches="tight",
        )

        # Plot the 2D residuals
        fig, axs = plt.subplots(
            ncols=2,
            figsize=(15, 5),
            gridspec_kw=dict(width_ratios=[1, 0.02]),
            constrained_layout=True,
        )
        im = axs[0].matshow(
            xmeans.T - np.nanmean(xmeans, axis=0)[:, None],
            aspect="auto",
            vmin=-1,
            vmax=1,
            cmap="RdBu",
        )
        # Add horizontal lines at each line fit
        [axs[0].axhline(y=i, linestyle="-", alpha=0.1, c="k") for i in range(N_lines)]
        axs[0].set_xlabel("Fibre Number")
        axs[0].xaxis.set_label_position("top")
        axs[0].set_ylabel("Arc Line Number")

        fig.colorbar(im, cax=axs[1], label="Residual (pixels)")
        fig.savefig(
            f"arc_{filename_stem}_pixel_residuals_vs_fibre_number_2D.pdf",
            bbox_inches="tight",
        )
