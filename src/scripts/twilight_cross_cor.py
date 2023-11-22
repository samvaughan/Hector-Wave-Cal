import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from ppxf import ppxf, ppxf_util as utils
import matplotlib.pyplot as plt
import scipy.constants as const
from tqdm import tqdm

filename = "src/data/raw/21apr10001red.fits"

hdu = fits.open(filename)

reference_spectrum = hdu[0].data[208, :]
wcs = WCS(hdu[0].header)
wave = wcs.pixel_to_world(np.arange(hdu[0].header["NAXIS1"]), 0)[0]

fig, ax = plt.subplots()
ax.plot(wave, reference_spectrum)

N_spectra = 819
starts = np.arange(200, 2000, 100)
offsets = np.zeros((N_spectra, len(starts)), dtype="float")
for i in tqdm(np.arange(1, N_spectra)):
    for j, start in enumerate(starts):
        spec = hdu[0].data[i, :]

        template_chunk = reference_spectrum[start - 2 : start + 100 + 2]
        template_wave_chunk = wave[start - 2 : start + 100 + 2]

        wave_chunk = wave[start : start + 100]
        chunk = spec[start : start + 100]

        log_chunk, log_lam_spec, velscale = utils.log_rebin(wave_chunk, chunk)
        log_template, log_lam_temp, velscale = utils.log_rebin(
            template_wave_chunk, template_chunk
        )

        templates = log_template.reshape(-1, 1)
        noise = np.ones_like(log_chunk)

        # Mask infs and NANs
        pixel_is_good = (np.isfinite(log_chunk)) & (~np.isnan(log_chunk))
        log_chunk[~pixel_is_good] = np.nanmean(pixel_is_good)
        assert np.all(np.isfinite(log_chunk)), "Should have no NaNs"

        start = [0, 0]
        vsyst = const.c / 1e3 * np.log(template_wave_chunk[0] / wave_chunk[0])

        pp = ppxf.ppxf(
            templates,
            log_chunk,
            noise,
            velscale,
            start,
            fixed=[0, 1],
            vsyst=vsyst,
            mask=pixel_is_good,
        )
        offsets[i, j] = pp.sol[0] / velscale
        # def run_ppxf_on_chunk(wave_chunk, chunk, reference_spectrum):
        #     log_chunk, log_lam, velscale = utils.log_rebin(wave_chunk, chunk)
        #     log_template, log_lam, velscale = utils.log_rebin(wave_chunk, chunk)
