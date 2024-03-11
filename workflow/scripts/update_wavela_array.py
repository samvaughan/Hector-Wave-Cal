import numpy as np
from astropy.io import fits
import sqlite3
import pandas as pd
import numpy.polynomial.chebyshev as cheb
import xarray as xr
import utils 
"""
Output we need is the wavelengths from our fit in an array which is (N_fibres vs N_pixels)
Goal here: 

* Take a list of x values the same length as the detector (4096 for Spector).
* Standardise them (between -1 and 1)
* Use the TLM map to get the y values of each fibre.
* Use the reduced arc to assign slitlet numbers to fibre
* Apply the best fitting model
* Save in the correct WAVELA array format
"""

database_file = "results/database/arc_data.db"
arc_name = "24jul30002"
tlm_map = "resources/tlm_maps/ccd_3/24jul30001tlm.fits"
N_x = 6
N_y = 4
ccd_number = "3"
if ccd_number in ["3", "4"]:
    ccd_name = "SPECTOR"
    N_slitlets_total = 19
    N_fibres_per_slitlet = 45
elif ccd_number in ["1", "2"]:
    ccd_name = "AAOmega"
    N_slitlets_total = 13
    N_fibres_per_slitlet = 63

arc_hdu = fits.open(f"resources/ArcFrames/{arc_name}red.fits")
arc_frame = arc_hdu[0].data
N_fibres_total, N_pixels_x = arc_frame.shape

# Connect to the database
con = sqlite3.connect(database_file)
df_full = utils.load_arc_from_db(con, arc_name)

# Ignore any duplicated columns we may have- e.g if an arc was added twice
df_full = df_full.drop_duplicates().dropna()
wavelengths = df_full.loc[:, "wave"].values

# Load the TLM map
tlm_map_hdu = fits.open(tlm_map)
tlm = tlm_map_hdu[0].data

# Make the values we want to predict on
x_values = np.tile(np.arange(N_pixels_x), N_fibres_total).ravel()
y_values = tlm.ravel().astype(float)  # Casting to float64 is important here!
fibre_numbers = np.arange(1, N_fibres_total + 1).repeat(N_pixels_x)
slitlet_numbers = 19 - np.ceil(fibre_numbers / N_fibres_per_slitlet) + 1

df_predict = pd.DataFrame(
    data=dict(
        x_pixel=x_values,
        y_pixel=y_values,
        fibre_number=fibre_numbers,
        slitlet=slitlet_numbers,
    )
)

# Add some columns to the predict dataframe
df_predict['ccd'] = ccd_number
df_predict['intensity'] = 100
df_predict['wave'] = 0
df_predict.loc[:10, 'wave'] = 10

df_predict, wave_standardised, X2 = utils.set_up_arc_fitting(df_predict, N_x=N_x, N_y=N_y, intensity_cut=20)


# # Make the Y values within each slitlet
# s_max = df_predict.groupby("slitlet")["y_pixels"].max()[df_predict.slitlet]
# s_min = df_predict.groupby("slitlet")["y_pixels"].min()[df_predict.slitlet]

# df_predict["y_slitlet"] = (
#     2 * (df_predict["y_pixels"] - s_max.values) / (s_max.values - s_min.values)
# ) + 1
# y_standardised = df_predict.y_slitlet.copy()

# x_standardised = (
#     2
#     * (df_predict.x_pixels - df_predict.x_pixels.max())
#     / (df_predict.x_pixels.max() - df_predict.x_pixels.min())
#     + 1
# )

# # Make the constants- we have a different constant for every fibre
# constants = np.array(
#     [(fibre_numbers == i).astype(int) for i in range(1, N_fibres_total + 1)]
# ).T

# # Get the Chebyshev polynomial columns
# # Note the order of N_x and N_y! This is deliberate...
# X = cheb.chebvander2d(x_standardised, y_standardised, [N_y, N_x])
# # And now make these on a per-slitlet basis, so all the coefficients we measure are per-slitlet
# X_values_per_slitlet = np.column_stack(
#     [
#         np.where(slitlet_numbers[:, None] == i, X, 0)
#         for i in range(1, N_slitlets_total + 1)
#     ]
# )

# # We now have one constant term per fibre (~720 terms) and (n_x + 1)(n_y + 1) Chebyshev polynomial coefficients per slitlet
# X2 = np.c_[constants, X_values_per_slitlet]


# Load the dataset
parameters = xr.open_dataset(f"results/FittedParameters/{arc_name}_parameters.nc")
beta_hat = np.concatenate(
    (parameters.fibre_constants.values, parameters.slitlet_params.values.ravel())
)
# Now make our predictions
predictions = X2 @ beta_hat

# The WAVELA array (in nanometres)
wavela = (
    predictions.reshape(N_fibres_total, N_pixels_x) * wavelengths.std()
    + wavelengths.mean() / 10
)

# Make the new shifts array
shifts_array = arc_hdu["SHIFTS"].data
new_shifts = np.zeros_like(shifts_array)
shifts_array[1, :] = 1.0


