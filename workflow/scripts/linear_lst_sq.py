import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import numpy.polynomial.chebyshev as cheb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from utils import AAOmega_results, Spector_results


def standardise(array):
    return 2 * (array - array.max()) / (array.max() - array.min()) + 1


# inputs
# smk = snakemake  # noqa
# database_file = smk.input.database
# arc_name = smk.params.arc_id
# output_file = smk.output.parameters
# plot_filename = smk.output.plot_filename

database_file = "results/database/arc_data.db"
arc_name = "24jul30002"
output_file = "results/FittedParameters/{arc_id}_parameters.npy"
plot_filename = "results/Plots/{arc_id}_residuals.pdf"


N_x = 6
N_y = 6
N_params_per_slitlet = (N_x + 1) * (N_y + 1)
ccd_number = "3"

if ccd_number in ["3", "4"]:
    ccd_name = "SPECTOR"
    N_fibres_total = 855
    N_slitlets_total = 19
elif ccd_number in ["1", "2"]:
    ccd_name = "AAOmega"
    N_fibres_total = 820
    N_slitlets_total = 13
else:
    raise NameError(
        f"CCD number must be '1', '2', '3', '4' and of type string. Currently {ccd_number}, {type(ccd_number)}"
    )

con = sqlite3.connect("results/database/arc_data.db")
df_full = pd.read_sql(
    f"select * from arc_data where arc_data.file_ID = '{arc_name}'", con
)

# Ignore NaNs
s_max = df_full.groupby("slitlet")["y_pixel"].max()[df_full.slitlet]
s_min = df_full.groupby("slitlet")["y_pixel"].min()[df_full.slitlet]

# These are y values **within a slitlet**
df_full["y_slitlet"] = (
    2 * (df_full["y_pixel"] - s_max.values) / (s_max.values - s_min.values)
) + 1
df_full = df_full.loc[df_full.intensity > 100]
df = df_full.dropna()  # .sample(frac=0.1)

N = len(df)
N_alive_slitlets = len(np.unique(df.slitlet))
N_alive_fibres = len(np.unique(df.fibre_number))

N_missing_fibres = N_fibres_total - N_alive_fibres
N_slitlets_total = N_slitlets_total - N_alive_slitlets
print(f"We have {N} measured arc lines")


wavelengths = df.loc[:, "wave"].values
slitlet_numbers = df.slitlet.astype(int) - 1
fibre_numbers = df.fibre_number.astype(int) - 1

# Find the fibres which are missing/turned off/broken/etc
missing_fibre_numbers = list(set(np.arange(N_fibres_total)) - set(fibre_numbers))
missing_slitlet_numbers = list(set(np.arange(N_slitlets_total)) - set(slitlet_numbers))

x = df.x_pixel
y_standardised = df.y_slitlet

# Standardise the X and Wavelength values
x_standardised = 2 * (x - x.max()) / (x.max() - x.min()) + 1
# y_standardised = 2 * (y - y.max()) / (y.max() - y.min()) + 1
wave_standardised = (wavelengths - wavelengths.mean()) / wavelengths.std()

#
d = dict(zip(np.unique(slitlet_numbers), np.arange(N_alive_slitlets)))
new_slitlet_numbers = np.array(list(map(lambda x: d[x], slitlet_numbers)))

d_fibres = dict(zip(np.unique(fibre_numbers), np.arange(N_alive_fibres)))
new_fibre_numbers = np.array(list(map(lambda x: d_fibres[x], fibre_numbers)))

# Make the constants- we have a different constant for every fibre
constants = np.array(
    [
        (new_fibre_numbers == i).astype(int)
        for i in range(new_fibre_numbers.min(), new_fibre_numbers.max() + 1)
    ]
).T

# Get the Chebyshev polynomial columns
X = cheb.chebvander2d(x_standardised, y_standardised, [N_x, N_y])
# And now make these on a per-slitlet basis, so all the coefficients we measure are per-slitlet
X_values_per_slitlet = np.column_stack(
    [
        np.where(new_slitlet_numbers[:, None] == i, X, 0)
        for i in range(new_slitlet_numbers.min(), new_slitlet_numbers.max() + 1)
    ]
)

# We now have one constant term per fibre (~720 terms) and (n_x + 1)(n_y + 1) Chebyshev polynomial coefficients per slitlet
X2 = np.c_[constants, X_values_per_slitlet]


def cost_function(params):
    prediction = X2 @ params

    return np.sum((prediction - wave_standardised) ** 2)


print("Doing the fitting...")
model = Ridge(alpha=1e-3, fit_intercept=False)
model.fit(X2, wave_standardised)
beta_hat = model.coef_
print("Done!")

predictions = (X2 @ beta_hat) * wavelengths.std() + wavelengths.mean()
mse = np.sqrt(
    mean_squared_error(
        y_true=wavelengths,
        y_pred=predictions,
    )
)
print(f"The MSE is {mse:.3f} A")

# Save the outputs in a nice dataclass
# Fibre results
fibre_constants = dict(
    zip(np.unique(fibre_numbers).tolist(), beta_hat[:N_alive_fibres])
)
missing_fibre_constants = dict(
    zip(sorted(missing_fibre_numbers), [np.nan] * N_missing_fibres)
)
# Combine the dictionaries (https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-in-python)
all_fibre_constants = fibre_constants | missing_fibre_constants
all_fibre_constants = dict(sorted(all_fibre_constants.items()))

# Slitlet results
slitlet_params = dict(
    zip(
        np.unique(slitlet_numbers).tolist(),
        beta_hat[N_alive_fibres:].reshape(N_alive_slitlets, -1),
    )
)
missing_slitlet_params = dict(
    zip(sorted(missing_slitlet_numbers), np.full(N_params_per_slitlet, np.nan))
)
all_slitlet_params = slitlet_params | missing_slitlet_params

# Save as a dataclass I've made
if ccd_name == "SPECTOR":
    results = Spector_results(
        N_alive_fibres, N_alive_slitlets, all_fibre_constants, all_slitlet_params
    )
elif ccd_name == "AAOmega":
    results = AAOmega_results(
        N_alive_fibres, N_alive_slitlets, all_fibre_constants, all_slitlet_params
    )
else:
    raise NameError(f"CCD name must be one of SPECTOR or AAOmega: currently {ccd_name}")

np.save(output_file, results)


# Get the residuals
residuals = wavelengths - predictions

fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(13, 7), constrained_layout=True)

axs[0, 0].hist(residuals, bins="fd")
axs[0, 0].set_title(rf"$\sigma$(Wavelength) = {residuals.std():.3f} A")

axs[0, 1].scatter(
    df.x_pixel, df.y_pixel, c=residuals, vmin=-0.5, vmax=0.5, rasterized=True
)
axs[0, 1].set_xlabel("Detector x pixel")
axs[0, 1].set_ylabel("Detector y pixel")

axs[1, 0].scatter(x, residuals, c=slitlet_numbers, cmap="prism")
axs[0, 1].set_xlabel("Detector x pixel")
axs[0, 1].set_ylabel("Residuals (A)")

axs[1, 1].scatter(y_standardised, residuals, c=slitlet_numbers, cmap="prism")
axs[0, 1].set_xlabel("Detector y pixel")
axs[0, 1].set_ylabel("Residuals (A)")

fig.savefig(plot_filename, bbox_inches="tight")
# # Plot the fibre constants
# fig, ax = plt.subplots()
# ax.scatter(np.unique(fibre_numbers), beta_hat[:N_alive_fibres])

# # Plot the slitlet solutions
# params = beta_hat[N_alive_fibres:].reshape(N_alive_slitlets, -1)

# fig, axs = plt.subplots(ncols=7, nrows=7, figsize=(13, 7))
# for i, ax in enumerate(axs.ravel()):
#     ax.scatter(np.arange(N_alive_slitlets), params[:, i])
#     ax.axis("off")
#     ax.set_ylim(-0.5, 2.5)
