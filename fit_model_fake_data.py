import numpy as np
import pandas as pd
import sqlite3
from cmdstanpy import CmdStanModel
import matplotlib.pyplot as plt
import numpy.polynomial.chebyshev as cheb

# from pathlib import Path


def standardise(array):
    return 2 * (array - array.max()) / (array.max() - array.min()) + 1


# Change this to poly2D?
def chebyshev_polynomial_array(X, Y):
    a = np.array(
        [
            cheb.chebval(X, [0, 0, 0, 0, 0, 0, 1]),  # x**3 term
            cheb.chebval(X, [0, 0, 0, 0, 0, 1]),  # x**3 term
            cheb.chebval(X, [0, 0, 0, 0, 1]),  # x**3 term
            cheb.chebval(X, [0, 0, 0, 1]),  # x**3 term
            cheb.chebval(X, [0, 0, 1]),  # x**2 term
            cheb.chebval(X, [0, 1]),  # x term
            cheb.chebval(Y, [0, 0, 1]),  # y^2 term
            cheb.chebval(Y, [0, 1]),  # y term
            cheb.chebval(X, [0, 1]) * cheb.chebval(Y, [0, 0, 1]),  # xy^2
            cheb.chebval(X, [0, 1]) * cheb.chebval(Y, [0, 1]),  # xy
            cheb.chebval(X, [0, 0, 1]) * cheb.chebval(Y, [0, 1]),  # x^2y
        ]
    )
    return a


# a_values = np.array(
#     [
#         -7.85702618e-03,
#         -3.28480681e-02,
#         2.03402365e00,
#         5.13521747e-04,
#         1.13785367e-02,
#         -7.88097681e-03,
#         -1.52744185e-02,
#         4.87543249e-04,
#         2.82828025e-04,
#     ]
# )

a_values = np.load("testing/model2/a_vals.npy")

constants = np.array(
    [
        0.01107232,
        -0.00561191,
        -0.00039092,
        -0.00824042,
        -0.11767028,
        -0.01550647,
        -0.01861277,
        -0.01891680,
        -0.01986827,
        -0.01851473,
        -0.01676301,
        -0.01564785,
        -0.0105566,
        -0.00527314,
        0.00758051,
        0.00455058,
        0.01568728,
        0.02480981,
        0.0346494,
    ]
)
sigma = 9.520252095e-05

# inputs
database_file = "results/database/arc_data.db"
stan_file = "workflow/stan_models/xy_model_per_slitlet.stan"
arc_name = "24jul30035"

# # outputs
# plot_fname = smk.output.plot
# polynomial_values_file = smk.output.polynomial_values_file
# fibre_numbers_file = smk.output.fibre_numbers_file

con = sqlite3.connect("results/database/arc_data.db")
df_full = pd.read_sql(
    f"select * from arc_data where arc_data.file_ID = '{arc_name}'", con
)

# Ignore NaNs
# df_full = df_full.loc[(df_full.y % 85 == 0)]
# Ignore NaNs
s_max = df_full.groupby("slitlet")["y_pixel"].max()[df_full.slitlet]
s_min = df_full.groupby("slitlet")["y_pixel"].min()[df_full.slitlet]
df_full["y_slitlet"] = (
    2 * (df_full["y_pixel"] - s_max.values) / (s_max.values - s_min.values)
)
df_full = df_full.loc[df_full.intensity > 100]
df = df_full.dropna().sample(frac=0.1)

N = len(df)

x = df.x_pixel
y = df.y_pixel

x_standardised = 2 * (x - x.max()) / (x.max() - x.min()) + 1
y_standardised = df.y_slitlet

X = chebyshev_polynomial_array(x_standardised, y_standardised)

slitlet_numbers = df.slitlet.astype(int).values

wavelengths_true = np.zeros(N)
for i in range(N):
    wavelengths_true[i] = constants[slitlet_numbers[i] - 1] + np.dot(
        a_values[slitlet_numbers[i] - 1], X[:, i]
    )

wavelengths_measured = np.random.randn(N) * sigma + wavelengths_true

wave_standardised = (
    wavelengths_measured - wavelengths_measured.mean()
) / wavelengths_measured.std()


# X = np.c_[np.ones(N), x_values, y_values]
print(f"N is {N}")
data = dict(
    N=N,
    N_predictors=X.shape[0],
    N_slitlets=19,
    wavelengths=wave_standardised,
    predictors=X.T,
    slitlet_number=slitlet_numbers,
    wavelengths_std=wavelengths_measured.std(),
    wavelengths_mean=wavelengths_measured.mean(),
)

model = CmdStanModel(stan_file=stan_file)
# fit = model.sample(data=data, max_treedepth=11)  # , adapt_delta=0.95, max_treedepth=12)
fit = model.sample(
    data=data, seed=123, max_treedepth=12
)  # , adapt_delta=0.95, max_treedepth=12)


# Get the residuals
ppc = fit.wavelengths_ppc
residuals = wavelengths_measured - ppc

one_realisation = residuals[0, :]

fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(15, 8))

axs[0, 0].hist(one_realisation, bins="fd")
axs[0, 0].set_title(rf"$\sigma$(Wavelength) = {residuals.ravel().std():.3f} A")

axs[0, 1].scatter(x, y, c=one_realisation)

axs[1, 0].scatter(x, one_realisation, c=slitlet_numbers, cmap="prism")

axs[1, 1].scatter(y, one_realisation, c=slitlet_numbers, cmap="prism")
