import numpy as np
import pandas as pd
import sqlite3
from cmdstanpy import CmdStanModel
import matplotlib.pyplot as plt
import numpy.polynomial.chebyshev as cheb

# from pathlib import Path


def standardise(array):
    return 2 * (array - array.max()) / (array.max() - array.min()) + 1


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
s_max = df_full.groupby("slitlet")["y_pixel"].max()[df_full.slitlet]
s_min = df_full.groupby("slitlet")["y_pixel"].min()[df_full.slitlet]
df_full["y_slitlet"] = (
    2 * (df_full["y_pixel"] - s_max.values) / (s_max.values - s_min.values)
) + 1
df_full = df_full.loc[df_full.intensity > 100]
df = df_full.dropna()  # .sample(frac=0.1)

N = len(df)
N_slitlets = len(np.unique(df.slitlet))
N_fibres = len(np.unique(df.fibre_number))
print(f"N is {N}")

wavelengths = df.loc[:, "wave"].values
slitlet_numbers = df.slitlet.astype(int)
fibre_numbers = df.fibre_number.astype(int)

x = df.x_pixel
y_standardised = df.y_slitlet

x_standardised = 2 * (x - x.max()) / (x.max() - x.min()) + 1
# y_standardised = 2 * (y - y.max()) / (y.max() - y.min()) + 1
wave_standardised = (wavelengths - wavelengths.mean()) / wavelengths.std()

X = chebyshev_polynomial_array(x_standardised, y_standardised).T


d = dict(zip(np.unique(slitlet_numbers), np.arange(N_slitlets)))
new_slitlet_numbers = np.array(list(map(lambda x: d[x], slitlet_numbers))) + 1

d_fibres = dict(zip(np.unique(fibre_numbers), np.arange(N_fibres)))
new_fibre_numbers = np.array(list(map(lambda x: d_fibres[x], fibre_numbers))) + 1

# Make the constants
constants = np.array(
    [
        (new_fibre_numbers == i).astype(int)
        for i in range(min(new_fibre_numbers), max(new_fibre_numbers) + 1)
    ]
).T


X_values_per_slitlet = np.column_stack(
    [np.where(new_slitlet_numbers[:, None] == i, X, 0) for i in range(1, 20)]
)


X2 = np.c_[constants, X_values_per_slitlet]

data = dict(
    N=N,
    N_predictors=X.shape[1],
    N_slitlets=N_slitlets,
    wavelengths=wave_standardised,
    predictors=X,
    slitlet_number=new_slitlet_numbers,
    wavelengths_std=wavelengths.std(),
    wavelengths_mean=wavelengths.mean(),
)
data["predictors"] = (data["predictors"] - data["predictors"].mean(0)) / data[
    "predictors"
].std(0)

model = CmdStanModel(stan_file=stan_file)
# fit = model.sample(data=data, max_treedepth=11)  # , adapt_delta=0.95, max_treedepth=12)
fit = model.sample(
    data=data, seed=123, max_treedepth=12
)  # , adapt_delta=0.95, max_treedepth=12)


# Get the residuals
ppc = fit.wavelengths_ppc
residuals = np.mean(wavelengths - ppc, 0)

fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(15, 8))

axs[0, 0].hist(residuals, bins="fd")
axs[0, 0].set_title(rf"$\sigma$(Wavelength) = {residuals.ravel().std():.3f} A")

axs[0, 1].scatter(
    df.x_pixel, df.y_pixel, c=residuals, vmin=-0.5, vmax=0.5, rasterized=True
)

axs[1, 0].scatter(x, residuals, c=slitlet_numbers, cmap="prism")

axs[1, 1].scatter(y_standardised, residuals, c=slitlet_numbers, cmap="prism")
