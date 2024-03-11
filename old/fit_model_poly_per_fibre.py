import numpy as np
import pandas as pd
import sqlite3
from cmdstanpy import CmdStanModel
import matplotlib.pyplot as plt
import numpy.polynomial.chebyshev as cheb

# from pathlib import Path


def standardise(array):
    return 2 * (array - array.max()) / (array.max() - array.min()) + 1


def chebyshev_polynomial_array(X):
    a = np.array(
        [
            cheb.chebval(X, [0, 0, 0, 1]),  # x**3 term
            cheb.chebval(X, [0, 0, 1]),  # x**2 term
            cheb.chebval(X, [0, 1]),  # x term
        ]
    )
    return a


# inputs
database_file = "results/database/arc_data.db"
stan_file = "workflow/stan_models/new_model.stan"
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
df_full = df_full.loc[df_full.slitlet.isin([1, 2, 3])]
df = df_full.dropna().sample(1000)

N = len(df)
N_slitlets = len(np.unique(df.slitlet))
N_fibres = len(np.unique(df.fibre_number))

print(f"N is {N}")
print(f"N_slitlets is {N_slitlets}")
print(f"N_fibres is {N_fibres}")

wavelengths = df.loc[:, "wave"].values
slitlet_numbers = df.slitlet.astype(int)
fibre_numbers = df.fibre_number.astype(int)

x = df.x_pixel
y = df.y_pixel

x_standardised = 2 * (x - x.max()) / (x.max() - x.min()) + 1
y_standardised = 2 * (y - y.max()) / (y.max() - y.min()) + 1
wave_standardised = (wavelengths - wavelengths.mean()) / wavelengths.std()

X = chebyshev_polynomial_array(x_standardised).T

# X = np.c_[np.ones(N), x_values, y_values]

d = dict(zip(np.unique(slitlet_numbers), np.arange(N_slitlets)))
new_slitlet_numbers = np.array(list(map(lambda x: d[x], slitlet_numbers))) + 1

d2 = dict(zip(np.unique(fibre_numbers), np.arange(N_fibres)))
new_fibre_numbers = np.array(list(map(lambda x: d2[x], fibre_numbers))) + 1


data = dict(
    N=N,
    N_fibres=N_fibres,
    N_predictors=X.shape[1],
    wavelengths=wave_standardised,
    fibre_numbers=new_fibre_numbers,
    predictors=X,
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
residuals = wavelengths - ppc

one_realisation = residuals[0, :]

fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(15, 8))

axs[0, 0].hist(one_realisation, bins="fd")
axs[0, 0].set_title(rf"$\sigma$(Wavelength) = {residuals.ravel().std():.3f} A")

axs[0, 1].scatter(x, y, c=one_realisation, vmin=-0.5, vmax=0.5)

axs[1, 0].scatter(x, one_realisation, c=slitlet_numbers, cmap="prism")

axs[1, 1].scatter(y, one_realisation, c=slitlet_numbers, cmap="prism")
