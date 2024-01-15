import numpy as np
import pandas as pd
import sqlite3
from cmdstanpy import CmdStanModel
import matplotlib.pyplot as plt
from pathlib import Path

smk = snakemake  # noqa

# inputs
database_file = smk.input.database  # "results/database/arc_data.db"
stan_file = smk.input.stan_file  # "workflow/stan_models/hierarchical_model.stan"
arc_name = Path(smk.input.arc_fname).stem  # "15sep30004red"

# outputs
plot_fname = smk.output.plot
polynomial_values_file = smk.output.polynomial_values_file
fibre_numbers_file = smk.output.fibre_numbers_file

con = sqlite3.connect("results/database/arc_data.db")
df_full = pd.read_sql("select * from arc_data", con)

# Ignore NaNs
df_full = df_full.loc[df_full.filename == arc_name]
# df_full = df_full.loc[df_full.y < 100]
df = df_full.dropna().loc[df_full["x_error"] < 0.1]  # .sample(frac=0.01)


N = len(df)
N_fibres = len(np.unique(df.y))
x_polynomial_order = 3
y_polynomial_order = 2

print(f"N is {N}")

wavelengths = df.loc[:, "wavelength"].values
fibre_numbers = df.loc[:, "y"].astype(int) + 1
x = df.x.values
x_standardised = 2 * (x - x.max()) / (x.max() - x.min()) + 1

wave_standardised = (wavelengths - wavelengths.mean()) / wavelengths.std()

# Legendre Polynomials
x_values = np.c_[
    np.ones_like(x),
    x_standardised,
    0.5 * (3 * x_standardised**2 - 1),
    0.5 * (5 * x_standardised**3 - 3 * x_standardised),
]
sigma_x = df.loc[:, "x_error"].values
sigma_x_standardised = sigma_x * 2 / (x.max() - x.min())

# sigma_x_values = np.c_[
#     np.ones_like(x),
#     sigma_x_standardised,
#     np.abs(
#         (0.5 * (3 * x_standardised**2 - 1))
#         * (sigma_x_standardised / sigma_x_standardised)
#         * 2
#     ),
#     np.abs(
#         0.5
#         * (5 * x_standardised**3 - 3 * x_standardised)
#         * (sigma_x_standardised / sigma_x_standardised)
#         * 3
#     ),
# ]

# fibre_distances = np.unique(df.loc[:, "y"].astype(int))
# y_standardised = (fibre_distances - fibre_distances.mean()) / fibre_distances.std()
# y_values = np.c_[np.ones_like(y_standardised), y_standardised, y_standardised**2]


d = dict(zip(np.unique(fibre_numbers), np.arange(N_fibres)))
new_fibre_numbers = np.array(list(map(lambda x: d[x], fibre_numbers))) + 1

data = dict(
    N=N,
    N_fibres=N_fibres,
    x_polynomial_order=x_polynomial_order,
    # y_polynomial_order=y_polynomial_order,
    wavelengths=wave_standardised,
    fibre_numbers=new_fibre_numbers,
    x_values=x_values,
    # y_values=y_values,
)

model = CmdStanModel(stan_file=stan_file)
fit = model.sample(data=data)  # , adapt_delta=0.95, max_treedepth=15)

# vi = model.variational(data=data)
xx = np.arange(N_fibres)
a_vals = fit.stan_variable("a")
fig, axs = plt.subplots(
    ncols=2, nrows=2, figsize=(12.86, 6.35), constrained_layout=True
)
for i, ax in enumerate(axs.ravel()):
    ax.errorbar(
        xx,
        a_vals.mean(axis=0)[:, i],
        yerr=a_vals.std(axis=0)[:, i],
        linestyle="None",
        marker="o",
        label="Hierarchical Model",
    )
    ax.set_xlabel("Fibre Number")


fig.savefig(plot_fname, bbox_inches="tight")
np.save(
    polynomial_values_file,
    a_vals,
)
np.save(
    fibre_numbers_file,
    np.unique(fibre_numbers),
)
