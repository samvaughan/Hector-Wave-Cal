import pandas as pd
import sqlite3
import numpy as np

con = sqlite3.connect("results/database/arc_data.db")

df = pd.read_sql("select * from arc_data", con)

input_matrix = np.zeros((25, 855 * 6))
input_matrix_errors = np.zeros((25, 855 * 6))
for i, wave in enumerate(df.wavelength.unique()):
    input_matrix[i, :] = df.loc[df.wavelength == wave, "x"]
    input_matrix_errors[i, :] = df.loc[df.wavelength == wave, "x_error"]


n_components = 10
mask = (input_matrix / input_matrix_errors) < 10

input_matrix[mask] = np.nan

yy = input_matrix - np.nanmean(input_matrix, axis=1)[:, None]
tolerance = 0.0001
mse = 1
n_iters = 0
while mse > tolerance:
    # Replace outliers/NaNs with the overall mean
    yy[~np.isfinite(yy)] = np.nanmean(yy)
    U, S, V = np.linalg.svd(yy, full_matrices=False)
    lowdim_representation = np.dot(
        U[:, :n_components] * S[:n_components], V[:n_components, :]
    )
    residuals = (yy - lowdim_representation) ** 2
    mse = np.nanmean(residuals)
    yy[residuals > 5 * mse] = np.nan
    n_iters += 1
