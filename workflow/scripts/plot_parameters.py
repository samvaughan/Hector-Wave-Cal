import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import xarray as xr

all_param_files = Path("results/FittedParameters/").glob("*.nc")

fibre_constants = pd.DataFrame()

aaomega_fibre_numbers = np.arange(820)
spector_fibre_numbers = np.arange(855)

arc_names = []
ccd_numbers = []

datasets = []

for param_file in tqdm(all_param_files):
    arc_ID = param_file.stem.split("_")[0]
    ccd_number = arc_ID[5]

    tmp = xr.open_dataset(param_file)
    datasets.append(tmp)

ds = xr.concat(datasets, dim="ccd")

fibre_constants = ds["fibre_constants"]
slitlet_params = ds["slitlet_params"]

# Stuff for plotting
colours = ["steelblue", "deepskyblue", "crimson", "orangered"]
markers = ["o", "s", "o", "s"]
titles = ["AAOmega", "AAOmega", "Spector", "Spector"]
xx_slitlets = np.arange(19)
xx_fibres = np.arange(855)


def plot_fibre_constants(fibre_constants, spectrograph, sharey=True):
    # Plot the medians and quantiles for the fibre constants
    groupby_ccd_number = fibre_constants.groupby("ccd")
    ccd_medians = groupby_ccd_number.median(skipna=True)
    ccd_percentiles = [
        groupby_ccd_number.quantile(16 / 100, skipna=True),
        groupby_ccd_number.quantile(84 / 100, skipna=True),
    ]

    if spectrograph == "AAOmega":
        indices = [0, 1]
        fig, axs = plt.subplots(
            ncols=5, nrows=3, figsize=(13, 8), sharey=sharey, constrained_layout=True
        )
        axs[-1, -2].axis("off")
        axs[-1, -1].axis("off")
        n_slitlets = 13
        fibres_per_slitlet = 63
    else:
        indices = [2, 3]
        fig, axs = plt.subplots(
            ncols=5, nrows=4, figsize=(13, 8), sharey=sharey, constrained_layout=True
        )
        axs[-1, -1].axis("off")
        n_slitlets = 19
        fibres_per_slitlet = 45

    for i, colour, marker in zip(
        indices, [colours[k] for k in indices], [markers[k] for k in indices]
    ):
        for j, ax in enumerate(axs.ravel()[:n_slitlets]):
            data_mask = (xx_fibres > j * fibres_per_slitlet) & (
                xx_fibres < (j + 1) * fibres_per_slitlet
            )
            ax.plot(
                xx_fibres[data_mask],
                ccd_medians.data[i, data_mask],
                label=f"CCD {i+1}",
                c=colour,
                marker=marker,
                linestyle="None",
            )
            ax.fill_between(
                xx_fibres[data_mask],
                ccd_percentiles[0].data[i, data_mask],
                ccd_percentiles[1].data[i, data_mask],
                alpha=0.1,
                facecolor=colour,
            )
            ax.plot(
                xx_fibres[data_mask],
                ccd_percentiles[0].data[i, data_mask],
                linestyle="dashed",
                color=colour,
            )
            ax.plot(
                xx_fibres[data_mask],
                ccd_percentiles[1].data[i, data_mask],
                linestyle="dashed",
                color=colour,
            )
            ax.set_xlabel("Fibre Number")
            ax.set_ylabel("Value")
            ax.set_title(f"Slitlet {j+1}")
    axs[0, 0].legend()
    fig.suptitle("Fibre Constants")
    return fig, axs


fig, axs = plot_fibre_constants(fibre_constants, spectrograph="AAOmega", sharey=False)
fig.savefig("results/ParamPlots/fibre_constants_AAOmega.pdf")

fig, axs = plot_fibre_constants(fibre_constants, spectrograph="Spector", sharey=False)
fig.savefig("results/ParamPlots/fibre_constants_Spector.pdf")


# Plot the median and percentile of each parameter for each CCD
# Group by the CCD number and median
for j in slitlet_params.polynomial_parameter.data:
    groupby_ccd_number = slitlet_params.sel(polynomial_parameter=j).groupby("ccd")
    ccd_medians = groupby_ccd_number.median(skipna=True)
    ccd_percentiles = [
        groupby_ccd_number.quantile(16 / 100, skipna=True),
        groupby_ccd_number.quantile(84 / 100, skipna=True),
    ]

    fig, axs = plt.subplots(ncols=2, figsize=(13, 4))
    fig.suptitle(f"Polynomial Param {j}")
    for i, (ax, colour, marker, title) in enumerate(
        zip([axs[0], axs[0], axs[1], axs[1]], colours, markers, titles)
    ):
        ax.plot(
            xx_slitlets,
            ccd_medians.data[i, :],
            label=f"CCD {i+1}",
            c=colour,
            marker=marker,
            linestyle="None",
        )
        ax.fill_between(
            xx_slitlets,
            ccd_percentiles[0].data[i, :],
            ccd_percentiles[1].data[i, :],
            alpha=0.1,
            facecolor=colour,
        )
        ax.plot(
            xx_slitlets,
            ccd_percentiles[0].data[i, :],
            linestyle="dashed",
            color=colour,
        )
        ax.plot(
            xx_slitlets,
            ccd_percentiles[1].data[i, :],
            linestyle="dashed",
            color=colour,
        )
        ax.legend()
        ax.set_title(title)
        ax.set_xlabel("Slitlet Number")
        ax.set_ylabel("Value")
        ax.axhline(0.0, c="k", linestyle="dotted", zorder=1)

    fig.savefig(f"results/ParamPlots/param_{j:02}.pdf", bbox_inches="tight")
    plt.close("all")
