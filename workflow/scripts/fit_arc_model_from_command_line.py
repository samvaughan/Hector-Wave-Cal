import utils
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "reduced_arc_filename",
    help="The filename of the reduced arc file, which usually ends in red.fits",
)
parser.add_argument(
    "dat_filename",
    help="The filename of the arc data file, which usually ends in .dat",
)
parser.add_argument("tlm_filename", help="The filename of the TLM map")
parser.add_argument(
    "--plot_residuals",
    action="store_true",
    help="Whether to display a plot of the residuals or not",
)
parser.add_argument(
    "--save_params",
    required=False,
    help="The netcdf filename of the paramters. Must end in .nc",
)

args = parser.parse_args()

arcdata_filename = args.dat_filename
tlm_filename = args.tlm_filename
reduced_arc_filename = args.reduced_arc_filename
plot_residuals = args.plot_residuals
save_params = args.save_params

arc_name = Path(reduced_arc_filename).stem.strip("red")
N_x = 8
N_y = 4
N_params_per_slitlet = (N_x + 1) * (N_y + 1)

# Read in the Arc Data
df_full = utils.read_arc(arcdata_filename, tlm_filename, reduced_arc_filename)

# Set up the fitting
df_fitting, wave_standardised, X2 = utils.set_up_arc_fitting(
    df_full, N_x=N_x, N_y=N_y, intensity_cut=20
)
wavelengths = df_fitting.wave.values

# Do the fitting
model = utils.fit_model(X=X2, y=wave_standardised)

# Get the predictions and the mean squared error
predictions = utils.get_predictions(model, X2, wavelengths)
mse = utils.calculate_MSE(model, X2, wavelengths)
print(f"The MSE is {mse:.3f} A\n")

# Save the parameters
if save_params is not None:
    parameters = utils.save_parameters(
        save_params,
        df_fitting,
        model,
        N_params_per_slitlet,
        mse,
        arc_name,
    )

# Optionally make a plot of the residuals
if plot_residuals:
    fig, axs = utils.plot_residuals(df_fitting, predictions, wavelengths)
    fig.show()


# Update the shifts and WAVELA array of the input arc file

