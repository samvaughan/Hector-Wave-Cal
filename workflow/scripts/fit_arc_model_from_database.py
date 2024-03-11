import sqlite3
import utils

# inputs
smk = snakemake  # noqa
database_file = smk.input.database
arc_name = smk.params.arc_id
output_file = smk.output.parameters
plot_filename = smk.output.plot_filename
N_x = smk.params.N_x
N_y = smk.params.N_y
intensity_cut = smk.params.intensity_cut
ccd_number = arc_name[5]
N_params_per_slitlet = (N_x + 1) * (N_y + 1)

# Connect to the database
con = sqlite3.connect(database_file)
df_full = utils.load_arc_from_db(con, arc_name)


# Set up the fitting
df_fitting, wave_standardised, X2 = utils.set_up_arc_fitting(
    df_full, N_x=N_x, N_y=N_y, intensity_cut=intensity_cut
)
wavelengths = df_fitting.wave.values

# Do the fitting
model = utils.fit_model(X=X2, y=wave_standardised)

# Get the predictions and the mean squared error
predictions = utils.get_predictions(model, X2, wavelengths)
mse = utils.calculate_MSE(model, X2, wavelengths)

# Save the parameters
parameters = utils.save_parameters(
    output_file,
    df_fitting,
    model,
    N_params_per_slitlet,
    mse,
    arc_name,
)

# Make the plot
fig, axs = utils.plot_residuals(df_fitting, predictions, wavelengths)
fig.savefig(plot_filename)
