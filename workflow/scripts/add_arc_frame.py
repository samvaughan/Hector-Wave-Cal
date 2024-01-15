from cycler import V
import pandas as pd
from tqdm import tqdm
import numpy as np
from pathlib import Path
import sqlite3
import scipy.interpolate as interpolate
from astropy.io import fits
from astropy.table import Table

smk = snakemake  # noqa

# Connect to the database
con = sqlite3.connect(smk.input.database)

# Load the arcfits.dat file
datafile = Path(smk.input.dat_filename)

# Load the tlm map
tlm_map = fits.open(smk.input.tlm_filename)
tlm = tlm_map["PRIMARY"].data
N_fibres, N_pixels = tlm.shape

# Load the arc file
arc_frame = fits.open(smk.input.arc_filename)
fibre_df = Table(arc_frame["FIBRES_IFU"].data).to_pandas()

# Get the file ID
stem = datafile.stem
file_id = stem.split("_")[1]
ccd = int(file_id[5])

fibre_numbers = []
data = []
N_arc_lines = []

# Reading the Arc Data file
with open(datafile, "r") as f:
    for i, line in enumerate(tqdm(f)):
        if line.startswith(" # FIBNO:"):
            fibre_numbers.append(int(line.split()[2]))
            continue
        elif line.startswith(" # fit parameters: "):
            N_arc_lines.append(int(line.split()[3]))
            continue
        elif line.startswith(
            " # I LNECHAN INTEN LNEWID CHANS WAVES FIT DEV ORIGPIX ORIGWAVES"
        ):
            column_names = line.lstrip("#").split()[1:]
            continue
        else:
            data.append([float(value) for value in line.split()])


df = pd.DataFrame(data, columns=column_names)
df["fibre_number"] = np.repeat(fibre_numbers, N_arc_lines)
df["CCD"] = ccd
df["file_id"] = file_id

# Now make the Y values
# Now interpolate the tlm map to get the Y value for each pixel
xx = np.arange(1, N_fibres + 1)
yy = np.arange(1, N_pixels + 1)
interp = interpolate.RegularGridInterpolator((xx, yy), values=tlm)
y_values = interp(np.c_[df.fibre_number.values, df.ORIGPIX.values])

df["y_pixel"] = y_values
df["x_pixel"] = df.ORIGPIX
df["wave"] = df.ORIGWAVES

# Now add the slitlet number
df = pd.merge(
    df,
    fibre_df.loc[:, ["SPEC_ID", "SLITLET"]],
    left_on="fibre_number",
    right_on="SPEC_ID",
    how="left",
)

# Delete from the database if this file ID already exists
cur = con.cursor()
cur.executescript(
    f"""
    DELETE FROM arc_data
    WHERE arc_data.file_ID='{file_id}';
    """
)
print("\tDone")

df = df.rename(dict(INTEN="intensity", LNEWID="linewidth"), axis=1)

df.loc[
    :,
    [
        "x_pixel",
        "y_pixel",
        "wave",
        "fibre_number",
        "intensity",
        "linewidth",
        "SLITLET",
        "CCD",
        "file_id",
    ],
].to_sql("arc_data", con, index=False, if_exists="append")
Path(smk.output.flag_file).touch()
