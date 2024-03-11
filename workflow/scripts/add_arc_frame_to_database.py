from pathlib import Path
import sqlite3
import utils
from pathlib import Path

smk = snakemake  # noqa

# Connect to the database
con = sqlite3.connect(smk.input.database)
arcdata_filename = Path(smk.input.dat_filename)
tlm_filename = smk.input.tlm_filename
reduced_arc_filename = smk.input.arc_filename

# Make the Arc Data
df = utils.read_arc(arcdata_filename, tlm_filename, reduced_arc_filename)


# Now add it to the database
stem = arcdata_filename.stem
file_id = stem.split("_")[1]

# Delete from the database if this file ID already exists
cur = con.cursor()
cur.executescript(
    f"""
    DELETE FROM arc_data
    WHERE arc_data.file_ID='{file_id}';
    """
)
print("\tDone")

df.to_sql("arc_data", con, index=False, if_exists="append")
Path(smk.output.flag_file).touch()
