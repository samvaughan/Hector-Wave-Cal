import sqlite3

smk = snakemake  # noqa

con = sqlite3.connect(smk.output.database)
c = con.cursor()

# Drop the table if it exists
c.executescript(
    """DROP TABLE IF EXISTS arc_data
        """
)

# Make the table
c.executescript(
    """
        CREATE TABLE IF NOT EXISTS arc_data
        ([index] INTEGER PRIMARY KEY,
        [x_pixel] FLOAT,
        [y_pixel] FLOAT,
        [wave] FLOAT,
        [fibre_number] INTEGER,
        [intensity] FLOAT,
        [linewidth] FLOAT,
        [slitlet] INTEGER,
        [CCD] INTEGER,
        [file_ID] TEXT)
        """
)

con.commit()
