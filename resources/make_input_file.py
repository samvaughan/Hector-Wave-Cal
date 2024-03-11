from astropy.io import fits
from tqdm import tqdm
from pathlib import Path
import pandas as pd

all_arc_files = Path(
    "/Users/samvaughan/Science/Hector/2d_wavelength_fit/fit_lines/resources/ArcFrames"
).glob("*red.fits")

parent = Path("/Users/samvaughan/Science/Hector/2d_wavelength_fit/fit_lines/")

all_files = pd.DataFrame()

for arc_file in tqdm(all_arc_files, total=1536):
    hdu = fits.open(arc_file)

    arc_id = arc_file.stem[:-3]
    frame_name = arc_id[:5] + arc_id[6:]

    tlm_map_name = (
        parent
        / "resources/tlm_maps"
        / Path(hdu[7].data[hdu[7].data["ARG_NAME"] == "TLMAP_FILENAME"][0][1])
    )
    arc_dat_name = parent / "resources/ArcDataFiles/" / Path(f"arcfits_{arc_id}.dat")

    row = pd.DataFrame(
        dict(
            arc_id=arc_id,
            frame_name=frame_name,
            arc_filename=arc_file.relative_to(parent),
            dat_filename=arc_dat_name.relative_to(parent),
            tlm_filename=tlm_map_name.relative_to(parent),
        ),
        index=[0],
    )
    all_files = pd.concat((all_files, row))

all_files = all_files.sort_values("arc_id").reset_index(drop=True)
all_files.to_csv("input_files.csv", index=False)
