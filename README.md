# Hector 2D wavelength calibration

Better wavelength calibration for the Hector Galaxy Survey. 

This repo contains code to build a two-dimensional model which infers wavelength in terms of $x$ and $y$ location on the CCD. This improves upon the standard arc-line calibration in `2dfdr` which fits an independent polynomial for each fibre, not utilising the fact that the wavelength solution will vary smoothly across the detector. Neighbouring fibres will therefore have similar wavelength solutions.

## How to use this code

### I want to apply this code to a single arc frame during data reduction

Use the script located in `workflow/scripts/fit_arc_model_from_command_line.py`. The required input parameters are:

- `reduced_arc_filename`: The filename of the _reduced_ arc frame, which has already been run through `2dfdr` once.
- `dat_filename`: The filename of the 'dat' file which is produced during the `2dfdr` reduction. This file contains data about the arc lines identified in each fibre, with their initial $x$ pixel location and their "true" wavelength.
- `tlm_filename`: The filename of the tramline map which can be used to find the $y$ pixel location from a combination of the $x$ pixel location and the fibre number in question.

A number of optional input parameters are available:
- `--plot_residuals`: If True, display a 4-panel plot of the residuals from the model.
- `--save_params`: If a filename is given, save the fitted parameters to a netcdf (`.nc`) file.
- `--intensity-cut`: Ignore all arc lines with an "INTENSITY" value less than this value. Default is 20.
- `--N_x`: Polynomial order in the x direction. Default is 8.
- `--N_y`: Polynomial order in the y direction within each slitlet. Default is 4.
- `--no-update-arc`: If this parameter is passed, the script won't make a new WAVELA and SHIFTS array and save the results.
- `--saved-file-suffix`: A suffix which is added to the reduced arc filename to make the output filename which is saved at the end. Default is `_wavela_updated`.


### I want to measure the wavelength solutions of hundreds of arc frames, build a database and compare the results

The workflow management tool [snakemake](https://snakemake.readthedocs.io/en/stable/) can be used to run the code on many arcs one after another. This can be useful to build up an idea of how the derived parameters vary over time.

In the `resources` folder, there is a file called `input_files.csv` which can be edited to link together a combination of a reduced arc frame, a data file and a TLM map. Each row of this file corresponds to a single run of the arc model code. The current version contains an example for every Hector arc frame which has been observed up to March 2024. Note that the arc files themselves aren't included in the github repo. 

To run this code on your own selection of arcs, arrange the required folders as you wish and fill out this file.  The entire pipeline can then be executed by running:

```
snakemake --cores 1 --keep-going
```

from the directory folder. This will run the following steps, in order:

1. Create an SQLite database in `results/database/arc_data.db`.
2. Add the arc line measurements from a reduced arc frame into this database. The columns stored are:
    - `x_pixel`: $x$ coordinate of arc line.
    - `y_pixel`: $y$ coordinate of arc line.
    - `wave`: $\lambda$ of arc line.
    - `fibre_number`: Fibre number of this arc line.
    - `intensity`: Intensity (in counts? Not sure) of this arc line.
    - `linewidth`: Width (in pixels) of arc line.
    - `slitlet`: Number of the slitlet that an arc line was measured in.
    - `ccd`: Which CCD this measurement is from.
    - `file_id`: The ID string of the reduced arc file (e.g. `01jan40001`).
3. Fit the 2-dimensional model to the arc-line data, which is read from the SQLite database. The parameter values are saved as a netcdf (`.nc`) file under `results/FittedParameters` and a plot of the residuals for this arc frame are saved under `results/Plots`.

Note that running this code on ~1000 arc frames takes around 4-5 hours in total. 

## Dependencies

This package relies on the following packages:

- pandas
- numpy
- scipy
- astropy
- sklearn
- matplotlib
- xarray

If you want to run the entire workflow on multiple arc files, you will also need:

- snakemake
- sqlite3

An `environment.yaml` file with these dependencies can be found in the repository. 

## Background

The usual approach of fitting a low-order polynomial to each fibre independently works well when the arc lines are bright throughout the spectrum. Unfortunately, the Hector team have found that there are areas where the arc lamps at the AAT are too weak for reliable centroiding, introducing noise into the wavelength calibration which appears as "wiggles" in the reduced arc frames. 

A two dimensional polynomial mitigates some of these issues by reducing the degrees of freedom of the problem: instead of fitting independent 5th-order polynomials to each of the ~800 fibres (i.e. 4000 free parameters), a two-dimensional model which describes how wavelength varies as a function of $x$ and $y$ can have significantly fewer. 

## The fitting process

This program relies on `2dfdr` to identify the arc lines in a given arc exposure. The $x$ pixel, $y$ pixel and true wavelength for each arc line in every fibre on the detector are saved in a file named `arcfits.dat`. 

> [!TIP]
> A future improvement to the wavelength calibration could be to re-implement this arc line identification on the raw 2D data, rather than performing a 1D identification on each fibre after extraction using the tramline map.

The reduced arc frames which this code uses have already been "scrunched" by `2dfdr`. This means that the coordinates of each arc line are ($x$ pixel, fibre). The variation on the detector happens in ($x$ pixel, $y$ pixel) space, however, and so the first step of the code is to use the tramline

The polynomial model in this code splits up an AAOmega or Spector arc frame by slitlet. Within each slitlet, we fit a two dimensional polynomial which varies in the $x$ and $y$ direction.


