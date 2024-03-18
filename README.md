# Hector 2D wavelength calibration

Better wavelength calibration for the Hector Galaxy Survey. 

This repo contains code to build a two-dimensional model which infers wavelength in terms of $x$ and $y$ location on the CCD. This improves upon the standard arc-line calibration in `2dfdr` which fits an independent polynomial for each fibre, not utilising the fact that the wavelength solution will vary smoothly across the detector. Neighbouring fibres will therefore have similar wavelength solutions.

## How to use this code

### I want to apply this code to a single arc frame during data reduction

Use the script located in `workflow/scripts/fit_arc_model_from_command_line.py`. The required input parameters are:

- `reduced_arc_filename`: The filename of the _reduced_ arc frame, which has already been run through `2dfdr` once


### I want to measure the wavelength solutions of hundreds of arc frames, build a database and compare the results

The workflow management tool [snakemake](https://snakemake.readthedocs.io/en/stable/) can be used to run the code on many arcs one after another. This can be useful to build up an idea of how the derived parameters vary over time.

The entire pipeline can be executed by running:

```
snakemake --cores 1
```

from the directory folder. This will 


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

An environment.yaml file with these dependencies can be found in the repository. 

## Background

The usual approach of fitting a low-order polynomial to each fibre independently works well when the arc lines are bright throughout the spectrum. Unfortunately, the Hector team have found that there are areas where the arc lamps at the AAT are too weak for reliable centroiding, introducing noise into the wavelength calibration which appears as "wiggles" in the reduced arc frames. 

A two dimensional polynomial mitigates some of these issues by reducing the degrees of freedom of the problem: instead of fitting independent 5th-order polynomials to each of the ~800 fibres (i.e. 4000 free parameters), a two-dimensional model which describes how wavelength varies as a function of $x$ and $y$ can have significantly fewer. 

