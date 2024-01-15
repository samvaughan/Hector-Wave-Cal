from dataclasses import dataclass
import numpy as np


class fitting_result:
    def __post_init__(self):
        """This function runs after __init__ in the dataclass.
        We'll use it for some basic sanity-checking of the arguments.
        """
        assert (
            len(self.fibre_constants) == self.N_fibres
        ), f"Must have {self.N_fibres} fibre constant values!"

        assert (
            len(self.slitlet_params) == self.N_slitlets
        ), f"Must have {self.N_slitlets} sets of slitlet parameters!"

        n_non_nan_fibres = np.sum(
            [np.isfinite(value) for key, value in self.fibre_constants.items()]
        )
        assert (
            n_non_nan_fibres == self.N_alive_fibres
        ), f"Must have {self.N_alive_fibres} fininite values for the fibre constants- currently {n_non_nan_fibres}"

    def make_predictions_array(self, input_matrix):
        fibre_constants = np.array(
            [value for key, value in self.fibre_constants.items() if np.isfinite(value)]
        )
        slitlet_params = np.array(
            [
                value
                for key, value in self.slitlet_params.items()
                if np.all(np.isfinite(value))
            ]
        ).ravel()

        beta = np.concatenate((fibre_constants, slitlet_params))
        return beta @ input_matrix


@dataclass
class AAOmega_results(fitting_result):
    N_fibres = 820
    N_slitlets = 13
    N_alive_fibres: int
    N_slitlet_with_live_fibres: int
    fibre_constants: dict
    slitlet_params: dict


@dataclass
class Spector_results(fitting_result):
    N_fibres = 855
    N_slitlets = 19
    N_alive_fibres: int
    N_slitlet_with_live_fibres: int
    fibre_constants: dict
    slitlet_params: dict
