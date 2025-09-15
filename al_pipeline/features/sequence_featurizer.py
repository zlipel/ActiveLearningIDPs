

import os
import numpy as np
from importlib.machinery import SourceFileLoader
import pandas as pd
from typing import List

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

MODEL_TO_FILE = {
    "hps_urry": "amino_acid_Urry.py",
    "hps_kr": "amino_acid_KR.py",
    "mpipi": "mpipi.py",
    "calvados": "calvados.py"
}

class SequenceFeaturizer:

    """Featurize amino acid sequences using coarse-grained models.

    Parameters
    ----------
    model_name : str
        Identifier for the force-field model to use (e.g., ``"mpipi"``).
    db_path : str
        Path to the database containing model parameters.

    Attributes
    ----------
    model : str
        Name of the selected model.
    db_path : str
        Filesystem location of the parameter database.
    charge_dict : dict
        Mapping from amino acid to bead charge.
    mass_dict : dict
        Mapping from amino acid to bead mass.
    lambda_dict : dict or None
        Hydrophobicity parameters when available.
    """


    def __init__(self, model_name: str, db_path: str):
        self.model = model_name
        self.db_path = db_path
        self._load_model_params()

    def _load_model_params(self):

        """Load force-field parameters for the chosen model.

        Side Effects
        ------------
        Populates dictionaries containing per-residue charge, mass and
        hydrophobicity values which are later used for feature generation.
        """


        self.ff_db = SourceFileLoader('ff_db', f"{self.db_path}/ff_db.py").load_module()
        self.ff_db.import_parameters(f"{self.db_path}/{MODEL_TO_FILE[self.model]}")
        self.atm_types = self.ff_db.atm_types
        self.charge_dict = {aa: self.atm_types[aa]['q'] for aa in AMINO_ACIDS}
        self.mass_dict = {aa: self.atm_types[aa]['m'] for aa in AMINO_ACIDS}
        self.lambda_dict = (
            {aa: self.atm_types[aa]['lam'] for aa in AMINO_ACIDS}
            if 'lam' in next(iter(self.atm_types.values())) else None
        )
        if self.model == "mpipi":
            self.nonbon_types = self.ff_db.nonbon_types
        if self.model == "calvados":
            # Add terminal charges
            for aa in AMINO_ACIDS:
                self.charge_dict[f"{aa}n"] = self.atm_types[f"{aa}n"]['q']
                self.charge_dict[f"{aa}c"] = self.atm_types[f"{aa}c"]['q']

    def featurize(self, sequence: str) -> list:

        """Generate numeric features for a single sequence.

        Parameters
        ----------
        sequence : str
            Amino acid sequence using the one-letter code.

        Returns
        -------
        list
            Ordered set of physicochemical features describing the sequence.
        """


        seq_len = len(sequence)
        comp = [sequence.count(aa) for aa in AMINO_ACIDS]
        entropy = -sum(p / seq_len * np.log2(p / seq_len) for p in comp if p > 0)

        net_charge = sum(self.charge_dict[aa] for aa in sequence)
        if self.model == "calvados":
            net_charge += self.charge_dict[f"{sequence[0]}n"] - self.charge_dict[sequence[0]]
            net_charge += self.charge_dict[f"{sequence[-1]}c"] - self.charge_dict[sequence[-1]]

        abs_net_charge = abs(net_charge)

        pos_frac = sum(1 for aa in sequence if self.charge_dict[aa] > 0)
        neg_frac = sum(1 for aa in sequence if self.charge_dict[aa] < 0)


        mass = sum(self.mass_dict[aa] for aa in sequence)

        if self.model == "mpipi":
            scd, shd, sum_lambda = self._extract_mpipi_features(sequence)
            return comp + [
                seq_len,
                scd,
                shd,
                abs_net_charge,
                sum_lambda,
                pos_frac,
                neg_frac,
                entropy,
                mass,
            ]
        else:
            scd = self._compute_scd(sequence)
            shd = self._compute_shd(sequence)
            lambda_sum = sum(self.lambda_dict[aa] for aa in sequence)
            return comp + [
                seq_len,
                scd,
                shd,
                abs_net_charge,
                lambda_sum,
                pos_frac,
                neg_frac,
                entropy,
                mass,
            ]

    def _compute_scd(self, seq):
        """Compute the sequence charge decoration (SCD)."""
        N = len(seq)
        return sum(
            self.charge_dict[seq[i]] * self.charge_dict[seq[j]] * np.sqrt(j - i)
            for i in range(N)
            for j in range(i + 1, N)
        ) / N

    def _compute_shd(self, seq):
        """Compute the sequence hydropathy decoration (SHD)."""
        N = len(seq)
        offset = 0.08 if self.model == "hps_urry" else 0.0
        return sum(
            ((self.lambda_dict[seq[i]] + self.lambda_dict[seq[j]])) / (j - i)
            for i in range(N)
            for j in range(i + 1, N)
        ) / N

    def _extract_mpipi_features(self, seq):
        """Extract SCD, SHD and average lambda for the Mpipi model."""
        n = len(seq)
        shd = 0.0
        scd = 0.0
        avg_lambda = sum(
            self.nonbon_types[tuple(sorted((aa, aa)))]['eps'] / 0.2 for aa in seq
        )

        for i in range(n):
            for j in range(i + 1, n):
                pair = tuple(sorted((seq[i], seq[j])))
                if pair in self.nonbon_types:
                    eps_ij = self.nonbon_types[pair]["eps"] / 0.2
                    shd += eps_ij / (j - i)
                    scd += self.charge_dict[seq[i]] * self.charge_dict[seq[j]] * np.sqrt(j - i)
        return [scd / n, shd / n, avg_lambda]
    
    def featurize_many(self, sequences):
        """Featurize a list of sequences and return a DataFrame."""
        feature_rows = [self.featurize(seq) for seq in sequences]
        columns = [f"{aa}" for aa in AMINO_ACIDS] + [
            "length",
            "SCD",
            "SHD",
            "|net charge|",
            "sum lambda",
            "beads(+)",
            "beads(-)",
            "shan ent",
            "mol wt",
        ]
        return pd.DataFrame(feature_rows, columns=columns)

