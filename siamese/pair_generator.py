"""
Module Description: Siamese Pair Generator
Purpose: Exposes pair generation functionality in the siamese package.
Author: Technical Lead
Version: 1.0.0
Dependencies: pandas, typing, utils.pair_utils
"""

from typing import List, Tuple
import pandas as pd
from utils.pair_utils import generate_balanced_pairs


class PairGenerator:
    """
    Generates balanced image pairs.
    """

    @staticmethod
    def get_pairs(df: pd.DataFrame) -> Tuple[List[Tuple[str, str]], List[int]]:
        """
        Constructs balanced positive and negative pairs.

        Args:
            df: DataFrame metadata.

        Returns:
            Tuple of (pairs, labels).
        """
        return generate_balanced_pairs(df)
