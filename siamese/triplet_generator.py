"""
Module Description: Siamese Triplet Generator
Purpose: Exposes triplet generation functionality in the siamese package.
Author: Technical Lead
Version: 1.0.0
Dependencies: pandas, typing, utils.triplet_utils
"""

from typing import List, Tuple
import pandas as pd
from utils.triplet_utils import generate_triplets


class TripletGenerator:
    """
    Generates image triplets.
    """

    @staticmethod
    def get_triplets(df: pd.DataFrame) -> List[Tuple[str, str, str]]:
        """
        Constructs triplets (Anchor, Positive, Negative).

        Args:
            df: DataFrame metadata.

        Returns:
            List of triplets.
        """
        return generate_triplets(df)
