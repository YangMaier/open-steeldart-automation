from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class LetterToDetermine:
    image: np.ndarray or None
    contour: np.ndarray or None


@dataclass
class PreSavedLetter:
    number: int
    image: np.ndarray or None
    contour: np.ndarray or None


@dataclass
class PreSavedLetterImages:
    number: int
    images: List[np.ndarray]


@dataclass
class LetterMatches:
    frame_letter: LetterToDetermine
    sequence_letter: PreSavedLetter
    match: float or None
