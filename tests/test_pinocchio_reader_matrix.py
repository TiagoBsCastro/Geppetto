import os
from pathlib import Path

import pytest
from scripts.validate_pinocchio_reader_matrix import validate_matrix


def test_pinocchio_reader_matrix_from_real_outputs():
    matrix_dir = os.environ.get("GEPPETTO_PINOCCHIO_READER_MATRIX_DIR")
    if not matrix_dir:
        pytest.skip("set GEPPETTO_PINOCCHIO_READER_MATRIX_DIR to validate real PINOCCHIO outputs")

    validate_matrix(Path(matrix_dir))
