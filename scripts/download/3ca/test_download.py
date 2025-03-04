import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from utils import path_of_file


class TestPathOfFile(unittest.TestCase):
    @patch("pathlib.Path.iterdir")
    def test_file_in_current_directory_cell(self, mock_iterdir):
        mock_file = MagicMock()
        mock_file.name = "cell_data.csv"
        mock_file.is_file = lambda: True
        mock_iterdir.return_value = [mock_file]

        test_path = Path("/some/fake/directory")
        result = path_of_file(test_path, "cell")
        self.assertEqual(result, test_path / "cell_data.csv")

    @patch("pathlib.Path.iterdir")
    def test_file_in_parent_directory_gene(self, mock_iterdir):
        mock_file = MagicMock()
        mock_file.is_file.return_value = True
        mock_file.name = "gene_data.txt"
        mock_file.__str__.return_value = str(Path("/some/fake/gene_data.txt"))

        mock_iterdir.side_effect = [[], [mock_file]]

        test_path = Path("/some/fake/directory")

        result = path_of_file(test_path, "gene")

        expected_path = Path("/some/fake/gene_data.txt")
        self.assertEqual(result, expected_path)

    @patch("pathlib.Path.iterdir")
    def test_multiple_files_in_current_directory(self, mock_iterdir):
        mock_iterdir.return_value = [
            MagicMock(name="cell_data1.csv", is_file=lambda: True),
            MagicMock(name="cell_data2.csv", is_file=lambda: True),
        ]
        test_path = Path("/some/fake/directory")
        with patch("builtins.print") as mock_print:
            result = path_of_file(test_path, "cell")
            mock_print.assert_called_with(f"Multiple files found in path {test_path}")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
