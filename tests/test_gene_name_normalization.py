import csv
import os
import tempfile

from scripts.preprocessing.gene_name_normalization import process_gene_names


def test_merge_multiple_forms_of_same_gene():
    """
    Test case: If the same gene appears in multiple forms across two input files—
    including approved name (TP53), lowercase version (tp53), alias name (P53),
    and previous symbol (LFS1)—and all of them map to the same HGNC-approved gene (TP53),
    then the merged result should include 4 rows, one for each matched input form.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cxg_path = os.path.join(tmpdir, "cxg.txt")
        ca3_path = os.path.join(tmpdir, "ca3.txt")
        hgnc_path = os.path.join(tmpdir, "hgnc.tsv")
        output_dir = os.path.join(tmpdir, "output")

        with open(cxg_path, "w") as f:
            f.write("TP53\n")
            f.write("tp53\n")

        with open(ca3_path, "w") as f:
            f.write("P53\n")  # alias name
            f.write("LFS1\n")  # previous name

        with open(hgnc_path, "w") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(
                [
                    "Approved symbol",
                    "Alias symbol",
                    "Previous symbol",
                    "Ensembl gene ID",
                ]
            )
            writer.writerow(["TP53", "P53", "LFS1", "ENSG00000141510"])

        process_gene_names(cxg_path, ca3_path, hgnc_path, intersec_output=output_dir)

        output_file = os.path.join(output_dir, "merged_matched_genes.csv")
        assert os.path.exists(output_file)

        with open(output_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # final output should contain 4 rows:
        assert len(rows) == 4

        # check that all rows map to the same approved name and Ensembl ID
        for row in rows:
            assert row["Approved Name"] == "TP53"
            assert row["Ensembl id"] == "ENSG00000141510"


def test_merge_same_approved_name_in_both_files():
    """
    Test case: When both CXG and 3CA input files contain the same gene with the same approved name,
    the merged output should contain only one entry representing that gene.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cxg_path = os.path.join(tmpdir, "cxg.txt")
        ca3_path = os.path.join(tmpdir, "ca3.txt")
        hgnc_path = os.path.join(tmpdir, "hgnc.tsv")
        output_dir = os.path.join(tmpdir, "output")

        # the same gene "BRCA1" in both files
        with open(cxg_path, "w") as f:
            f.write("BRCA1\n")

        with open(ca3_path, "w") as f:
            f.write("BRCA1\n")

        # build a minimal HGNC database with BRCA1
        with open(hgnc_path, "w") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(
                [
                    "Approved symbol",
                    "Alias symbol",
                    "Previous symbol",
                    "Ensembl gene ID",
                ]
            )
            writer.writerow(["BRCA1", "", "", "ENSG00000012048"])

        process_gene_names(cxg_path, ca3_path, hgnc_path, intersec_output=output_dir)

        # check the output file
        output_file = os.path.join(output_dir, "merged_matched_genes.csv")
        assert os.path.exists(output_file)

        with open(output_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # result should contain only one row for BRCA1
        assert len(rows) == 1
        assert rows[0]["Approved Name"] == "BRCA1"
        assert rows[0]["Ensembl id"] == "ENSG00000012048"
        assert rows[0]["occur in 3ca"] == "True"
        assert rows[0]["occur in cellxgene"] == "True"
