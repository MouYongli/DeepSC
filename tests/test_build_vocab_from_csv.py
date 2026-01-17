import types

import sys


def test_build_vocab_from_csv_basic(tmp_path):
    csv_path = tmp_path / "genes.csv"
    csv_path.write_text(
        "feature_name,id\n" "GeneA,0\n" "GeneB,2\n" "GeneC,7\n",
        encoding="utf-8",
    )

    # Stub heavy optional deps to allow importing utils without installing them
    for mod in ("scanpy", "wandb"):
        if mod not in sys.modules:
            sys.modules[mod] = types.ModuleType(mod)

    # Ensure the src directory is importable
    sys.path.append("/home/angli/baseline/DeepSC/src")
    from deepsc.utils.utils import build_vocab_from_csv

    vocab2id, id2vocab, pad_token, pad_id = build_vocab_from_csv(str(csv_path))

    assert pad_token == "<pad>"
    assert pad_id == 0

    # Raw ids are [0,2,7], all shifted by +1
    assert vocab2id["GeneA"] == 1
    assert vocab2id["GeneB"] == 3
    assert vocab2id["GeneC"] == 8

    max_gene_id = max(vocab2id["GeneA"], vocab2id["GeneB"], vocab2id["GeneC"])
    assert vocab2id["<cls>"] == max_gene_id + 1
    # Default special tokens are (<pad>, <cls>, <mlm>)
    assert vocab2id["<mlm>"] == max_gene_id + 2

    # id2vocab must be the inverse of vocab2id
    for token, idx in vocab2id.items():
        assert id2vocab[idx] == token


def test_build_vocab_from_csv_custom_tokens(tmp_path):
    csv_path = tmp_path / "genes_custom.csv"
    csv_path.write_text("feature_name,id\nX,5\n", encoding="utf-8")

    # Stub heavy optional deps to allow importing utils without installing them
    for mod in ("scanpy", "wandb"):
        if mod not in sys.modules:
            sys.modules[mod] = types.ModuleType(mod)

    # Ensure the src directory is importable
    sys.path.append("/home/angli/baseline/DeepSC/src")
    from deepsc.utils.utils import build_vocab_from_csv

    special_tokens = ("<p>", "<s>", "</s>")
    vocab2id, id2vocab, pad_token, pad_id = build_vocab_from_csv(
        str(csv_path), special_tokens=special_tokens
    )

    assert pad_token == "<p>"
    assert pad_id == 0

    # Raw id 5 shifted by +1 -> 6
    assert vocab2id["X"] == 6
    max_gene_id = vocab2id["X"]
    assert vocab2id["<s>"] == max_gene_id + 1
    assert vocab2id["</s>"] == max_gene_id + 2

    # Inverse mapping holds
    for token, idx in vocab2id.items():
        assert id2vocab[idx] == token
