"""
Tokenizer and Vocab classes for gene expression data.
Self-contained implementation without torchtext dependency.
"""
import json
import pickle
from pathlib import Path
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch

try:
    from typing_extensions import Self
except ImportError:
    from typing import Self


class SimpleVocab:
    """Simple vocabulary implementation to replace torchtext.vocab.Vocab"""

    def __init__(self, token2idx: OrderedDict, default_index: Optional[int] = None):
        self.vocab = token2idx  # OrderedDict[str, int]
        self.itos = list(token2idx.keys())  # index to string
        self.stoi = token2idx  # string to index
        self._default_index = default_index

    def __getitem__(self, token: str) -> int:
        """Get the index of a token"""
        if token in self.stoi:
            return self.stoi[token]
        if self._default_index is not None:
            return self._default_index
        raise KeyError(f"Token '{token}' not found in vocabulary")

    def __contains__(self, token: str) -> bool:
        """Check if token is in vocabulary"""
        return token in self.stoi

    def __len__(self) -> int:
        """Get vocabulary size"""
        return len(self.stoi)

    def set_default_index(self, index: int):
        """Set default index for unknown tokens"""
        self._default_index = index

    def get_stoi(self) -> Dict[str, int]:
        """Get string to index mapping"""
        return dict(self.stoi)

    def get_itos(self) -> List[str]:
        """Get index to string list"""
        return self.itos.copy()

    def insert_token(self, token: str, index: int):
        """Insert a token at a specific index"""
        if index != len(self.stoi):
            raise ValueError("Can only insert at the end for SimpleVocab")
        self.stoi[token] = index
        self.itos.append(token)


class GeneVocab(SimpleVocab):
    """
    Vocabulary for genes.
    """

    def __init__(
        self,
        gene_list_or_vocab: Union[List[str], SimpleVocab],
        specials: Optional[List[str]] = None,
        special_first: bool = True,
        default_token: Optional[str] = "<pad>",
    ) -> None:
        """
        Initialize the vocabulary.
        Note: add specials only works when init from a gene list.

        Args:
            gene_list_or_vocab (List[str] or SimpleVocab): List of gene names or a
                SimpleVocab object.
            specials (List[str]): List of special tokens.
            special_first (bool): Whether to add special tokens to the beginning
                of the vocabulary.
            default_token (str): Default token, by default will set to "<pad>",
                if "<pad>" is in the vocabulary.
        """
        if isinstance(gene_list_or_vocab, SimpleVocab):
            if specials is not None:
                raise ValueError(
                    "receive non-empty specials when init from a SimpleVocab object."
                )
            # Copy the vocab
            super().__init__(
                OrderedDict(gene_list_or_vocab.vocab),
                gene_list_or_vocab._default_index
            )
        elif isinstance(gene_list_or_vocab, list):
            token2idx = self._build_vocab_from_list(
                gene_list_or_vocab,
                specials=specials,
                special_first=special_first,
            )
            super().__init__(token2idx)
        else:
            raise ValueError(
                "gene_list_or_vocab must be a list of gene names or a SimpleVocab object."
            )

        if default_token is not None and default_token in self:
            self.set_default_token(default_token)

    @classmethod
    def from_file(cls, file_path: Union[Path, str]) -> Self:
        """
        Load the vocabulary from a file. The file should be either a pickle or a
        json file of token to index mapping.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if file_path.suffix == ".pkl":
            with file_path.open("rb") as f:
                vocab = pickle.load(f)
                return cls(vocab)
        elif file_path.suffix == ".json":
            with file_path.open("r") as f:
                token2idx = json.load(f)
                return cls.from_dict(token2idx)
        else:
            raise ValueError(
                f"{file_path} is not a valid file type. "
                "Only .pkl and .json are supported."
            )

    @classmethod
    def from_dict(
        cls,
        token2idx: Dict[str, int],
        default_token: Optional[str] = "<pad>",
    ) -> Self:
        """
        Load the vocabulary from a dictionary.

        Args:
            token2idx (Dict[str, int]): Dictionary mapping tokens to indices.
        """
        # initiate an empty vocabulary first
        _vocab = cls([])

        # add the tokens to the vocabulary, GeneVocab requires consecutive indices
        for t, i in sorted(token2idx.items(), key=lambda x: x[1]):
            _vocab.insert_token(t, i)

        if default_token is not None and default_token in _vocab:
            _vocab.set_default_token(default_token)

        return _vocab

    def _build_vocab_from_list(
        self,
        gene_list: List[str],
        min_freq: int = 1,
        specials: Optional[List[str]] = None,
        special_first: bool = True,
    ) -> OrderedDict:
        """
        Build a vocabulary from a gene list.
        """
        from collections import Counter

        counter = Counter(gene_list)

        if specials is not None:
            for tok in specials:
                if tok in counter:
                    del counter[tok]

        # Sort by frequency then alphabetically
        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[0])
        sorted_by_freq_tuples.sort(key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)

        # Add special tokens
        if specials is not None:
            if special_first:
                specials = specials[::-1]
            for symbol in specials:
                ordered_dict.update({symbol: min_freq})
                ordered_dict.move_to_end(symbol, last=not special_first)

        # Convert to indices
        token2idx = OrderedDict()
        for idx, (token, freq) in enumerate(ordered_dict.items()):
            if freq >= min_freq:
                token2idx[token] = idx

        return token2idx

    @property
    def pad_token(self) -> Optional[str]:
        """
        Get the pad token.
        """
        if getattr(self, "_pad_token", None) is None:
            self._pad_token = None
        return self._pad_token

    @pad_token.setter
    def pad_token(self, pad_token: str) -> None:
        """
        Set the pad token. Will not add the pad token to the vocabulary.

        Args:
            pad_token (str): Pad token, should be in the vocabulary.
        """
        if pad_token not in self:
            raise ValueError(f"{pad_token} is not in the vocabulary.")
        self._pad_token = pad_token

    def save_json(self, file_path: Union[Path, str]) -> None:
        """
        Save the vocabulary to a json file.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        with file_path.open("w") as f:
            json.dump(self.get_stoi(), f, indent=2)

    def set_default_token(self, default_token: str) -> None:
        """
        Set the default token.

        Args:
            default_token (str): Default token.
        """
        if default_token not in self:
            raise ValueError(f"{default_token} is not in the vocabulary.")
        self.set_default_index(self[default_token])

    def append_token(self, token: str) -> None:
        """
        Append a token to the vocabulary.
        """
        if token not in self:
            idx = len(self)
            self.stoi[token] = idx
            self.itos.append(token)


def tokenize_batch(
    data: np.ndarray,
    gene_ids: np.ndarray,
    return_pt: bool = True,
    append_cls: bool = True,
    include_zero_gene: bool = False,
    cls_id: int = 0,
    mod_type: np.ndarray = None,
    cls_id_mod_type: int = None,
) -> List[Tuple[Union[torch.Tensor, np.ndarray]]]:
    """
    Tokenize a batch of data. Returns a list of tuple (gene_id, count).
    """
    if data.shape[1] != len(gene_ids):
        raise ValueError(
            f"Number of features in data ({data.shape[1]}) does not match "
            f"number of gene_ids ({len(gene_ids)})."
        )

    tokenized_data = []
    for i in range(len(data)):
        row = data[i]
        mod_types = None
        if include_zero_gene:
            values = row
            genes = gene_ids
            if mod_type is not None:
                mod_types = mod_type
        else:
            idx = np.nonzero(row)[0]
            values = row[idx]
            genes = gene_ids[idx]
            if mod_type is not None:
                mod_types = mod_type[idx]
        if append_cls:
            genes = np.insert(genes, 0, cls_id)
            values = np.insert(values, 0, 0)
            if mod_type is not None:
                mod_types = np.insert(mod_types, 0, cls_id_mod_type)
        if return_pt:
            genes = torch.from_numpy(genes).long()
            values = torch.from_numpy(values).float()
            if mod_type is not None:
                mod_types = torch.from_numpy(mod_types).long()
        tokenized_data.append((genes, values, mod_types))
    return tokenized_data


def pad_batch(
    batch: List[Tuple],
    max_len: int,
    vocab: SimpleVocab,
    pad_token: str = "<pad>",
    pad_value: int = 0,
    cls_appended: bool = True,
    vocab_mod: SimpleVocab = None,
) -> Dict[str, torch.Tensor]:
    """
    Pad a batch of data. Returns a list of Dict[gene_id, count].
    """
    max_ori_len = max(len(batch[i][0]) for i in range(len(batch)))
    max_len = min(max_ori_len, max_len)

    pad_id = vocab[pad_token]
    if vocab_mod is not None:
        mod_pad_id = vocab_mod[pad_token]
    gene_ids_list = []
    values_list = []
    mod_types_list = []

    for i in range(len(batch)):
        gene_ids, values, mod_types = batch[i]

        if len(gene_ids) > max_len:
            # sample max_len genes
            if not cls_appended:
                idx = np.random.choice(len(gene_ids), max_len, replace=False)
            else:
                idx = np.random.choice(len(gene_ids) - 1, max_len - 1, replace=False)
                idx = idx + 1
                idx = np.insert(idx, 0, 0)
            gene_ids = gene_ids[idx]
            values = values[idx]
            if mod_types is not None:
                mod_types = mod_types[idx]
        if len(gene_ids) < max_len:
            gene_ids = torch.cat(
                [
                    gene_ids,
                    torch.full(
                        (max_len - len(gene_ids),), pad_id, dtype=gene_ids.dtype
                    ),
                ]
            )
            values = torch.cat(
                [
                    values,
                    torch.full((max_len - len(values),), pad_value, dtype=values.dtype),
                ]
            )
            if mod_types is not None:
                mod_types = torch.cat(
                    [
                        mod_types,
                        torch.full(
                            (max_len - len(mod_types),),
                            mod_pad_id,
                            dtype=mod_types.dtype,
                        ),
                    ]
                )

        gene_ids_list.append(gene_ids)
        values_list.append(values)
        if mod_types is not None:
            mod_types_list.append(mod_types)

    batch_padded = {
        "genes": torch.stack(gene_ids_list, dim=0),
        "values": torch.stack(values_list, dim=0),
    }
    if mod_types is not None:
        batch_padded["mod_types"] = torch.stack(mod_types_list, dim=0)
    return batch_padded


def tokenize_and_pad_batch(
    data: np.ndarray,
    gene_ids: np.ndarray,
    max_len: int,
    vocab: SimpleVocab,
    pad_token: str,
    pad_value: int,
    append_cls: bool = True,
    include_zero_gene: bool = False,
    cls_token: str = "<cls>",
    return_pt: bool = True,
    mod_type: np.ndarray = None,
    vocab_mod: SimpleVocab = None,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize and pad a batch of data. Returns a list of tuple (gene_id, count).
    """
    cls_id = vocab[cls_token]
    if mod_type is not None:
        cls_id_mod_type = vocab_mod[cls_token]
    tokenized_data = tokenize_batch(
        data,
        gene_ids,
        return_pt=return_pt,
        append_cls=append_cls,
        include_zero_gene=include_zero_gene,
        cls_id=cls_id,
        mod_type=mod_type,
        cls_id_mod_type=cls_id_mod_type if mod_type is not None else None,
    )

    batch_padded = pad_batch(
        tokenized_data,
        max_len,
        vocab,
        pad_token,
        pad_value,
        cls_appended=append_cls,
        vocab_mod=vocab_mod,
    )
    return batch_padded
