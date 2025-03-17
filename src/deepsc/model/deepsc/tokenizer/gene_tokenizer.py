import re
import json
import os
import os.path
from transformers import PreTrainedTokenizer

# 获取当前模块所在目录，并构造默认词表文件路径
here = os.path.abspath(os.path.dirname(__file__))
gene_vocab_file = os.path.join(here, "gene_vocab.json")

class GeneTokenizer(PreTrainedTokenizer):
    """
    自定义的基因序列 Tokenizer，与 transformers 的 PreTrainedTokenizer 兼容。
    该 Tokenizer 从指定的词表文件（gene_vocab.json）中加载词表，
    并实现了基本的 tokenize、id 与 token 之间的转换等方法。
    """
    vocab_files_names = {"vocab_file": "gene_vocab.json"}
    pretrained_vocab_files_map = {
        "vocab_file": {
            "gene": "gene_vocab.json",
        }
    }
    pretrained_init_configuration = {
        "gene": {
            "do_lower_case": False,
        }
    }
    max_model_input_sizes = {"gene": 100000}

    def __init__(self, vocab_file: str=gene_vocab_file, **kwargs):
        # 加载词表文件
        with open(vocab_file, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        # 构造 id 到 token 的映射
        self.ids_to_tokens = {id_: token for token, id_ in self.vocab.items()}
        # 设置默认的特殊标记
        kwargs.setdefault("unk_token", "[UNK]")
        kwargs.setdefault("sep_token", "[SEP]")
        kwargs.setdefault("pad_token", "[PAD]")
        kwargs.setdefault("cls_token", "[CLS]")
        kwargs.setdefault("mask_token", "[MASK]")
        super().__init__(**kwargs)

    def _tokenize(self, text):
        """
        先去除空格和常见分隔符，然后贪心匹配词表中最长的子串。
        假设词表中的条目为完整的基因序列，如 "GTF3C6P3"。
        """
        text = re.sub(r"[ ,;]+", "", text)
        tokens = []
        i = 0
        while i < len(text):
            found = False
            for j in range(len(text), i, -1):
                substr = text[i:j]
                if substr in self.vocab:
                    tokens.append(substr)
                    i = j
                    found = True
                    break
            if not found:
                tokens.append(text[i])
                i += 1
        return tokens

    def convert_tokens_to_string(self, tokens):
        if tokens and isinstance(tokens[0], int):
            tokens = self.convert_ids_to_tokens(tokens)
        return " ".join(tokens)

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        """
        将 id 列表转换为对应的 token 列表。
        如果 skip_special_tokens 为 True，则过滤掉特殊标记。
        """
        tokens = [self.ids_to_tokens.get(i, self.unk_token) for i in ids]
        if skip_special_tokens:
            special_tokens = {self.unk_token, self.sep_token, self.pad_token, self.cls_token, self.mask_token}
            tokens = [t for t in tokens if t not in special_tokens]
        return tokens

    def convert_tokens_to_ids(self, tokens):
        """
        将 token 列表转换为对应的 id 列表。
        """
        if isinstance(tokens, str):
            tokens = self._tokenize(tokens)
        return [self.vocab.get(token, self.vocab.get(self.unk_token)) for token in tokens]

    def get_vocab(self):
        """
        返回词表字典。
        """
        return self.vocab

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
        将词表保存到指定目录下的文件中，并返回该文件路径。
        """
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)
        vocab_filename = (filename_prefix + "-" if filename_prefix else "") + "gene_vocab.json"
        vocab_file = os.path.join(save_directory, vocab_filename)
        with open(vocab_file, "w", encoding="utf-8") as writer:
            json.dump(self.vocab, writer, ensure_ascii=False)
        return (vocab_file,)

if __name__ == "__main__":
    tokenizer = GeneTokenizer(vocab_file=gene_vocab_file)
    print(tokenizer.tokenize("BRCA1, BRCA2"))
    print(tokenizer.encode("BRCA1 BRCA2"))
    print(tokenizer.decode([2897, 2899]))
