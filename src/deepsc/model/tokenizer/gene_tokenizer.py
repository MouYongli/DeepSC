# src/deepsc/model/tokenizer/gene_tokenizer.py

import os
import os.path as osp
from transformers import PreTrainedTokenizer

# 定义词表文件名的约定，便于保存和加载

here = osp.dirname(osp.abspath(__file__))
gene_vocab_file = osp.join(here, "gene_vocab.json")


class GeneTokenizer(PreTrainedTokenizer):
    
    def __init__(self, vocab_file, **kwargs):
        super(GeneTokenizer, self).__init__(vocab_file, **kwargs)
    
    def _tokenize(self, text):
        return text.split()
    
    def _convert_token_to_id(self, token):
        return self.vocab[token]
    
    def _convert_id_to_token(self, index):
        return self.ids_to_tokens[index]
    
    def _convert_token_to_string(self, token):
        return token
    
    def _encode(self, text):
        return [self._convert_token_to_id(token) for token in self._tokenize(text)]
    
    def _decode(self, ids):
        return [self._convert_id_to_token(i) for i in ids]
    
    def save_vocabulary(self, save_directory):
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)
        vocab_file = os.path.join(save_directory, "gene_vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(self.vocab_file)
        return vocab_file
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        if pretrained_model_name_or_path in ["gene_vocab"]:
            return cls(gene_vocab_file)
        raise ValueError(
            "Unrecognized pretrained model name: {}. Should be 'gene_vocab'".format(pretrained_model_name_or_path)
        )


if __name__ == "__main__":
    # 测试 GeneTokenizer
    tokenizer = GeneTokenizer(gene_vocab_file)
    print(tokenizer.tokenize("BRCA1 BRCA2"))
    print(tokenizer.convert_tokens_to_string(["BRCA1"]))
    print(tokenizer.encode("BRCA1 BRCA2"))