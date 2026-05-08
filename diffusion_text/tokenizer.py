import os
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors


class TextTokenizer:
    SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]"]

    def __init__(self, tokenizer=None):
        self._tokenizer = tokenizer

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size=32000, special_tokens=None):
        if special_tokens is None:
            special_tokens = cls.SPECIAL_TOKENS

        tok = Tokenizer(models.BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            show_progress=True,
        )
        tok.train_from_iterator(text_iterator, trainer=trainer)

        bos_id = tok.token_to_id("[BOS]")
        eos_id = tok.token_to_id("[EOS]")
        tok.post_processor = processors.TemplateProcessing(
            single=f"[BOS]:0 $A:0 [EOS]:0",
            special_tokens=[("[BOS]", bos_id), ("[EOS]", eos_id)],
        )

        return cls(tok)

    @classmethod
    def load(cls, path):
        tok = Tokenizer.from_file(path)
        return cls(tok)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._tokenizer.save(path)

    def encode(self, text, add_special_tokens=True):
        if not add_special_tokens:
            proc = self._tokenizer.post_processor
            self._tokenizer.post_processor = None
            enc = self._tokenizer.encode(text)
            self._tokenizer.post_processor = proc
            return enc.ids
        return self._tokenizer.encode(text).ids

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], int):
            return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        return [self._tokenizer.decode(seq, skip_special_tokens=skip_special_tokens) for seq in ids]

    @property
    def vocab_size(self):
        return self._tokenizer.get_vocab_size()

    def token_to_id(self, token):
        return self._tokenizer.token_to_id(token)

    @property
    def pad_id(self):
        return self.token_to_id("[PAD]")

    @property
    def mask_id(self):
        return self.token_to_id("[MASK]")

    @property
    def bos_id(self):
        return self.token_to_id("[BOS]")

    @property
    def eos_id(self):
        return self.token_to_id("[EOS]")

    def save_meta(self, path, shard_files=None, shard_hashes=None):
        meta = {
            "vocab_size": self.vocab_size,
            "special_tokens": {tok: self.token_to_id(tok) for tok in self.SPECIAL_TOKENS},
            "training_shards": shard_files or [],
            "training_shard_hashes": shard_hashes or {},
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)
