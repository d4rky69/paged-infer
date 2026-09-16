"""Lightweight vocabulary and tokenizer for end-to-end inference."""

from typing import Dict, List


class SimpleTokenizer:
    """Deterministic character/word-level tokenizer mapping text to token IDs."""

    def __init__(self, vocab_size: int = 4096):
        self.vocab_size: int = vocab_size
        self.special_tokens: Dict[str, int] = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<unk>": 3,
        }
        self.inv_special_tokens: Dict[int, str] = {v: k for k, v in self.special_tokens.items()}

        # Generate sample word vocabulary for readable demos
        sample_words = [
            "the", "of", "and", "a", "to", "in", "is", "you", "that", "it",
            "he", "was", "for", "on", "are", "as", "with", "his", "they", "i",
            "at", "be", "this", "have", "from", "or", "one", "had", "by", "word",
            "but", "not", "what", "all", "were", "we", "when", "your", "can", "said",
            "paged", "attention", "continuous", "batching", "inference", "engine",
            "memory", "cache", "throughput", "latency", "system", "performance",
            "blocks", "tokens", "allocation", "gpu", "virtual", "server", "model",
            "transformer", "query", "key", "value", "algorithm", "speed", "optimization",
            "operating", "high", "efficiency", "distributed", "serving", "fast", "low",
        ]

        self.token_to_id: Dict[str, int] = dict(self.special_tokens)
        self.id_to_token: Dict[int, str] = dict(self.inv_special_tokens)

        # Populate ascii characters
        curr_id = len(self.special_tokens)
        for ch in range(32, 127):
            char_str = chr(ch)
            if char_str not in self.token_to_id:
                self.token_to_id[char_str] = curr_id
                self.id_to_token[curr_id] = char_str
                curr_id += 1

        # Populate sample words
        for word in sample_words:
            if word not in self.token_to_id and curr_id < vocab_size:
                self.token_to_id[word] = curr_id
                self.id_to_token[curr_id] = " " + word
                curr_id += 1

    def encode(self, text: str) -> List[int]:
        """Encodes text into token IDs."""
        tokens = []
        words = text.split()
        for word in words:
            lower = word.lower().strip(".,!?;:\"'")
            if lower in self.token_to_id:
                tokens.append(self.token_to_id[lower])
            else:
                for ch in word:
                    tokens.append(self.token_to_id.get(ch, self.special_tokens["<unk>"]))
        return tokens or [self.special_tokens["<bos>"]]

    def decode(self, token_ids: List[int]) -> str:
        """Decodes token IDs into readable text."""
        parts = []
        for tid in token_ids:
            if tid in self.inv_special_tokens:
                if tid == self.special_tokens["<eos>"]:
                    break
                continue
            token_str = self.id_to_token.get(tid, "")
            parts.append(token_str)
        return "".join(parts).strip()
