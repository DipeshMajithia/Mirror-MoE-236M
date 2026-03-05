import mlx.core as mx
import numpy as np
import random
import re
import os

try:
    from tokenizers import Tokenizer as HFTokenizer
    HAS_HF_TOKENIZERS = True
except ImportError:
    HAS_HF_TOKENIZERS = False

class CustomBPETokenizer:
    """Custom BPE Tokenizer trained on our data with proper encode/decode."""
    
    TOKENIZER_PATH = "custom_bpe_32k.json"
    
    def __init__(self):
        if not HAS_HF_TOKENIZERS:
            raise ImportError("tokenizers not installed. Run: pip install tokenizers")
        
        if not os.path.exists(self.TOKENIZER_PATH):
            raise FileNotFoundError(f"Tokenizer not found: {self.TOKENIZER_PATH}. Run: python train_tokenizer.py")
        
        self._tokenizer = HFTokenizer.from_file(self.TOKENIZER_PATH)
        self.vocab_size = self._tokenizer.get_vocab_size()
        print(f"Loaded custom BPE tokenizer: vocab_size={self.vocab_size}")
        
    def encode(self, text):
        """Encode text to token IDs."""
        return self._tokenizer.encode(text).ids
        
    def decode(self, tokens):
        """Decode token IDs to text."""
        # Handle numpy/mlx arrays
        if hasattr(tokens, 'tolist'):
            tokens = tokens.tolist()
        tokens = [int(t) for t in tokens]
        return self._tokenizer.decode(tokens)

class CharTokenizer:
    """Legacy char tokenizer for backwards compatibility."""
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for i,ch in enumerate(chars)}
        
    def encode(self, s):
        fallback = self.stoi.get(' ', 0)
        return [self.stoi.get(c, fallback) for c in s]
        
    def decode(self, l):
        if hasattr(l, 'tolist'):
            l = l.tolist()
        return ''.join([self.itos.get(int(i), '?') for i in l])

def load_tokenizer_and_data(path, use_bpe=True):
    """Load tokenizer. If use_bpe=True, returns custom BPE tokenizer."""
    if use_bpe:
        return CustomBPETokenizer()
    
    # Legacy char tokenizer
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        return CharTokenizer(text)
    else:
        return CharTokenizer("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")

class DataAugmentor:
    """
    On-the-fly string augmentation for code/JSON data.
    """
    def __init__(self):
        self.var_pattern = re.compile(r'\b([a-z_][a-z0-9_]*)\b')
        self.num_pattern = re.compile(r'\b\d+\b')
        
    def shuffle_variables(self, text):
        # Find potential variable names (simple heuristic)
        # Avoid keywords in a real scenario, but for now simple swap
        words = list(set(self.var_pattern.findall(text)))
        if len(words) < 2: return text
        
        # Shuffle a subset
        to_swap = random.sample(words, min(len(words), 5))
        shuffled = to_swap.copy()
        random.shuffle(shuffled)
        
        mapping = dict(zip(to_swap, shuffled))
        
        # Regex replace is tricky safely, so we do a simple pass if tokenized, 
        # but here we operate on string.
        # We will just replace distinct matches.
        for k, v in mapping.items():
            # simple replace might be dangerous (e.g. "var" inside "variable")
            # skipped for safety in this simple version, 
            # instead we just perturb numbers which is safer.
            pass
            
        return text

    def perturb_numbers(self, text):
        def repl(match):
            val = int(match.group())
            if random.random() < 0.3:
                return str(val + random.randint(-5, 5))
            return str(val)
        return self.num_pattern.sub(repl, text)
        
    def augment(self, text):
        # 1. Perturb numbers
        text = self.perturb_numbers(text)
        # 2. (Optional) Shuffle JSON keys - requires parsing, potentially slow.
        # We skip complex parsing for speed in this embedded loop.
        return text

class AugmentedClusterLoader:
    def __init__(self, tokenizer, batch_size, block_size, data_files, weights=None):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.block_size = block_size
        self.augmentor = DataAugmentor()
        
        # Files - track which are JSONL
        self.files = data_files
        self.valid_files = []
        self.is_jsonl = []
        for f in self.files:
            if os.path.exists(f):
                self.valid_files.append(f)
                self.is_jsonl.append(f.endswith('.jsonl'))
            else:
                print(f"Warning: {f} not found.")
        
        if not self.valid_files:
            raise ValueError("No data files found!")
            
        # Weights
        self.weights = weights if weights else [1.0/len(self.valid_files)] * len(self.valid_files)
        # Normalize weights
        s = sum(self.weights)
        self.weights = [w/s for w in self.weights]
        
        self.file_handles = [open(f, 'r', encoding='utf-8', errors='ignore') for f in self.valid_files]
        self.buffers = ["" for _ in self.valid_files]

    def _read_chunk(self, idx, min_chars):
        import json
        f = self.file_handles[idx]
        while len(self.buffers[idx]) < min_chars:
            if self.is_jsonl[idx]:
                # JSONL: read lines and extract text field
                line = f.readline()
                if not line:
                    f.seek(0)
                    line = f.readline()
                try:
                    obj = json.loads(line)
                    self.buffers[idx] += obj.get('text', '') + "\n"
                except:
                    pass
            else:
                # Plain text
                chunk = f.read(65536) 
                if not chunk:
                    f.seek(0) 
                    chunk = f.read(65536)
                self.buffers[idx] += chunk

    def __iter__(self):
        return self

    def __next__(self):
        c_idx = random.choices(range(len(self.valid_files)), weights=self.weights)[0]
        needed_tokens = self.batch_size * (self.block_size + 1)
        
        # We read text, augment, then tokenize
        # We need enough text to produce needed_tokens.
        # Approx chars per token ~ 1 (since char tokenizer)
        # Safe buffer: 1.5x
        
        tokens = []
        while len(tokens) < needed_tokens:
            self._read_chunk(c_idx, 4096)
            # Take a chunk
            text_slice = self.buffers[c_idx][:4096]
            
            # Augment (probabilistic)
            if random.random() < 0.3: # 30% augmentation chance
                text_slice = self.augmentor.augment(text_slice)
            
            new_tokens = self.tokenizer.encode(text_slice)
            tokens.extend(new_tokens)
            
            # Advance buffer (using original length logic? strictly speaking we might consume "virtual" text
            # but we just slide the buffer by the amount we read)
            self.buffers[c_idx] = self.buffers[c_idx][4096:]
            
        tokens = tokens[:needed_tokens]
        # Reshape
        # Convert to numpy/mlx
        batch_arr = np.array(tokens).reshape(self.batch_size, self.block_size + 1)
        return mx.array(batch_arr)
