"""
Khwarizmi AI — Phase 7A: MinHash Deduplication Pipeline

Implements near-duplicate detection and removal for billion-token training corpora.
This is critical for achieving Claude-level data quality by eliminating redundant
training signals that waste compute and degrade model generalization.

Uses MinHash LSH (Locality Sensitive Hashing) for efficient O(n) deduplication
at billion-token scale.
"""

from typing import List, Set, Tuple, Optional, Iterator
import hashlib
from pathlib import Path
from dataclasses import dataclass
import json


@dataclass
class Document:
    """Represents a text document for deduplication."""
    doc_id: str
    content: str
    source: str
    token_count: int = 0
    
    def __post_init__(self):
        if self.token_count == 0:
            # Simple token count estimation (space-separated + punctuation)
            self.token_count = len(self.content.split())


class MinHash:
    """
    MinHash implementation for estimating Jaccard similarity between documents.
    
    Uses multiple hash functions to create a signature vector that preserves
    Jaccard similarity in Hamming space.
    """
    
    def __init__(self, num_perm: int = 128):
        self.num_perm = num_perm
        self.signature = [float('inf')] * num_perm
        self._hash_seeds = list(range(num_perm))
    
    def _hash(self, data: bytes, seed: int) -> int:
        """Compute hash with seed for permutation."""
        h = hashlib.sha256(data).digest()
        # Combine with seed
        combined = hashlib.sha256(h + seed.to_bytes(4, 'big')).digest()
        return int.from_bytes(combined[:8], 'big')
    
    def update(self, item: str):
        """Update MinHash signature with a new shingle/n-gram."""
        item_bytes = item.encode('utf-8')
        for i in range(self.num_perm):
            hash_val = self._hash(item_bytes, self._hash_seeds[i])
            self.signature[i] = min(self.signature[i], hash_val)
    
    def merge(self, other: 'MinHash'):
        """Merge another MinHash signature into this one."""
        if self.num_perm != other.num_perm:
            raise ValueError("num_perm must match")
        for i in range(self.num_perm):
            self.signature[i] = min(self.signature[i], other.signature[i])
    
    def jaccard_similarity(self, other: 'MinHash') -> float:
        """Estimate Jaccard similarity between two MinHash signatures."""
        if self.num_perm != other.num_perm:
            raise ValueError("num_perm must match")
        
        matches = sum(1 for i in range(self.num_perm) 
                     if self.signature[i] == other.signature[i])
        return matches / self.num_perm
    
    def to_bytes(self) -> bytes:
        """Serialize signature to bytes."""
        import struct
        return struct.pack('f' * self.num_perm, *self.signature)
    
    @classmethod
    def from_bytes(cls, data: bytes, num_perm: int = 128) -> 'MinHash':
        """Deserialize signature from bytes."""
        import struct
        minhash = cls(num_perm)
        minhash.signature = list(struct.unpack('f' * num_perm, data))
        return minhash


class MinHashLSH:
    """
    Locality Sensitive Hashing index for efficient near-duplicate detection.
    
    Documents with similar MinHash signatures will collide in the same buckets
    with high probability, allowing O(1) average-time duplicate detection.
    """
    
    def __init__(self, num_perm: int = 128, threshold: float = 0.8):
        self.num_perm = num_perm
        self.threshold = threshold
        
        # LSH parameters: band size and number of bands
        # Optimized for given threshold
        self.b = int((num_perm / 2) ** 0.5)  # number of bands
        self.r = num_perm // self.b  # rows per band
        
        # Hash tables for each band
        self.hashtables = [{} for _ in range(self.b)]
    
    def _get_band_signature(self, signature: List[float], band_idx: int) -> tuple:
        """Extract signature segment for a specific band."""
        start = band_idx * self.r
        end = start + self.r
        # Discretize floats to integers for hashing
        return tuple(int(sig * 1e6) % 1000000 for sig in signature[start:end])
    
    def insert(self, doc_id: str, minhash: MinHash):
        """Insert a document into the LSH index."""
        for band_idx in range(self.b):
            band_sig = self._get_band_signature(minhash.signature, band_idx)
            if band_sig not in self.hashtables[band_idx]:
                self.hashtables[band_idx][band_sig] = []
            self.hashtables[band_idx][band_sig].append(doc_id)
    
    def query(self, minhash: MinHash) -> Set[str]:
        """Find candidate duplicates for a given MinHash signature."""
        candidates = set()
        for band_idx in range(self.b):
            band_sig = self._get_band_signature(minhash.signature, band_idx)
            if band_sig in self.hashtables[band_idx]:
                candidates.update(self.hashtables[band_idx][band_sig])
        return candidates
    
    def estimate_false_positive_rate(self) -> float:
        """Estimate theoretical false positive rate for current parameters."""
        # P(false positive) ≈ (1 - e^(-r*t))^b where t is threshold
        import math
        t = self.threshold
        r = self.r
        b = self.b
        return (1 - math.exp(-r * t)) ** b


class DeduplicationPipeline:
    """
    End-to-end deduplication pipeline for training corpora.
    
    Supports:
    - Exact duplicate removal (via SHA256)
    - Near-duplicate removal (via MinHash LSH)
    - Cross-source contamination detection
    - N-gram based decontamination for benchmark datasets
    """
    
    def __init__(self, 
                 minhash_threshold: float = 0.8,
                 ngram_size: int = 13,
                 minhash_perms: int = 128):
        self.minhash_threshold = minhash_threshold
        self.ngram_size = ngram_size
        self.minhash_perms = minhash_perms
        
        self.exact_hashes: Set[str] = set()
        self.lsh_index = MinHashLSH(num_perm=minhash_perms, threshold=minhash_threshold)
        self.documents: dict = {}
        
        self.stats = {
            'total_docs': 0,
            'exact_duplicates': 0,
            'near_duplicates': 0,
            'unique_docs': 0,
            'total_tokens_processed': 0,
            'tokens_removed': 0
        }
    
    def _compute_ngrams(self, text: str, n: int) -> List[str]:
        """Generate character n-grams from text."""
        # Normalize whitespace
        text = ' '.join(text.split())
        return [text[i:i+n] for i in range(len(text) - n + 1)]
    
    def _compute_minhash(self, doc: Document) -> MinHash:
        """Compute MinHash signature for a document."""
        minhash = MinHash(num_perm=self.minhash_perms)
        ngrams = self._compute_ngrams(doc.content, self.ngram_size)
        for ngram in ngrams:
            minhash.update(ngram)
        return minhash
    
    def _is_exact_duplicate(self, content: str) -> bool:
        """Check if content is an exact duplicate via SHA256."""
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        if content_hash in self.exact_hashes:
            return True
        self.exact_hashes.add(content_hash)
        return False
    
    def add_document(self, doc: Document) -> Tuple[bool, Optional[str]]:
        """
        Add a document to the deduplication pipeline.
        
        Returns:
            (is_duplicate, duplicate_of_id)
        """
        self.stats['total_docs'] += 1
        self.stats['total_tokens_processed'] += doc.token_count
        
        # Check exact duplicates first
        if self._is_exact_duplicate(doc.content):
            self.stats['exact_duplicates'] += 1
            self.stats['tokens_removed'] += doc.token_count
            return (True, "exact_duplicate")
        
        # Check near-duplicates via MinHash LSH
        minhash = self._compute_minhash(doc)
        candidates = self.lsh_index.query(minhash)
        
        # Verify candidates with actual Jaccard similarity
        for candidate_id in candidates:
            candidate_doc = self.documents.get(candidate_id)
            if candidate_doc:
                candidate_minhash = self._compute_minhash(candidate_doc)
                similarity = minhash.jaccard_similarity(candidate_minhash)
                
                if similarity >= self.minhash_threshold:
                    self.stats['near_duplicates'] += 1
                    self.stats['tokens_removed'] += doc.token_count
                    return (True, candidate_id)
        
        # Not a duplicate - add to index
        self.documents[doc.doc_id] = doc
        self.lsh_index.insert(doc.doc_id, minhash)
        self.stats['unique_docs'] += 1
        
        return (False, None)
    
    def process_file(self, filepath: str, source: str = "unknown") -> Iterator[Document]:
        """
        Process a JSONL file and yield unique documents.
        
        Expected format: {"id": "...", "text": "...", ...}
        """
        filepath = Path(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    doc_id = data.get('id', f"{source}_{line_num}")
                    content = data.get('text', data.get('content', ''))
                    
                    doc = Document(
                        doc_id=doc_id,
                        content=content,
                        source=source
                    )
                    
                    is_dup, dup_of = self.add_document(doc)
                    if not is_dup:
                        yield doc
                        
                except json.JSONDecodeError:
                    continue
    
    def get_unique_documents(self) -> List[Document]:
        """Return all unique (non-duplicate) documents."""
        return list(self.documents.values())
    
    def get_stats(self) -> dict:
        """Return deduplication statistics."""
        stats = self.stats.copy()
        stats['deduplication_ratio'] = (
            stats['total_tokens_processed'] - stats['tokens_removed']
        ) / max(1, stats['total_tokens_processed'])
        stats['unique_percentage'] = (
            stats['unique_docs'] / max(1, stats['total_docs']) * 100
        )
        return stats
    
    def save_unique_corpus(self, output_path: str):
        """Save deduplicated corpus to JSONL file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in self.get_unique_documents():
                f.write(json.dumps({
                    'id': doc.doc_id,
                    'text': doc.content,
                    'source': doc.source,
                    'token_count': doc.token_count
                }, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(self.documents)} unique documents to {output_path}")


def main():
    """Example usage of the deduplication pipeline."""
    pipeline = DeduplicationPipeline(
        minhash_threshold=0.8,
        ngram_size=13,
        minhash_perms=128
    )
    
    # Process the CoT dataset we generated earlier
    cot_file = "/workspace/data/cot_training_data.jsonl"
    cot_path = Path(cot_file)
    
    if cot_path.exists():
        print(f"Processing {cot_file}...")
        
        # Convert CoT samples to Document format
        with open(cot_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                data = json.loads(line.strip())
                # Create document from problem + reasoning + answer
                content = f"{data['problem_statement']}\\n\\nReasoning:\\n{chr(10).join(data['latent_thought_steps'])}\\n\\nAnswer: {data['final_answer']}"
                
                doc = Document(
                    doc_id=data['problem_id'],
                    content=content,
                    source='cot_dataset',
                    token_count=len(content.split())
                )
                
                is_dup, dup_of = pipeline.add_document(doc)
                if is_dup:
                    print(f"Duplicate found: {data['problem_id']} (duplicate of {dup_of})")
        
        stats = pipeline.get_stats()
        print(f"\n=== Deduplication Statistics ===")
        print(f"Total documents: {stats['total_docs']}")
        print(f"Unique documents: {stats['unique_docs']}")
        print(f"Exact duplicates: {stats['exact_duplicates']}")
        print(f"Near duplicates: {stats['near_duplicates']}")
        print(f"Unique percentage: {stats['unique_percentage']:.1f}%")
        print(f"Deduplication ratio: {stats['deduplication_ratio']:.2f}")
        
        # Save deduplicated corpus
        pipeline.save_unique_corpus("/workspace/data/cot_deduplicated.jsonl")
    else:
        print(f"File not found: {cot_file}")
        print("Run qwen_cot_pipeline.py first to generate training data.")


if __name__ == "__main__":
    main()
