#!/usr/bin/env python3
"""Build Detokenizer by reproducing C4's KMeans and aligning with T5."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
import torch
from transformers import T5Tokenizer

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PHASE0_RESULTS = Path(__file__).resolve().parent.parent / "phase0" / "results"

def load_sentences(num_sentences: int) -> list[str]:
    from datasets import load_dataset
    print("Loading wikitext-2-raw-v1...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    sentences = []
    for row in ds:
        text = row["text"].strip()
        if not text or text.startswith(" =") or len(text) < 30:
            continue
        sentences.append(text)
        if len(sentences) >= num_sentences:
            break
    return sentences

def main():
    print("Loading Phase 0 embeddings...")
    x_128 = np.load(PHASE0_RESULTS / "x_128.npy")
    seq_ids = np.load(PHASE0_RESULTS / "seq_ids.npy")
    pos_ids = np.load(PHASE0_RESULTS / "pos_ids.npy")

    # Reproduce C4's KMeans
    n_clusters = 1024
    subsample_size = min(50000, x_128.shape[0])
    rng = np.random.default_rng(42)
    sub_idx = rng.choice(x_128.shape[0], subsample_size, replace=False)
    
    print("Fitting KMeans to reproduce C4 pseudo-tokens...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024)
    kmeans.fit(x_128[sub_idx])

    # Now we want to map each of the 1024 cluster centers to a string.
    # To do this, we find the closest embedding in the subsample to each centroid.
    centers = kmeans.cluster_centers_  # (1024, 128)
    # Actually, let's find the nearest neighbor among all x_128 to be safe and accurate
    from sklearn.metrics import pairwise_distances_argmin_min
    print("Finding nearest original T5 embeddings to cluster centers...")
    closest_indices, distances = pairwise_distances_argmin_min(centers, x_128)

    # Now we have the index in x_128 for each cluster centroid.
    # We need to know what token that corresponds to.
    print("Loading original sentences to extract strings...")
    sentences = load_sentences(6000)
    
    print("Loading T5 tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=True)

    detokenizer_map = {}
    print("Mapping pseudo-tokens to strings...")
    for cluster_id in range(n_clusters):
        orig_idx = closest_indices[cluster_id]
        sid = seq_ids[orig_idx]
        pid = pos_ids[orig_idx]
        
        # Get the sentence
        text = sentences[sid]
        # Tokenize exactly like test_01
        toks = tokenizer(
            [text],
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = toks["input_ids"][0]
        # The T5 embedding corresponding to pos_id is input_ids[pos_id]
        # But wait, test_01 did:
        # valid_len = int(mask_np[j].sum())
        # embs = hidden[j, :valid_len]
        # all_seq_ids.append(...)
        # all_pos_ids.append(...)
        # So pid is exactly the index in the unpadded valid tokens.
        token_id = input_ids[pid].item()
        token_str = tokenizer.decode([token_id])
        
        # Clean up the string (T5 adds   for space)
        if token_str.startswith(' '):
            token_str = ' ' + token_str[1:]
            
        detokenizer_map[cluster_id] = token_str

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "detokenizer_map.json", "w") as f:
        json.dump(detokenizer_map, f, indent=2)

    print(f"Detokenizer map built with {len(detokenizer_map)} entries.")
    # Show a few examples
    for i in range(5):
        print(f"  ID {i} -> '{detokenizer_map[i]}'")

if __name__ == "__main__":
    main()
