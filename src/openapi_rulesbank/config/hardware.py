# src/app/config/hardware.py
"""
Hardware detection and device configuration.
Detects GPU availability using torch.
Minimal side effects: Clears cache and prints device (can be silenced if needed).
"""

import torch

# Device for embeddings (GPU if available)
torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using Device: {device}")  # From your notebook images; remove if too verbose