# EvoProject

Genomic sequence embedding extraction and variant analysis using [Evo2](https://github.com/arcinstitute/evo2).

## Overview

EvoProject provides tools to extract representations from the Evo2 foundation model (e.g., `evo2_1b_base`) and compute difference matrices between wild-type DNA sequences and variant sequences (e.g., missense, silent variants) across model layers.

## Requirements

- Python >= 3.11, < 3.13 (Python 3.12 recommended)
- CUDA-compatible GPU (recommended for model inference)
- PyTorch with CUDA support
- FlashAttention (`flash-attn`) for accelerated attention computations

## Installation

### 1. Set up environment (using `uv` or `venv`)

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
```

### 2. Install PyTorch and FlashAttention

For CUDA 12.8:
```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn==2.8.0.post2 --no-build-isolation
```

### 3. Install EvoProject and dependencies

Install in editable mode:
```bash
pip install -e .
```

To install with development / notebook dependencies:
```bash
pip install -e ".[dev]"
```

## Usage

### Command Line Interface

You can run the extraction tool via the console script:

```bash
extract-embedding <path_to_wildtype.txt> <cut_site_index> <replacement_nucleotide> --destination <output_dir> --target_layer 17
```

Or invoke the script directly with Python:

```bash
python extract_embeding.py data/wildtype.txt 2222 G --destination differences/ --target_layer 17
```

### Interactive Notebooks

Jupyter notebooks for preliminary analysis and exploration are located in the `prelim/` directory:
- `prelim/create_datafile.ipynb`
- `prelim/testrun.ipynb`
- `prelim/testrun2.ipynb`