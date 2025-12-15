# NeAS: Neural Attenuation Fields
Unofficial implementation of NeAS that works with TIGRE formatted data.

## Prerequisites

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. Ensure you have it installed.

## Data Preparation

1. Download the dataset from [this Google Drive link](https://drive.google.com/drive/folders/1BJYR4a4iHpfFFOAdbEe5O_7Itt1nukJd).
2. Create a `data` directory in the root of the project if it doesn't exist.
3. Place the downloaded pickle file (e.g., `foot_50.pickle`) inside the `data` directory.

## Usage

To train the model, run the following command:

```bash
uv run src/train.py --data_path data/foot_50.pickle
```

### Arguments

- `--data_path`: Path to the input pickle file (default: `data/foot_50.pickle`).
- `--epochs`: Number of training epochs (default: 60).
- `--batch_size`: Batch size (default: 1).
- `--checkpoint_dir`: Directory to save checkpoints (default: `./checkpoints/`).
