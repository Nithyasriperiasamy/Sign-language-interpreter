# Dataset Setup

This project uses the **ASL Alphabet** dataset from Kaggle:
https://www.kaggle.com/datasets/grassknoted/asl-alphabet

It contains 87,000 images across 29 classes: `A`-`Z`, `space`, `del`, and `nothing`.

## Download options

### Option 1: Kaggle CLI (recommended)

```bash
pip install kaggle
# Place your kaggle.json API token in ~/.kaggle/kaggle.json first
kaggle datasets download -d grassknoted/asl-alphabet
unzip asl-alphabet.zip -d dataset/
```

### Option 2: Manual download

1. Visit the dataset page and click **Download**.
2. Extract the ZIP file.
3. Move the extracted `asl_alphabet_train` folder into this `dataset/` directory.

## Required folder structure

After setup, this directory should look like:

```
dataset/
    asl_alphabet_train/
        A/
            A1.jpg
            A2.jpg
            ...
        B/
        C/
        ...
        Z/
        space/
        del/
        nothing/
```

Each subfolder name is the class label and must match exactly (case-sensitive)
for `ImageDataGenerator` to pick it up correctly.

Once the data is in place, train the model with:

```bash
python train.py --data-dir dataset/asl_alphabet_train --epochs 30
```
