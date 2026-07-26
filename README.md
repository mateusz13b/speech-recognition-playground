# Voice Commands Recognition

Practice pet-project for voice command recognition in Python.

The project is built around the Google Speech Commands dataset and compares three model families:
- `raw-cnn`
- `raw-cnn-lstm`
- `spec-cnn`

Target commands:
- `go`
- `stop`
- `left`
- `right`
- `on`
- `off`
- `up`
- `down`

Additional classes used in training:
- `unknown`
- `background`

## Project Structure

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── results/
│   ├── checkpoints/
│   ├── metrics/
│   ├── plots/
│   └── weights/
├── scripts/
└── src/iop/
```

## 1. Clone Repository And Install Packages

Clone the repository:

```bash
git clone https://github.com/mateusz13b/speech-recognition-cnn-lstm-speccnn.git
cd speech-recognition-cnn-lstm-speccnn
```

Create and activate a virtual environment.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.venv\Scripts\Activate.ps1
```

Upgrade `pip` and install the project:

```bash
python -m pip install --upgrade pip
pip install -e .
```

Main dependencies are installed from `pyproject.toml`:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `soundfile`
- `torch`
- `sounddevice`
- `pyyaml`

If you want GPU training, make sure PyTorch is installed with CUDA support.

Check CUDA availability:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 2. Download Dataset And Prepare Data

This project uses the Google Speech Commands dataset.

Download `speech_commands_v0.02.tar.gz` and place it into:

```text
data/raw/speech_commands_v0.02.tar.gz
```

Then run:

```bash
python scripts/prepare_data.py
```

The script will:
- find the dataset archive or extracted folder
- extract the archive if needed
- build a speaker-aware split `80/10/10`
- create a manifest file
- create a short summary

Expected outputs:
- `data/processed/dataset_index.csv`
- `data/processed/summary.json`

If needed, you can also extract manually:

```bash
tar -xzf data/raw/speech_commands_v0.02.tar.gz -C data/raw
```

After manual extraction, this folder should exist:

```text
data/raw/speech_commands_v0.02
```

## 3. How To Change Network Architecture

All model architectures are defined in:

[`src/iop/models.py`](C:/Users/matth/STU/I-OP/git/src/iop/models.py)

Current models:
- `Conv1DClassifier` for `raw-cnn`
- `Conv1DLSTMClassifier` for `raw-cnn-lstm`
- `SpectrogramCNN` for `spec-cnn`

If you want to change the architecture, edit this file.

Typical things you can modify:
- number of convolution layers
- number of channels
- kernel sizes
- strides
- pooling layers
- LSTM hidden size and number of layers
- dropout
- fully connected head

Example workflow:
1. Open `src/iop/models.py`
2. Modify the selected model class
3. Save the file
4. Re-run training

Important:
- if you change the model architecture, old `.pt` files may become incompatible with the new model
- in that case, retrain the model and generate new weights

## 4. Train Models

Training is started with:

```bash
python scripts/train.py --model-name raw-cnn --epochs 10 --batch-size 32
```

Available model names:
- `raw-cnn`
- `raw-cnn-lstm`
- `spec-cnn`

Examples:

```bash
python scripts/train.py --model-name raw-cnn --epochs 30 --batch-size 32
python scripts/train.py --model-name raw-cnn-lstm --epochs 40 --batch-size 16
python scripts/train.py --model-name spec-cnn --epochs 60 --batch-size 32
```

Useful parameters:
- `--model-name` selects the architecture
- `--epochs` sets the number of epochs
- `--batch-size` sets batch size
- `--learning-rate` sets optimizer learning rate
- `--weight-decay` sets L2 regularization
- `--num-workers` sets dataloader workers
- `--device` can be `auto`, `cpu`, or `cuda`
- `--max-train-samples` limits train samples for quick experiments
- `--max-val-samples` limits validation samples for quick experiments

Example with explicit GPU:

```bash
python scripts/train.py --model-name spec-cnn --epochs 60 --batch-size 32 --learning-rate 0.0005 --weight-decay 0.0001 --device cuda
```

## 5. Evaluation, Metrics, And Saved Weights

Run evaluation with:

```bash
python scripts/evaluate.py --model-name raw-cnn
```

Examples:

```bash
python scripts/evaluate.py --model-name raw-cnn
python scripts/evaluate.py --model-name raw-cnn-lstm
python scripts/evaluate.py --model-name spec-cnn
```

The project tracks:
- accuracy
- recall
- macro F1
- train loss
- validation loss
- validation accuracy
- validation macro F1
- confusion matrix

Saved outputs:

Full training checkpoint:
- `results/checkpoints/<model-name>.pt`

Weights only:
- `results/weights/<model-name>_weights.pt`

Metrics and reports:
- `results/metrics/<model-name>_history.json`
- `results/metrics/<model-name>_test_metrics.json`
- `results/metrics/<model-name>_classification_report.json`
- `results/metrics/<model-name>_confusion_matrix.csv`

Plots:
- `results/plots/<model-name>_loss.png`
- `results/plots/<model-name>_val_accuracy.png`
- `results/plots/<model-name>_confusion_matrix.png`

Notes:
- the best model is saved according to validation accuracy
- `.pt` files can be loaded later for testing on real recordings

## 6. Test A Saved Model On A WAV File

If you already have trained weights, you can test a single audio file:

```bash
python scripts/predict_file.py --model-name raw-cnn --wav test/stop.wav
```

Examples:

```bash
python scripts/predict_file.py --model-name raw-cnn --wav test/stop.wav
python scripts/predict_file.py --model-name raw-cnn-lstm --wav test/stop.wav
python scripts/predict_file.py --model-name spec-cnn --wav test/stop.wav
```

The script prints:
- wav path
- model name
- weights path
- predicted label
- confidence
- top predictions

Audio requirements for input wav:
- `16 kHz`
- `mono`
- preferably `16-bit PCM`
- around `1 second`

## 7. Real Microphone Test

For a real microphone test, the practical workflow is:

1. Record audio with any open and trusted tool.
2. Convert the recording to:
   - `16 kHz`
   - `mono`
   - WAV format
3. Save the file, for example as:
   - `test/stop.wav`
4. Run `predict_file.py` on this recording.

Example:

```bash
python scripts/predict_file.py --model-name spec-cnn --wav test/stop.wav
```

Important note:
- audio conversion should be done manually with open resources or common audio tools
- the project expects a properly prepared `16 kHz mono` wav file before testing

In other words, for now the safest real test path is:
- record externally
- convert externally
- then test inside this project the same way we tested with saved wav files

## 8. Recommended Workflow

Minimal full pipeline:

```bash
git clone https://github.com/mateusz13b/speech-recognition-cnn-lstm-speccnn.git
cd speech-recognition-cnn-lstm-speccnn
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
python scripts/prepare_data.py
python scripts/train.py --model-name raw-cnn --epochs 10 --batch-size 32
python scripts/evaluate.py --model-name raw-cnn
python scripts/predict_file.py --model-name raw-cnn --wav test/stop.wav
```

## 9. Notes

- The dataset split is speaker-aware.
- The project is designed as a practice and portfolio pet-project.
- If you significantly change `models.py`, retrain the model before evaluating or testing old weights.
