<<<<<<< HEAD
# speech-recognition-cnn-lstm-speccnn
pet project about voice command recognition, training different architechtures (raw cnn/raw cnn+lstm/spectrogram cnn) to recognize commands such as [go, stop, left, right, on, off, up, down] with noise or other sounds
=======
# I-OP

Minimal practice project for voice command recognition.

Planned pipeline:
- raw audio models: 1D CNN and 1D CNN + LSTM
- feature model: mel-spectrogram + 2D CNN
- inference from file and microphone

Step 1:
- place `speech_commands_v0.02.tar.gz` into `data/raw/`
- run `iop-prepare`
- get `data/processed/dataset_index.csv` and `summary.json`

Step 2:
- train a raw baseline
- run `python scripts\train.py --model-name raw-cnn --epochs 5`
- save checkpoint in `results/checkpoints/`

- spectrogram model: spec-cnn
>>>>>>> 07e6696 (Initial commit)
