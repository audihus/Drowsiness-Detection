# Dlib Shape Predictor Training

This folder contains the scripts used to create the custom eye-and-mouth
landmark models in [`../../models/eye-mouth-landmarks`](../../models/eye-mouth-landmarks).

| Script | Purpose |
| --- | --- |
| `train_shape_predictor.py` | Trains a dlib shape predictor with the selected hyperparameters. |
| `tune_predictor_hyperparams.py` | Searches for a better training configuration. |
| `evaluate_shape_predictor.py` | Calculates prediction error against an XML annotation file. |
| `parse_xml.py` | Prepares XML annotations for the selected facial landmarks. |
| `additional/config.py` | Defines dataset paths and tuning settings. |

## Dataset setup

The IBUG 300-W dataset is intentionally excluded from GitHub. Place the
extracted dataset in this folder using the path below before training:

```text
ibug_300W_large_face_landmark_dataset/
├── labels_ibug_300W_train_eyes_mouth.xml
└── labels_ibug_300W_test_eyes_mouth.xml
```

## Train and evaluate

From this directory:

```bash
python train_shape_predictor.py --model ../../models/eye-mouth-landmarks/eye_mouth_predictor.dat
python evaluate_shape_predictor.py --predictor ../../models/eye-mouth-landmarks/eye_mouth_predictor.dat --xml ibug_300W_large_face_landmark_dataset/labels_ibug_300W_test_eyes_mouth.xml
```

Training requires Python and dlib. The dataset files and generated tuning output
are excluded by `.gitignore`.
