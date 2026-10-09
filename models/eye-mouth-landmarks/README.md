# Custom Eye & Mouth Landmark Models

These dlib shape-predictor models were trained for the Drowsiness Detection
project using eye-and-mouth annotations prepared from the IBUG 300-W dataset.
The detected landmarks can be used to calculate EAR and MAR.

| File | Description | Size |
| --- | --- | ---: |
| `eye_mouth_predictor.dat` | Initial trained model. | 57.4 MB |
| `optimal_eye_mouth_predictor.dat` | Tuned model selected for the final experiment. | 63.9 MB |

Load the tuned model with dlib:

```python
import dlib

predictor = dlib.shape_predictor(
    "models/eye-mouth-landmarks/optimal_eye_mouth_predictor.dat"
)
```

The original training scripts are retained outside this repository in the
project workspace. The training dataset is deliberately excluded because it is
too large to store in the source repository.

SHA-256 checksums:

```text
CA30832F9CA120304554E3B86FEB5F6A8B7E2A94B8EFE044DCB23C377CAA81B6  eye_mouth_predictor.dat
49D45744F2C19E322E7DFB557007E8F33D6A2B551A01F448A8674ED3856AD2B4  optimal_eye_mouth_predictor.dat
```
