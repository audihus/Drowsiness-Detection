from additional import config
from sklearn.model_selection import ParameterGrid
import multiprocessing
import numpy as np
import random 
import time
import dlib
import cv2
import os

def evaluate_model_acc(xmlPath, predPath):
    return dlib.test_shape_predictor(xmlPath, predPath)

def evaluate_model_speed(predictor, imagePath, tests=10):
    timings = []

    for i in range(0, tests):
        image =  cv2.imread(config.IMAGE_PATH)
        gray =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        detector = dlib.get_frontal_face_detector()
        rects = detector(gray, 1)

        if len(rects) > 0:
            start = time.time()
            shape = predictor(gray, rects[0])
            end = time.time()
        
        timings.append(end - start)
    
    return np.average(timings)

cols = [
    "tree_depth",
    "nu",
    "cascade_depth",
    "feature_pool_size",
    "num_test_splits",
    "oversampling_translation_jitter",
    "inference_speed",
    "training_time",
    "training_error",
    "testing error",
    "model_size"
]

csv =  open(config.CSV_PATH, "w")
csv.write("{}\n".format(",".join(cols)))

procs = multiprocessing.cpu_count()
procs = config.PROCS if config.PROCS > 0 else procs

hyperparams = {
    "tree_depth": list(range(2,6,2)),
    "nu": [0.1, 0.25],
    "cascade_depth" : list(range(6,12,2)),
    "feature_pool_size": [500, 1000],
    "num_test_splits": [20, 30],
    "oversampling_amount": [5, 10],
    "oversampling_translation_jitter": [0.1, 0.25]
}

combos = list(ParameterGrid(hyperparams))
random.shuffle(combos)
sampledCombos = combos[:config.MAX_TRIALS]
print("[INFO] sampling {} of {} possible combinations".format(len(sampledCombos), len(combos)))

for (i, p) in enumerate(sampledCombos):
    print("[INFO] starting trial {}/{}...".format(i + 1, len(sampledCombos)))

    options = dlib.shape_predictor_training_options()
    options.tree_depth = p["tree_depth"]
    options.nu = p["nu"]
    options.cascade_depth = p["cascade_depth"]
    options.feature_pool_size = p["feature_pool_size"]
    options.num_test_splits = p["num_test_splits"]
    options.oversampling_amount = p["oversampling_amount"]
    options.oversampling_translation_jitter = p["oversampling_translation_jitter"]

    options.be_verbose = True
    options.num_threads = procs

    start = time.time()
    dlib.train_shape_predictor(config.TRAIN_PATH, config.TEMP_MODEL_PATH, options)
    trainingTime = time.time() - start

    trainingError = evaluate_model_acc(config.TRAIN_PATH, config.TEMP_MODEL_PATH)
    testingError = evaluate_model_acc(config.TEST_PATH, config.TEMP_MODEL_PATH)

    predictor = dlib.shape_predictor(config.TEMP_MODEL_PATH)
    inferenceSpeed = evaluate_model_speed(predictor, config.IMAGE_PATH)

    modelSize = os.path.getsize(config.TEMP_MODEL_PATH)

    row = [
        p["tree_depth"],
		p["nu"],
		p["cascade_depth"],
		p["feature_pool_size"],
		p["num_test_splits"],
		p["oversampling_amount"],
		p["oversampling_translation_jitter"],
        inferenceSpeed,
        trainingTime,
        trainingError,
        testingError,
        modelSize,
    ]

    row= [str(x) for x in row]

    csv.write("{}\n".format(",".join(row)))

    if os.path.exists(config.TEMP_MODEL_PATH):
        os.remove(config.TEMP_MODEL_PATH)

print("[INFO] cleaning up...")
csv.close()