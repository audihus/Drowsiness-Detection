import multiprocessing
import argparse
import dlib
from additional import config

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path serialized shape predictor model")
args = vars(ap.parse_args())

print("[INFO] mengatur hyperparameter algoritma...")
options = dlib.shape_predictor_training_options()

options.tree_depth = 4
options.nu = 0.1033
options.cascade_depth = 20
options.feature_pool_size = 677
options.num_test_splits = 295 
options.oversampling_amount = 29
options.oversampling_translation_jitter = 0
options.feature_pool_region_padding = 0.0975
options.lambda_param = 0.0251
options.be_verbose = True
options.num_threads = multiprocessing.cpu_count()

print("[INFO] hypermarameter yang dipilih:")
print(options)

print("[INFO] training model shape predictor...")
dlib.train_shape_predictor(config.TRAIN_PATH, args["model"], options)

