import argparse
import dlib

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--predictor", required=True,
	help="path fine-tuned model dlib ")
ap.add_argument("-x", "--xml", required=True,
	help="path file xml training dan testing")
args = vars(ap.parse_args())


print("[INFO] evaluating model...")
error = dlib.test_shape_predictor(args["xml"], args["predictor"])
print("[INFO] error: {}".format(error))