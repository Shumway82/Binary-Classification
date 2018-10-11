from flask import Flask
from flask import request
from Binary_Classifier.inferencer import Inferencer, Inferencer_Params
from tfcore.utilities.preprocessing import Preprocessing
import gflags
import sys
import numpy as np
from PIL import Image
import requests
from io import BytesIO

class FlaskApp(Flask):

    def __init__(self, *args, **kwargs):
        super(FlaskApp, self).__init__(*args, **kwargs)

        flags = gflags.FLAGS
        gflags.DEFINE_string("model_dir", "/app/Nails", "Model directory")

        flags(sys.argv)
        model_params = Inferencer_Params(image_size=112,
                                         model_path=flags.model_dir)

        self.model_inferencer = Inferencer(model_params)

        self.pre_processing = Preprocessing()
        self.pre_processing.add_function_x(Preprocessing.Central_Crop(size=(960, 960)).function)
        self.pre_processing.add_function_x(Preprocessing.Crop_by_Center(treshold=32, size=(224 * 2, 224 * 2)).function)
        self.pre_processing.add_function_x(Preprocessing.DownScale(factors=4).function)

app = FlaskApp(__name__)

@app.route('/predict')
def predict():
    url = request.args.get('image_url')

    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = np.asarray(img)

    if img.ndim != 2:
        return "Image needs to be uint8"

    img_x, _ = app.pre_processing.run(img, None)

    probs = app.model_inferencer.inference(img_x)

    label = np.argmax(probs)
    labels = 'BAD NAIL' if label == 0 else 'GOOD NAIL'
    return_string = 'Prediction: ' + labels + ' | Propabilitys: ' + 'BAD: ' + str(probs[0,0]) + ' GOOD: ' + str(probs[0,1])
    return return_string

if __name__ == "__main__":
  app.run(host="0.0.0.0", debug=True)