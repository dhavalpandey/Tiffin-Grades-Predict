import flask
import tensorflow as tf
import os
import numpy as np
from flask_cors import CORS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = flask.Flask(__name__)
CORS(app, origins=["https://tiffingrades.netlify.app", "http://localhost:3000"])

@app.route('/<input>', methods=['GET'])
def hello(input):
    params = input.split(',')

    # Data
    scores = list(map(int, params))

    input_grades = scores[:-1]
    output_grades = scores[1:]

    scores_prev = np.array(input_grades,  dtype=float)
    scores_now = np.array(output_grades,  dtype=float)

    # Model Layers
    l0 = tf.keras.layers.Dense(units=50, input_shape=[1])
    l1 = tf.keras.layers.Dense(units=50)
    l2 = tf.keras.layers.Dense(units=50)

    # Model
    model = tf.keras.Sequential([l0, l1, l2])
    model.compile(loss='mean_squared_error',
                optimizer=tf.keras.optimizers.Adam(1))

    history = model.fit(scores_prev, scores_now, epochs=500, verbose=False)


    def predict_next_score(current_score):
        if round(model.predict([current_score])[0][0]) > 100:
            return 100
        else:
            num = round(model.predict([current_score])[0][0])
            return num

    return flask.jsonify(prediction=str(predict_next_score(scores[-1])))

if __name__ == "__main__":
  app.run()