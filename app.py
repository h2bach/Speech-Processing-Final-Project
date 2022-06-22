import base64
import re
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
import gender as gd
import emotion as em

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    """
    Render the main page
    """
    return render_template('index.html')


@app.route('/about', methods=['GET'])
def about():
    """
    Render the about page
    """
    return render_template('about.html')


@app.route('/gender', methods=['GET'])
def gender():
    """
    Render the gender page
    """
    return render_template('gender.html')


@app.route('/emotion', methods=['GET'])
def emotion():
    """
    Render the emotion page
    """
    return render_template('emotion.html')


@app.route('/gender/predict', methods=['GET', 'POST'])
def gender_predict():
    '''
    predict function to predict the image
    Api hits this function when someone clicks submit.
    '''

    #app.logger.info(request.get_json())
    #app.logger.info(base64.b64decode(request.get_json()))

    wav_file = open("temp.wav", "wb")
    audio_data = re.sub('^data:audio/.+;base64,', '', request.json)
    decode_string = base64.b64decode(audio_data)
    wav_file.write(decode_string)

    if request.method == 'POST':
        result = gd.predict("temp.wav")
        return jsonify(result=result)
    return None


@app.route('/emotion/predict', methods=['GET', 'POST'])
def emotion_predict():
    '''
    predict function to predict the image
    Api hits this function when someone clicks submit.
    '''

    wav_file = open("temp.wav", "wb")
    audio_data = re.sub('^data:audio/.+;base64,', '', request.json)
    decode_string = base64.b64decode(audio_data)
    wav_file.write(decode_string)

    if request.method == 'POST':
        #data = request.get_json()
        result = em.predict("temp.wav")
        app.logger.info(result)
        return jsonify(result=result)
    return None


if __name__ == '__main__':
    app.run(debug=True)
