import numpy as np
from keras.models import model_from_json
import sys
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import librosa
import librosa.display




from IPython.display import Audio

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

EXAMPLE_SR_PATH = "temp.wav"
model_json = "model/emotion/model.json"
model_weight = "model/emotion/model.h5"


def getSampleRate():
    data, sample_rate = librosa.load(EXAMPLE_SR_PATH)
    return sample_rate

sample_rate = getSampleRate()

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

def extract_features(data):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))  # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))  # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))  # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))  # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))  # stacking horizontally

    return result


def get_features(path):
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    # Trích xuất đặc trưng âm thanh gốc
    res1 = extract_features(data)
    result = np.array(res1)

    # Trích xuất đặc trưng âm thanh bị chèn nhiễu
    noise_data = noise(data)
    res2 = extract_features(noise_data)
    result = np.vstack((result, res2))  # stacking vertically

    # Trích xuất đặc trưng âm thanh bị kéo
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch)
    result = np.vstack((result, res3))  # stacking vertically

    return result

def getEncoder():
    Y = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
    encoder = OneHotEncoder()
    Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()
    return encoder

def getModel():
    json_file = open(model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_weight)
    return loaded_model
    #loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

def predict(path):
    res1 = get_features(path)
    result = np.array(res1)
    pred_label = getModel().predict(result)
    print(type(pred_label))
    res = getEncoder().inverse_transform(pred_label)[0][0]
    print(res)
    return res

# path = "temp.wav"
# print(predict(path))
