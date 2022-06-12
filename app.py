from flask import Flask, request, redirect,render_template,json,url_for, session
from time import sleep
import os
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
from tensorflow import keras
import h5py
from keras.models import load_model
from os import path
from pydub import AudioSegment
from flask_cors import  CORS,cross_origin
from flask.helpers import send_from_directory

print('Hello')
app = Flask(__name__,static_folder='my-app/build',static_url_path='')
# ,static_folder='./my-app/build',static_url_path=''
def features_extractor(file):
    audio,sampling_rate=librosa.load(file,offset=0.5)
    mfccs_features=np.mean(librosa.feature.mfcc(y=audio,sr=sampling_rate,n_mfcc=40).T,axis=0)
    return mfccs_features
    
model = load_model('model_conv (1).h5')
cors = CORS(app)
# @app.route('/')
# def index():
#     return {
#         render_template('index.html')
#     }
@app.route('/upload',methods=['POST'])
@cross_origin()
def upload():
    file = request.files['file'] 
    filename = (file.filename)
    
    src = "transcript.mp3"
    dst = "test.wav"
    # newfile = 
    
    if filename.endswith('.wav'):
        file.save(f'test.wav')
    else:
        file.save(f'uploads/{filename}')
        sound = AudioSegment.from_mp3(f'uploads/{filename}')
        sound.export(dst, format="wav")
    ext_fil = features_extractor(dst)
    print(ext_fil.shape)
    ext_fil = np.expand_dims(ext_fil,axis = 0)
    ext_fil=np.expand_dims(ext_fil,-1)
    print(ext_fil.shape)
    predicted = model.predict(ext_fil)
    print(predicted)
    maxi = -1
    k = 0
    for num in predicted[0]:
        if num > maxi:
            maxi = num
            ind = k
        k = k+1
    print(ind) 
    print(maxi)
    lister = ["anger", "disgust", "fear", "happiness", "neutral", "pleasant", "surprise", "sadness"]
    
    # sleep(2)

    return {"message":lister[ind]}

@app.route('/')
@cross_origin()
def serve():
    # return "Hello World";
    return send_from_directory(app.static_folder,'index.html')

if __name__ == '__main__':
    app.run(debug=True)