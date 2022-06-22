import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.models as models

import librosa
import numpy as np
from PIL import Image

class ResNet(nn.Module):
    def __init__(self, dataset, pretrained=True):
        super(ResNet, self).__init__()
        num_classes = 2
        self.model = models.resnet18(pretrained=pretrained)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        output = self.model(x)
        return output

def getModel(): 

    #global model 

    # if model is not None:
    #     return model

    model = ResNet('abc')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

    PATH = "model/gender/model_checkpoint/model_best_0.pth.tar"
    checkpoint = torch.load(PATH,map_location ='cpu')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']

def extract_melspectrogram(file_path, sr=22050, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):
    wav,sr = librosa.load(file_path, sr= 22050)
    spec=librosa.feature.melspectrogram(y=wav, sr=sr)
    eps = 1e-6
    spec = np.log(spec+ eps)
    spec = np.asarray(torchvision.transforms.Resize((128, 1000))(Image.fromarray(spec)))
    list_abc = []
    list_abc.append(np.array(spec))
    list_abc.append(np.array(spec))
    list_abc.append(np.array(spec))
    values = np.array(list_abc).reshape(-1, 128, 1000)
    values = torch.Tensor(values)

    return values

PATH_AUDIO = "test_sample/data_validation_Actor_10_03-02-04-02-02-02-10_sad.wav"
# spec = extract_melspectrogram(PATH_AUDIO)
# print(spec.shape)
# with torch.no_grad():
#     inputs = spec
#     inputs = inputs.unsqueeze(0)

#     model = getModel()
#     outputs = model(inputs)
#     _, predicted = torch.max(outputs.data, 1)
#     print(predicted)
#     print(predicted.numpy()[0])
#     print ('male' if predicted.numpy()[0] == 1 else 'female')

def predict(audio_file_path):
    model = getModel()
    spec = extract_melspectrogram(audio_file_path) 
    with torch.no_grad():
        inputs = spec
        inputs = inputs.unsqueeze(0)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        # print(predicted)
        # print(predicted.numpy()[0])
        # print ('male' if predicted.numpy()[0] == 1 else 'female')
        return 'male' if predicted.numpy()[0] == 1 else 'female'

#print(predict(PATH_AUDIO, getModel()))