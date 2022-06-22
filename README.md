# Speech-Processing-Final-Project
## Web phân loại âm thanh:
- Chức năng: Phân loại giới tính, cảm xúc của người nói từ file âm thanh đầu vào.
- Demo: https://drive.google.com/file/d/16SZ2OGnNt1yizQVkja4vqAsxds7fx8uq/view?usp=sharing
## Nhận diện cảm xúc
Xây dựng mô hình nhận diện cảm xúc của người nói trong 1 đoạn âm thanh.
### Dữ liệu
- Tập dữ liệu RAVDESS: là một trong những tập dữ liệu phổ biến được sử dụng trong việc nhận dạng cảm xúc. Nó cũng được sử dụng nhiều vì chất lượng của loa, thu âm và nó có 24 diễn giả thuộc các giới tính khác nhau. Các xác định các thành phần từ tên File của tập dữ liệu Ravdess:
- - Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
- - Vocal channel (01 = speech, 02 = song).
- - Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
- - Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
- - Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
- - Repetition (01 = 1st repetition, 02 = 2nd repetition).
- - Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
- - Ví dụ: 02-01-06-01-02-01-12.mp4 được cấu tạo bởi Video-only (02), Speech (01), Fearful (06), Normal intensity (01), Statement "dogs" (02), 1st Repetition (01), 12th Actor (12) - Female

- Tập dữ liệu Crema-D: CREMA-D là tập dữ liệu gồm 7.442 clip gốc từ 91 diễn viên. Các clip này do 48 nam và 43 diễn viên nữ trong độ tuổi từ 20 đến 74 đến từ nhiều chủng tộc và sắc tộc. Các diễn viên đã nói từ tuyển chọn 12 câu. Các câu được trình bày bằng một trong sáu cảm xúc khác nhau (Anger, Disgust, Fear, Happy, Neutral, and Sad) và bốn mức độ cảm xúc khác nhau (Thấp, Trung bình, Cao và Không xác định).

- Tập dữ liệu Savee: Các tệp âm thanh được đặt tên theo cách mà các chữ cái tiền tố mô tả các lớp cảm xúc như sau:
- - 'a' = 'anger'
- - 'd' = 'disgust'
- - 'f' = 'fear'
- - 'h' = 'happiness'
- - 'n' = 'neutral'
- - 'sa' = 'sadness'
- - 'su' = 'surprise'
- - Ban đầu có 4 thư mục, mỗi thư mục đại diện cho một diễn giả, nhưng họ đã gộp tất cả chúng vào một thư mục duy nhất và do đó tiền tố 2 chữ cái đầu tiên của tên tệp đại diện cho tên viết tắt của người nói. Ví dụ. 'DC_d03.wav' là câu kinh tởm thứ 3 được cho ra bởi diễn giả DC.

- Tập dữ liệu Tess: Đối với tập dữ liệu TESS, nó chỉ dựa trên 2 diễn giả, một phụ nữ trẻ và một phụ nữ lớn tuổi. Điều này hy vọng sẽ cân bằng các diễn giả nam nổi trội mà chúng ta có trên SAVEE. Nó có cùng 7 cảm xúc chính mà chúng ta quan tâm.

### Chuẩn bị dữ liệu
- Từ các định dạng cấu trúc tên file, tách ra các cảm xúc của file âm thanh và dùng 1 DataFrame để lưu trữ cảm xúc và đường dẫn đến các file âm thanh có cảm xúc đó. Sau cùng, gộp các DataFrame của 4 tập dữ liệu thành 1 bộ dữ liệu gồm các cảm xúc và đường dẫn file.
- Tăng thêm lượng dữ liệu bằng cách: Chèn nhiễu, kéo giãn âm thanh, ...

### Trích xuất đặc trưng
- Sử dụng các hàm có trong thư viện để trích xuất ra các đặc trưng cơ bản
--  Số lần tín hiệu đi từ + sang - hoặc từ - sang +: Zero Crossing Rate (ZCR)
-- Sắc độ của âm thanh (Chroma)
-- Mel frequency cepstral coefficients (MFCCs)
-- Giá trị Root Mean Square (RMS Value)
-- Mel-Spectrogram

### Xây dựng mô hình
- Chia tập huấn luyện/ kiểm thử
- Xây dựng mô hình MLP 
- Huấn luyện và dự đoán mẫu trong tập kiểm thử


## Nhận diện giới tính
### 1. Tiền xử lý dữ liệu
#### 1.1 Bộ dữ liệu sử dụng 
##### 1.1.1 Thông tin chung
 - Mô hình sử dụng bộ dữ liệu Common Voice. 
  Đây là kho dữ liệu giọng nói được người dùng thực hiện ở trên trang web Common Voice,
  và từ các nguồn dữ liệu công khai khác.
  Mục đích chính của bộ dữ liệu là cho phép 
  đào tạo và thử nghiệm các hệ thống nhận dạng giọng nói tự động (ASR).
- Bộ dữ liệu được đăng tải trên Kaggle: 
  https://www.kaggle.com/datasets/mozillaorg/common-voice

##### 1.1.2 Cấu trúc bộ dữ liệu
- Bộ dữ liệu được chia 3 tập dữ liệu con:
    - Tập dữ liệu "valid" chứa các đoạn âm thanh, có ít nhất hai người nghe trở lên và phần lớn những người nghe cho rằng đoạn âm thanh khớp với văn bản
    - Tập dữ liệu "invalid" là những đoạn âm thanh có ít nhất 2 người nghe và phần lớn cho rằng âm thanh không khớp với văn bản
    - Tập dữ liệu "other" là những đoạn âm thanh có tỉ lệ tán thành và phản đối về việc âm thanh khớp với văn bản là như nhau
    
- Mỗi tập dữ con được tách thành các tập nhỏ hơn: "train", "development", "test"
- Tập dữ liệu sẽ bao gồm các trường:
    - filename
    - text: phiên âm được cho là của âm thanh
    - up_votes: số người cho biết âm thanh khớp với văn bản
    - down_votes: số người cho biết âm thanh không khớp với văn bản
    - age: tuổi của người nói
        - teens: '< 19'
        - twenties: '19 - 29'
        - thirties: '30 - 39'
        - fourties: '40 - 49'
        - fifties: '50 - 59'
        - sixties: '60 - 69'
        - seventies: '70 - 79'
        - eighties: '80 - 89'
        - nineties: '> 89'
    - gender: giới tính người nói
        - male
        - female
        - other
    - accent: người nói là người ở đâu
    
- Vì đây là bài toán phân lớp gender nên ta chỉ cần quan tâm đến các trường là filename, text và gender


#### 1.2 Phân tích dữ liệu
##### Trực quan hóa dữ liệu
Ở đây, ta sẽ sử dụng tập dữ liệu train trong valid để thực hiện huấn luyện mô hình và tập dữ liệu development trong valid để test độ chính xác của mô hình

Phân tích chung dữ liệu ở tập train:

```python
    valid_train_df = pd.read_csv("../input/common-voice/cv-valid-train.csv")
    valid_train_df.count()
``` 
Kết quả trả về:
```python
    filename      195776
    text          195776
    up_votes      195776
    down_votes    195776
    age            73768
    gender         74059
    accent         64711
    duration           0
    dtype: int64
```

Ta có thể thấy có 195 776 file âm thanh. Tuy nhiên ở trường gender chỉ có 73 768 file. Để phục vụ cho mô hình ta chỉ sử dụng các file đã được gán gender
```python
valid_train_filter_df = valid_train_df[valid_train_df["gender"].notnull()]
```

Thêm một trường target_gender để huyển các giá trị ở trường gender, với 0 là female và 1 là male
```python
valid_train_filter_df["target_gender"] = valid_train_filter_df['gender'].apply(lambda gender: 0 if str(gender) == "female" else 1)
```

Trực quan hóa số lượng, và sự phân phối gender trong tập huấn luyện

```python
# Biểu đồ 
plt.subplot(1, 2, 1)
valid_train_filter_df.groupby('target_gender')['filename'].count().plot.bar()
plt.grid(True)
plt.title('Gender Count')
plt.subplots_adjust(right=1.9)

# Biểu đồ phân phối 
plt.subplot(1, 2, 2)
values = [valid_train_filter_df[valid_train_filter_df['target_gender']==0].shape[0], valid_train_filter_df[valid_train_filter_df['target_gender']==1].shape[0]]
labels = ['Female', 'Male']

plt.pie(values, labels=labels, autopct='%1.1f%%', shadow=True)
plt.title('Gender Distribution')
plt.tight_layout()
plt.subplots_adjust(right=1.9)

plt.show()

```
<img width="720" alt="Screen Shot 2022-06-21 at 15 38 38" src="https://user-images.githubusercontent.com/63334287/174860125-e5e1a3bd-057b-4b18-a69f-6575ceed1790.png">

Vì các file âm thanh có định dạng mp3 và để dữ liệu huấn luyện có sự cân bằng ta sẽ trích 10000 file âm thanh với tỉ lệ female:male là 50:50 và chuyển file âm thanh sang dạng file wav

##### Chuyển đổi file .mp3 sang .wav
```python
import subprocess
import timeit
def convert_to_wav(dict_list_audio):
    start = timeit.default_timer()
    for index, audio in enumerate(dict_list_audio.items()):
        path_audio = os.path.join(dir_path,audio[0].split("/")[0],audio[0])
        wav_file = audio[0].replace("mp3", "wav")
        new_path_audio = os.path.join("./cv-valid-train-2", wav_file.split("/")[1])
        subprocess.call(['ffmpeg', '-loglevel', 'panic', '-i',  path_audio, 
                '-acodec', 'pcm_s16le', '-ac', '1', '-ar', "22050", new_path_audio])
        if (index+1)%1000 == 0 or index == len(dict_list_audio)-1:
            stop = timeit.default_timer()
            print('{}/{}: {}, time: {}'.format(index+1, len(dict_list_audio), new_path_audio, stop-start))
            start = timeit.default_timer()
```



### 2. Mô hình sử dụng
Trong bài toán phân lớp giới tính, nhóm em sẽ sử dụng mô hình Resnet

#### 2.1 Giới thiệu mô hình
ResNet (viết tắt của residual network), là mạng học sâu nhận được quan tâm từ những năm 2012 sau cuộc thi LSVRC2012 và trở nên phổ biến trong lĩnh vực thị giác máy. ResNet khiến cho việc huấn luyện hàng trăm thậm chí hàng nghìn lớp của mạng nơ ron trở nên khả thi và hiệu quả. 

Ý tưởng chính của ResNet là sử dụng kết nối tắt đồng nhất để xuyên qua một hay nhiều lớp. Một khối như vậy được gọi là một residual block như trong hình sau:
![resnet-model](https://neurohive.io/wp-content/uploads/2019/01/resnet-e1548261477164.png
)
Việc xếp chồng các lớp sẽ không làm giảm hiệu suất mạng. Chúng ta có thể đơn giản xếp chồng các ánh xạ đồng nhất lên mạng hiện tại và hiệu quả của kiến trúc không thay đổi. Điều này giúp cho kiến trúc sâu ít nhất là không kém hơn các kiến trúc nông. Hơn nữa, với kiến trúc này, các lớp ở phía trên có được thông tin trực tiếp hơn từ các lớp dưới nên sẽ điều chỉnh trọng số hiệu quả hơn.

ResNet có nhiều biến thể, cụ thể là ResNet16, ResNet18, ResNet34, ResNet50, ResNet101, ResNet110, ResNet152, ResNet164, ResNet1202, v.v.
![resnet-model-bien-the](https://pytorch.org/assets/images/resnet.png)    

#### 2.2 Kiến trúc mô hình

<img width="850" alt="Screen Shot 2022-06-22 at 00 22 34" src="https://user-images.githubusercontent.com/63334287/174860321-ba1ce955-af5d-40ed-abc1-03a57463c2b0.png">

##### 2.2.1 Trích xuất đặc trưng
Spectrograms được tạo ra từ tín hiệu âm thanh bằng cách sử dụng Fourier Transforms. Biến đổi Fourier phân tách tín hiệu thành các tần số cấu thành của nó và hiển thị biên độ của mỗi tần số có trong tín hiệu.

Spectrogram hiển thị biểu đồ Tần số (trục y) với Thời gian (trục x) và sử dụng các màu khác nhau để biểu thị Biên độ của mỗi tần số. Màu càng sáng thì năng lượng của tín hiệu càng cao.

Tuy nhiên spectrogram không mang lại nhiều thông tin. Vì vậy ta sẽ sử dụng mel spectrogram

Biểu đồ Mel spectrogram thực hiện hai thay đổi quan trọng so với biểu đồ spectrogram thông thường hiển thị biểu đồ Tần suất và Thời gian.

- Nó sử dụng Thang đo Mel thay vì Tần số trên trục y.
- Nó sử dụng Thang đo Decibel thay vì Biên độ để chỉ ra màu sắc.

Trích xuất mel-spectrogram và lưu vào file .pkl
```python
def extract_melspectrogram(file_path, sr=22050, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):
    wav,sr = librosa.load(file_path, sr= 22050)
    spec=librosa.feature.melspectrogram(y=wav, sr=sr)
    print(spec)
    print(np.shape(spec))
    spec_db=librosa.power_to_db(spec,top_db=top_db,ref=np.max)
    return wav, spec, spec_db

def extract_features(dict_list_audio):
    specs = []
    for index, audio in enumerate(dict_list_audio.items()):
        path_audio = os.path.join(NEW_DIR_PATH, audio[0].split("/")[3])
        wav, spec, spec_db = extract_melspectrogram(path_audio)
        eps = 1e-6
        spec = np.log(spec+ eps)
        spec = np.asarray(torchvision.transforms.Resize((128, 1000))(Image.fromarray(spec)))
        new_entry = {}
        new_entry["values"] = np.array(spec)
        new_entry["target"] = audio[1]
        specs.append(new_entry)
        
        if (index+1)%100 ==0:
            print("{}/{}: {}".format(index, len(dict_list_audio), audio))
    return specs

training_values = extract_features(dict_audio_wav)

print("Start export to pkl")
import pickle as pkl 
with open("./trainingMel1.pkl","wb") as handler:
    pkl.dump(training_values, handler, protocol=pkl.HIGHEST_PROTOCOL)
print("Save pkl done")
```

##### 2.2.2 Xây dựng Resnet
Ở đây, ta sử dụng Resnet18
```python
class ResNet(nn.Module):
    def __init__(self, dataset, pretrained=False):
        super(ResNet, self).__init__()
        num_classes = 2
        self.model = models.resnet18(pretrained=pretrained)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        output = self.model(x)
        return output

```

### 3. Kết quả
Kết quả huấn luyện sau 10 epoch, batch size = 64. Sau mỗi epoch thực hiện đo độ chính xác trên tập dev. Kết quả dự đoán cao nhất trong tập test là 97%

<img width="549" alt="Screen Shot 2022-06-21 at 17 07 06" src="https://user-images.githubusercontent.com/63334287/174859794-25d7a10d-d763-4078-8d1c-c897e202212f.png">
