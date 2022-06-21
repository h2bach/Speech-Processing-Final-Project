# Speech-Processing-Final-Project

## Nhận diện cảm xúc
Xây dựng mô hình nhận diện cảm xúc của người nói trong 1 đoạn âm thanh.
### Dữ liệu
- Tập dữ liệu RAVDESS: là một trong những tập dữ liệu phổ biến được sử dụng trong việc nhận dạng cảm xúc. Nó cũng được sử dụng nhiều vì chất lượng của loa, thu âm và nó có 24 diễn giả thuộc các giới tính khác nhau. Các xác định các thành phần từ tên File của tập dữ liệu Ravdess:
--Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
--Vocal channel (01 = speech, 02 = song).
--Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
--Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
--Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
--Repetition (01 = 1st repetition, 02 = 2nd repetition).
--Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
-- Ví dụ: 02-01-06-01-02-01-12.mp4 được cấu tạo bởi Video-only (02), Speech (01), Fearful (06), Normal intensity (01), Statement "dogs" (02), 1st Repetition (01), 12th Actor (12) - Female

- Tập dữ liệu Crema-D

- Tập dữ liệu Savee: Các tệp âm thanh được đặt tên theo cách mà các chữ cái tiền tố mô tả các lớp cảm xúc như sau:
-- 'a' = 'anger'
-- 'd' = 'disgust'
-- 'f' = 'fear'
-- 'h' = 'happiness'
-- 'n' = 'neutral'
-- 'sa' = 'sadness'
-- 'su' = 'surprise'
-- Ban đầu có 4 thư mục, mỗi thư mục đại diện cho một diễn giả, nhưng họ đã gộp tất cả chúng vào một thư mục duy nhất và do đó tiền tố 2 chữ cái đầu tiên của tên tệp đại diện cho tên viết tắt của người nói. Ví dụ. 'DC_d03.wav' là câu kinh tởm thứ 3 được cho ra bởi diễn giả DC.

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
