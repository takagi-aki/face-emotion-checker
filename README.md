# 顔認証+表情認識モジュール

顔認証や表情認証等の機能をまとめたラッパーです。  
モデルデータは付属してません(別途DL)。  

## インストール

```
git clone https://github.com/takagi-aki/face-emotion-checker.git fec
```

## 利用モデル

必要に応じて手動でダウンロードしてください  

### OpenCV YuNet 顔検出

"https://github.com/ShiqiYu/libfacedetection.train/tree/master/tasks/task1/onnx"
からダウンロード
model/opencvフォルダ内にインストールする  

### OpenCV SFace 顔認証

https://drive.google.com/file/d/1ClK9WiB492c5OZFKveF3XiHCejoOxINW/view  
からダウンロード  
model/opencvフォルダ内にインストールする  

### Dlib 顔検出

"http://dlib.net/files/mmod_human_face_detector.dat.bz2"
からダウンロードして解凍  
model/dlibフォルダ内にインストールする  


### Dlib 特徴点検出

"http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
からダウンロードして解凍  
model/dlibフォルダ内に(shape_predictor_68_face_landmarks.dat)をインストールする  


### FaceNet 顔認証

[GitHub](https://github.com/davidsandberg/facenet)からオリジナルのものをダウンロードできる  
model/FaceNetフォルダ内に別途FaceNetモデル(20180402-114759.pb)をインストールする  

### Andrey Savchenko's Multi-task MobileNet-v1 表情認識

[GitHub](https://github.com/HSE-asavchenko/face-emotion-recognition)からダウンロードできる  
model/HSE-asavchenko.face-emotion-recognitionフォルダ内に別途モデル(mobilenet_7.h5)をインストールする  
