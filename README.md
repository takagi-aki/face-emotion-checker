# 顔認証+表情認識モジュール

顔認証や表情認証等の機能をまとめたラッパーです。  
モデルデータは付属してません(別途DL)。  

## インストール

```
git clone https://github.com/takagi-aki/face-emotion-checker.git fec
```

## 利用モデル

必要に応じて手動でダウンロードしてください  

### FaceNet

[GitHub](https://github.com/davidsandberg/facenet)  
GitHubからオリジナルのものをダウンロードできる  
model/FaceNetフォルダ内に別途FaceNetモデル(20180402-114759.pb)をインストールする  

### Andrey Savchenko's Multi-task MobileNet-v1

[GitHub](https://github.com/HSE-asavchenko/face-emotion-recognition)  
GitHubからダウンロードできる  
model/HSE-asavchenko.face-emotion-recognitionフォルダ内に別途モデル(mobilenet_7.h5)をインストールする  
