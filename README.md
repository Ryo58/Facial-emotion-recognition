# Facial emotion recognition using Azure Cognitive Services

## 仮想環境の作成（初回のみ）
```
$ python3 -m venv .venv
```

## 仮想環境の起動
```
$ source .venv/bin/activate
```

## 仮想環境のpipのアップグレード
```
$ python3 -m pip install --upgrade pip
```

## 仮想環境に必要なライブラリのインストール
```
$ pip3 install -r requirements.txt
```
上手くいかない場合下記コマンドを参考にインストール
```
$ pip3 install numpy
$ pip3 install matplotlib
$ pip3 install opencv-python==3.4.17.61
$ pip3 install azure-cognitiveservices-vision-computervision
$ pip3 install azure-cognitiveservices-vision-face
$ pip3 install urllib3==1.26.15
```

## 仮想環境に入っているライブラリの確認
```
$ pip3 list
```

## Microsoft Azure Keyを下記ファイルに記載
myutils/const/const.py

## 実行
```
$ python3 main.py
```
Macbookの場合、ターミナルからカメラの利用許可がされていない場合があるので「システム設定->プライバシーとセキュリティ」から許可する
