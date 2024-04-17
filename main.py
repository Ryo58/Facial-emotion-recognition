#!/usr/bin/env python
import os # ディレクトリ作成用のライブラリ
import traceback # エラー出力用のライブラリ
import numpy as np # 配列計算用のライブラリ
import matplotlib.pyplot as plt # グラフ描画用のライブラリ
import cv2 # 画像処理用のライブラリ
import threading # 並列処理用のライブラリ
from azure.cognitiveservices.vision.face import FaceClient # Microsoftのライブラリ
from msrest.authentication import CognitiveServicesCredentials # Microsoftのライブラリ
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person # Microsoftのライブラリ

from myutils.const.const import * # 設定が書いてあるファイル

# 年齢の初期値定義
age = 0
# 感情値の初期値定義
emotion_val = np.zeros(8)
emotion_label_en = np.array(['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise'])
emotion_label_ja = np.array(['怒り', '軽蔑', '嫌悪', '恐怖', '喜び', '平常', '悲しみ', '驚き'])
emotion_label_max = 'null'

# 感情値を更新する関数
def update_emotion():
    # グローバル変数の取得
    global age
    global emotion_val
    global emotion_label_ja
    global emotion_label_en
    global emotion_label_max
    try:
        # FaceClientの作成
        face_client = FaceClient(MICROSOFT_AZURE_ENDPOINT, CognitiveServicesCredentials(MICROSOFT_AZURE_KEY))
        # 画像を取得
        with open(CAMERA_SAVE_FILENAME, 'rb') as f:
            # FaceClientから顔検出
            detected_faces = face_client.face.detect_with_stream(f, return_face_id=True, return_face_attributes=['accessories','age','emotion','gender','glasses','hair','makeup','smile'], ecognition_model='recognition_01')
            # 顔検出が成功した場合
            if detected_faces:
                # 年齢の更新
                age = detected_faces[0].face_attributes.age
                # 感情値の更新
                emotion_val[0] = detected_faces[0].face_attributes.emotion.anger
                emotion_val[1] = detected_faces[0].face_attributes.emotion.contempt
                emotion_val[2] = detected_faces[0].face_attributes.emotion.disgust
                emotion_val[3] = detected_faces[0].face_attributes.emotion.fear
                emotion_val[4] = detected_faces[0].face_attributes.emotion.happiness
                emotion_val[5] = detected_faces[0].face_attributes.emotion.neutral
                emotion_val[6] = detected_faces[0].face_attributes.emotion.sadness
                emotion_val[7] = detected_faces[0].face_attributes.emotion.surprise
                # 感情値が最大の名前を取得
                emotion_label_max = emotion_label_en[np.argmax(emotion_val)]
                # 感情値をグラフ化して保存
                plt.bar(emotion_label_en, emotion_val * 100, color=EMOTION_GRAPH_COLOR, align='center')
                plt.savefig(EMOTION_GRAPH_SAVE_FILENAME)
                plt.close()
                # 感情値をコンソールに表示
                print('** Emotion ****************************************')
                print('%s %s: %.2f' %(emotion_label_en[0], emotion_label_ja[0], emotion_val[0]))
                print('%s %s: %.2f' %(emotion_label_en[1], emotion_label_ja[1], emotion_val[1]))
                print('%s %s: %.2f' %(emotion_label_en[2], emotion_label_ja[2], emotion_val[2]))
                print('%s %s: %.2f' %(emotion_label_en[3], emotion_label_ja[3], emotion_val[3]))
                print('%s %s: %.2f' %(emotion_label_en[4], emotion_label_ja[4], emotion_val[4]))
                print('%s %s: %.2f' %(emotion_label_en[5], emotion_label_ja[5], emotion_val[5]))
                print('%s %s: %.2f' %(emotion_label_en[6], emotion_label_ja[6], emotion_val[6]))
                print('%s %s: %.2f' %(emotion_label_en[7], emotion_label_ja[7], emotion_val[7]))
    except:
        print('** Error ******************************************')
        print(traceback.print_exc())
        plt.close()

# main関数
def main():

    # フレームの初期値定義
    t_frame = 0
    # ディレクトリの作成
    if not os.path.exists('output'):
        os.mkdir('output')
    # カメラを取得
    camera = cv2.VideoCapture(CAMERA_CHANNEL)
    camera_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    camera_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # カスケード検出器の特徴量を取得（顔検出用。顔検出はopencvのカスケード検出器というものを用いている）
    cascade = cv2.CascadeClassifier(CASCADE_CLASSIFIER_PASS)
    # 最初に表示するグラフの作成
    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    plt.subplots_adjust(bottom=0.42)
    plt.style.use('dark_background')
    plt.xticks(rotation=290)
    plt.ylim([0, 100])
    plt.bar(emotion_label_en, emotion_val * 100, color=EMOTION_GRAPH_COLOR, align='center')
    plt.savefig(EMOTION_GRAPH_SAVE_FILENAME)
    plt.close()

    # フレームを1枚ずつ取得（Trueの間ループ）
    while True:
        # フレームを更新
        t_frame += 1
        # カメラからフレームを取得
        ret, frame = camera.read()

        # 感情値の更新（現在のフレームが設定した値で割り切れる場合）
        if t_frame % GET_EMOTION_INTERVAL == 0:
            # フレームを画像として保存
            cv2.imwrite(CAMERA_SAVE_FILENAME, frame)
            # グラフの下書き作成
            plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
            plt.subplots_adjust(bottom=0.42)
            plt.style.use('dark_background')
            plt.xticks(rotation=290)
            plt.ylim([0, 100])
            # 並列処理でupdate_emotion関数を実行
            t2 = threading.Thread(target=update_emotion)
            t2.start()

        '''
        # 顔検出
        gry_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # グレースケール画像に変換
        facerect = cascade.detectMultiScale(gry_img, scaleFactor=1.2, minNeighbors=6, minSize=(200, 200)) # 顔検出の実行
        # 描画する枠の色を設定
        rectangle_color = (0, 255, 0)  # 緑色
        # 顔検出が成功した場合に描画
        if len(facerect) > 0:
            # 検出した顔の数の分ループ
            for rect in facerect:
                # カスケード検出器で検出した顔の枠線を描画
                cv2.rectangle(frame, tuple(rect[0:2]),tuple(rect[0:2] + rect[2:4]), rectangle_color, thickness=2)
        '''

        # フレームにグラフを合成
        emotion_graph = cv2.imread(EMOTION_GRAPH_SAVE_FILENAME)
        height = emotion_graph.shape[0]
        width = emotion_graph.shape[1]
        frame[camera_height-height-150:camera_height-150, 30:30+width] = emotion_graph
        # フレームに文字を描画
        cv2.putText(frame, 'Frame: ' + str(GET_EMOTION_INTERVAL - (t_frame % GET_EMOTION_INTERVAL)), (30, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, 'Age: ' + str(age), (30, camera_height-60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, 'Emotion: ' + emotion_label_max, (30, camera_height-30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3, cv2.LINE_AA)

        # フレームを画面に表示
        cv2.imshow('camera', frame)

        # キー操作があればwhileループを抜ける（qキーで撮影終了）
        if cv2.waitKey(CAMERA_WAIT_KEY) & 0xFF == ord('q'):
            break

    # 撮影用オブジェクトの解放
    camera.release()
    # 撮影用ウィンドウの解放
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
