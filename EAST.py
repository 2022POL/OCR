from imutils.object_detection import non_max_suppression
from imutils.perspective import four_point_transform
from imutils.contours import sort_contours
import matplotlib.pyplot as plt
import imutils
import numpy as np
import requests
import pytesseract
import cv2


def plt_imshow(title='image', img=None, figsize=(8, 5)):
    plt.figure(figsize=figsize)

    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
            titles = []

            for i in range(len(img)):
                titles.append(title)

        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)

            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])

        plt.show()
    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()


def decode_predictions(scores, geometry):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < min_confidence:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return (rects, confidences)


layerNames = ["feature_fusion/Conv_7/Sigmoid",
              "feature_fusion/concat_3"]

# 사전에 훈련된 EAST text detector 모델 Load
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet("model/frozen_east_text_detection.pb") # 오류가 나,,,,,

width = 640
height = 640
min_confidence = 0.5
padding = 0.0

url = 'https://user-images.githubusercontent.com/69428232/149087561-4803b3e0-bcb4-4f9f-a597-c362db24ff9c.jpg'

image_nparray = np.asarray(bytearray(requests.get(url).content), dtype=np.uint8)
org_image = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)

plt_imshow("Original", org_image)

orig = org_image.copy()
(origH, origW) = org_image.shape[:2]

(newW, newH) = (width, height)
rW = origW / float(newW)
rH = origH / float(newH)

org_image = cv2.resize(org_image, (newW, newH))
(H, W) = org_image.shape[:2]

blob = cv2.dnn.blobFromImage(org_image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)

(rects, confidences) = decode_predictions(scores, geometry)
boxes = non_max_suppression(np.array(rects), probs=confidences)

for (startX, startY, endX, endY) in boxes:
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)

    dX = int((endX - startX) * padding)
    dY = int((endY - startY) * padding)

    startX = max(0, startX - dX)
    startY = max(0, startY - dY)
    endX = min(origW, endX + (dX * 2))
    endY = min(origH, endY + (dY * 2))

    # 영역 추출
    roi = orig[startY:endY, startX:endX]

    config = ("-l eng --psm 4")
    text = pytesseract.image_to_string(roi, config=config)

    results.append(((startX, startY, endX, endY), text))

results = sorted(results, key=lambda r: r[0][1])

output = orig.copy()

# 결과 출력
for ((startX, startY, endX, endY), text) in results:
    cv2.rectangle(output, (startX, startY), (endX, endY), (0, 255, 0), 5)
    cv2.putText(output, text, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 5)

plt_imshow("Text Detection", output, figsize=(16, 10))



"""

from imutils.object_detection import non_max_suppression
import numpy as np
import argparse #터미널에서 인수값을 받아 사용할 수 있게 하는 패키지
import time
import cv2

ap = argparse.ArgumentParser()
# 입력 이미지 경로
ap.add_argument("-i", "--a-1.png", type=str, help="path to input image")
# 텍스트 탐지 모델(사전에 만들어진 모델 호출) 경로
ap.add_argument("-east", "--east", type=str, help="path to input EAST text detector")
# 텍스트인지 결정하는 확률 임계 값 (기본값=0.5)
ap.add_argument("-c", "--min-confidence", type=float, default=0.5, help="minimum probability required to inspect a region")
# 크기 조정된 이미지 너비
ap.add_argument("-w", "--width", type=int, default=320, help="resized image width (should be multiple of 32")
# 크기 조정된 이미지 높이
ap.add_argument("-e", "--height", type=int, default=320, help="resized image height (should be multiple of 32")
args = vars(ap.parse_args())

# 이미지 로드 및 사이즈 조정
image = cv2.imread("a-1.png")

orig = image.copy() # 사진 복사
(H, W) = image.shape[:2] # 높이, 넓이 저장

(newW, newH) = (args["width"], args["height"]) # 입력받은 넓이와 높이를 할당
rW = W / float(newW) # 기존 이미지와의 비율을 할당
rH = H / float(newH)

image = cv2.resize(image, (newW, newH)) #이미지 resize
(H,W) = image.shape[:2] # 높이, 넓이 저장

layerNames = [
    "feature_fusion/Conv_7/Sigmoid", # 시그모이드 활성함수/텍스트를 포함하고 있는지 아닌지에 대한 확률
    "feature_fusion/concat_3" # 피쳐맵 출력/이미지의 기하학적 구조 -> 이미지에 텍스트박스 그리기
]
# EAST 모델 안에서 두가지 출력레이어 호출

print("[INFO] loading EAST text detector...")
# cv2를 이용하여 EAST의 신경망 구조 호출
net = cv2.dnn.readNet("model/frozen_east_text_detection.pb") # 여기서 자꾸 오류가 나,,,
# blobFromImages 을 통해 이미지 전처리 Net에 입력되는 데이터는 blob 형식으로 변경
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)

net.setInput(blob) # 전처리된 이미지를 EAST에 넣어 실행
(scores, geometry) = net.forward(layerNames) # scores와 geometry에 생성된 값들을 저장한다

# 피쳐맵 생성
(numRows, numCols) = scores.shape # 행과 열의 수를 두개의 리스트에 초기화
rects = [] # 텍스트가 있는 부분에 박스 좌표
confidences = [] # 각 텍스트 박스가 텍스트를 가지고 있을지에 대한 확률

for y in range(0, numRows):
    scoresData = scores[0,0,y]
    xData0 = geometry[0,0,y]
    xData1 = geometry[0,0,y]
    xData2 = geometry[0,0,y]
    xData3 = geometry[0,0,y]
    anglesData = geometry[0,4,y]

    for x in range (0, numCols):
        if scoresData[x] < args["min_confidence"]:
            continue
        (offsetX, offsetY) = (x * 4.0, y * 4.0) # 박스의 확률이 임계값을 넘을 경우 4를 곱해서 저장

        angle = anglesData[x]
        cos = np.cos(angle)
        sin = np.sin(angle)

        h = xData0[x] + xData2[x]
        w = xData1[x] + xData3[x]

        # 각도 계산
        endX = int(offsetX + (cos + xData1[x]) + (sin + xData2[x]))
        endY = int(offsetX - (sin + xData1[x]) + (cos + xData2[x]))
        startX = int(endX - w)
        startY = int(endY - h)

        # 경계 박스영역의 좌표를 구하고 할당
        rects.append((startX, startY, endX, endY))
        confidences.append(scoresData[x])

# NMS 적용 -> 중복 검출 피하고 연산량 줄임 주변의 픽셀과 비교했을 때 확률상 높은 것만 두는 방법

# 중복 박스 할당
boxes = non_max_suppression(np.array(rects), probs=confidences)

# 텍스트 박스 그리기
for (startX, startY, endX, endY) in boxes:
    startX = int(startX + rW)
    startY = int(startY + rH)
    endX = int(endX + rW)
    endY = int(endY + rH)

    cv2.rectangle(orig, (startX,startY), (endX,endY), (0, 255, 0, 2))

#이미지 출력
cv2.imshow("Text Detection", orig)
cv2.waitKey(0)
"""