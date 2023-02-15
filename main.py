import os
import cv2
import pytesseract
from matplotlib import pyplot as plt
import numpy as np
from imutils.object_detection import non_max_suppression


def loadData():
    imgs = []
    lines = []
    total_boxes = []
    with open("output.txt") as file_in:
        for line in file_in:
            lines.append(line.strip('\n').split(","))

    for filename in os.listdir('images'):
        img = cv2.imread(os.path.join('images', filename))
        if img is not None:
            imgs.append(img)

    for i in lines:
        if (i[0] == ''):
            total_boxes.append(0)
        else:
            total_boxes.append(len(i))

    return imgs, lines, total_boxes


def processImage(image):
    # Saving a original image and shape
    orig = image.copy()

    # new height and width for uniformity
    (newW, newH) = (320, 320)
    hRatio = image.shape[0] / float(newH)
    wRatio = image.shape[1] / float(newW)

    # resize image
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # construct blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (163.68, 166.78, 153.94), swapRB=True, crop=False)

    # load EAST model for detection
    network = cv2.dnn.readNet("frozen_east_text_detection.pb")

    # pull layers from east model to improve prediction accuracy
    layers = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # Forward pass network
    network.setInput(blob)
    (scores, geometry) = network.forward(layers)
    return scores, geometry, hRatio, wRatio


def predict(prob_score, g):
    (numR, numC) = prob_score.shape[2:4]
    boxes = []
    confidence = []

    for y in range(0, numR):
        scoresData = prob_score[0, 0, y]
        x0 = g[0, 0, y]
        x1 = g[0, 1, y]
        x2 = g[0, 2, y]
        x3 = g[0, 3, y]
        anglesData = g[0, 4, y]

        # loop over the number of columns and computes coordinated od bounding box
        for i in range(0, numC):
            if scoresData[i] < 0.5:
                continue

            (offX, offY) = (i * 4.0, y * 4.0)

            # extracting the rotation angle for the prediction
            angle = anglesData[i]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = x0[i] + x2[i]
            w = x1[i] + x3[i]

            eX = int(offX + (cos * x1[i]) + (sin * x2[i]))
            eY = int(offY - (sin * x1[i]) + (cos * x2[i]))
            sX = int(eX - w)
            sY = int(eY - h)

            boxes.append((sX, sY, eX, eY))
            confidence.append(scoresData[i])

    # return boxes and confidence
    return (boxes, confidence)


def displayResults(orig, boxes, wRatio, hRatio):
    results = []

    # iterate over each bounding box
    for (sX, sY, eX, eY) in boxes:
        # scale coordinates
        sX = int(sX * wRatio)
        sY = int(sY * hRatio)
        eX = int(eX * wRatio)
        eY = int(eY * hRatio)
        r = orig[sY:eY, sX:eX]

        # configuration to convert image to string.
        configuration = ("-l eng --oem 1 --psm 8")
       # recognizes text from bounding box

        pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
        text = pytesseract.image_to_string(r, config=configuration)

        # append bbox coordinate and associated text to the list of results
        results.append(((sX, sY, eX, eY), text))

    return results


def showOutput(result, orig_image):

    for ((start_X, start_Y, end_X, end_Y), text) in result:
        # display the text detected by Tesseract
        print("{}\n".format(text))

        # Displaying text
        text = "".join([x if ord(x) < 128 else "" for x in text]).strip()
        cv2.rectangle(orig_image, (start_X, start_Y), (end_X, end_Y),
                      (255, 0, 0), 2)
        cv2.putText(orig_image, text, (start_X, start_Y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    plt.imshow(orig_image)
    plt.title('Output')
    plt.show()


def calcAccuracy(results, lines, total_boxes):
    pred_boxes = 0
    correct_pred = 0
    # compare precicted and actual outputs and prints accuracy value
    for i in range(len(results)):
        pred_boxes += len(results[i])
        for pred in results[i]:
            for j in lines:
                text = pred[1].strip('\n').upper().replace(" ", "")
                if text in j:
                    correct_pred += 1
    print('Detection Accuracy: {:.2f}'.format(
        1 - abs(sum(total_boxes)-pred_boxes)/pred_boxes))
    print('Prediction Accuracy: {:.2f}'.format(correct_pred/sum(total_boxes)))


def main():
    imgs, lines, total_boxes = loadData()
    print(total_boxes)
    results = []
    for image in imgs:
        scores, box, hRatio, wRatio = processImage(image)
        orig = image.copy()
        # apply suppression
        (boxes, conf) = predict(scores, box)
        boxes = non_max_suppression(np.array(boxes), probs=conf)
        # Display the image with boxes and text
        orig_image = orig.copy()
        result = displayResults(orig, boxes, wRatio, hRatio)
        results.append(result)

        showOutput(result, orig_image)
    calcAccuracy(results, lines, total_boxes)


main()
