import cv2, math
import numpy as np

def detection_preprocessing(image: cv2.Mat) -> np.ndarray:
    inpWidth = 640
    inpHeight = 480

    # pre-process image
    blob = cv2.dnn.blobFromImage(
        image, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False
    )
    blob = np.transpose(blob, (0, 2, 3, 1))
    return blob

def detection_postprocessing(detection_response,preprocessed_images):
    def fourPointsTransform(frame, vertices):
        vertices = np.asarray(vertices)
        outputSize = (100, 32)
        targetVertices = np.array(
            [
                [0, outputSize[1] - 1],
                [0, 0],
                [outputSize[0] - 1, 0],
                [outputSize[0] - 1, outputSize[1] - 1],
            ],
            dtype="float32",
        )

        rotationMatrix = cv2.getPerspectiveTransform(vertices, targetVertices)
        result = cv2.warpPerspective(frame, rotationMatrix, outputSize)
        return result

    def decodeBoundingBoxes(scores, geometry, scoreThresh=0.5):
        detections = []
        confidences = []

        ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ########
        assert len(scores.shape) == 4, "Incorrect dimensions of scores"
        assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
        assert scores.shape[0] == 1, "Invalid dimensions of scores"
        assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
        assert scores.shape[1] == 1, "Invalid dimensions of scores"
        assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
        assert (
            scores.shape[2] == geometry.shape[2]
        ), "Invalid dimensions of scores and geometry"
        assert (
            scores.shape[3] == geometry.shape[3]
        ), "Invalid dimensions of scores and geometry"
        height = scores.shape[2]
        width = scores.shape[3]
        for y in range(0, height):
            # Extract data from scores
            scoresData = scores[0][0][y]
            x0_data = geometry[0][0][y]
            x1_data = geometry[0][1][y]
            x2_data = geometry[0][2][y]
            x3_data = geometry[0][3][y]
            anglesData = geometry[0][4][y]
            for x in range(0, width):
                score = scoresData[x]

                # If score is lower than threshold score, move to next x
                if score < scoreThresh:
                    continue

                # Calculate offset
                offsetX = x * 4.0
                offsetY = y * 4.0
                angle = anglesData[x]

                # Calculate cos and sin of angle
                cosA = math.cos(angle)
                sinA = math.sin(angle)
                h = x0_data[x] + x2_data[x]
                w = x1_data[x] + x3_data[x]

                # Calculate offset
                offset = [
                    offsetX + cosA * x1_data[x] + sinA * x2_data[x],
                    offsetY - sinA * x1_data[x] + cosA * x2_data[x],
                ]

                # Find points for rectangle
                p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
                p3 = (-cosA * w + offset[0], sinA * w + offset[1])
                center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
                detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
                confidences.append(float(score))

        # Return detections and confidences
        return [detections, confidences]

    # Process responses from detection model
    scores = detection_response.as_numpy("feature_fusion/Conv_7/Sigmoid:0")
    geometry = detection_response.as_numpy("feature_fusion/concat_3:0")
    cropped_images = []
    for i in range(preprocessed_images.shape[0]): # crop image one by one
        # matching dimension
        preprocessed_image = np.array([preprocessed_images[i]]) # (1, 480, 640, 3)
        scores_each = np.array([scores[i]]).transpose(0, 3, 1, 2)
        geometry_each = np.array([geometry[i]]).transpose(0, 3, 1, 2)

        frame = np.squeeze(preprocessed_image, axis=0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        [boxes, confidences] = decodeBoundingBoxes(scores_each, geometry_each)
        indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, 0.5, 0.4)

        cropped_list = []
        cv2.imwrite("../intermediates/frame.png", frame)
        count = 0
        for i in indices:
            # get 4 corners of the rotated rect
            count += 1
            vertices = cv2.boxPoints(boxes[i])
            cropped = fourPointsTransform(frame, vertices)
            cv2.imwrite("../intermediates/"+str(count) + ".png", cropped)
            cropped = np.expand_dims(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY), axis=0)
            cropped_list.append(((cropped / 255.0) - 0.5) * 2)
        if(len(cropped_list)>1):
            cropped_arr = np.stack(cropped_list, axis=0)
        else:
            cropped_arr = np.array(cropped_list)
        cropped_images.extend(cropped_arr)

    return cropped_images

def recognition_postprocessing(scores: np.ndarray) -> str:
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"

    text_list = []
    for i in range(scores.shape[0]):
        text = ""
        for j in range(scores.shape[1]):
            c = np.argmax(scores[i][j])
            if c != 0:
                text += alphabet[c - 1]
            else:
                text += "-"
        text_list.append(text)
    # adjacent same letters as well as background text must be removed
    # to get the final output
    final_text_list = []
    for text in text_list:
        char_list = []
        for i, char in enumerate(text):
            if char != "-" and (not (i > 0 and char == text[i - 1])):
                char_list.append(char)
        final_text = "".join(char_list)
        final_text_list.append(final_text)
    return final_text_list