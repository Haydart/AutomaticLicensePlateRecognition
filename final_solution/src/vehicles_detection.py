import cv2
import numpy as np


class VehiclesDetector:

    def __init__(self):
        self.classes = self.load_classes()

    def load_classes(self):
        classes = []
        with open("../yolo_suite/classes.txt", 'r') as file:
            for line in file:
                classes.append(line.strip())

        return classes

    def __get_output_layers(self, net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        return output_layers

    def detect_vehicles(self, image):
        image_width = image.shape[1]
        image_height = image.shape[0]
        scale = 0.00392

        net = cv2.dnn.readNet("../yolo_suite/yolov3.weights", "../yolo_suite/yolov3.cfg")
        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(self.__get_output_layers(net))
        class_ids = []
        confidences = []
        boxes = []
        confidence_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > confidence_threshold:
                    center_x = int(detection[0] * image_width)
                    center_y = int(detection[1] * image_height)
                    w = int(detection[2] * image_width)
                    h = int(detection[3] * image_height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
        print(indices)
        bands = []
        for i in indices:
            i = i[0]
            if class_ids[i] < len(self.classes):
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]

                y0 = round(y)
                y1 = round(y + h)
                x0 = round(x)
                x1 = round(x + w)
                bands.append((y0, y1, x0, x1))
                return image[y0:y1,x0:x1]
                break

        return bands
