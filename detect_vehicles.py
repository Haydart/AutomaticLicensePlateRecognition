import cv2
import numpy as np

from utils import load_image

classes = None
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))


def _get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def _draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = '{} {}'.format(str(classes[class_id]), confidence)
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def detect_vehicles(image_path):
    image = load_image(image_path)
    image_width = image.shape[1]
    image_height = image.shape[0]
    scale = 0.00392

    with open("yolo_suite/detection_classes", 'r') as classes_file:
        classes = [line.strip() for line in classes_file.readlines()]

    net = cv2.dnn.readNet("yolo_suite/yolov3.weights", "yolo_suite/yolov3.cfg")
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(_get_output_layers(net))
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

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        _draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

    cv2.imshow("object detection", image)
    cv2.waitKey()

    cv2.imwrite("object-detection.jpg", image)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect_vehicles("dataset/track001.png")
>> >> >> > Stashed
changes
