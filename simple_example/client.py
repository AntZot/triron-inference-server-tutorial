
import cv2
import tritonclient.http as httpclient
import numpy as np
from PIL import ImageDraw, Image

def detect_preprocessing(image):
    orig_img_size = image.shape[0:2]
    image = cv2.resize(image,(640,640))
    image = np.expand_dims(np.transpose(image, (2,0,1)).astype("float32"),axis=0) / 255.0

    return image, orig_img_size


def iou(box1,box2):
    return intersection(box1,box2)/union(box1,box2)

def union(box1,box2):
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1)
    box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1)
    return box1_area + box2_area - intersection(box1,box2)

def intersection(box1,box2):
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    x1 = max(box1_x1,box2_x1)
    y1 = max(box1_y1,box2_y1)
    x2 = min(box1_x2,box2_x2)
    y2 = min(box1_y2,box2_y2)
    return (x2-x1)*(y2-y1)


def detect_postprocessing(response,orig_img_size: tuple,probability=0.5):
    response = response.as_numpy("output0")
    output = response[0].transpose()
    boxes = []
    for row in output:
        prob = row[4:].max()
        if prob < probability:
            continue
        class_id = row[4:].argmax()
        label = class_id
        xc, yc, w, h = row[:4]
        x1 = (xc - w/2) / 640 * orig_img_size[0]
        y1 = (yc - h/2) / 640 * orig_img_size[1]
        x2 = (xc + w/2) / 640 * orig_img_size[0]
        y2 = (yc + h/2) / 640 * orig_img_size[1]
        boxes.append([x1, y1, x2, y2, label, prob])
    
    boxes.sort(key=lambda x: x[5], reverse=True)
    result = []
    while len(boxes) > 0:
        result.append(boxes[0])
        boxes = [box for box in boxes if iou(box, boxes[0]) < 0.7]
    return result


def main():
    client = httpclient.InferenceServerClient("localhost:8000")

    orig_image = cv2.imread("image.png")
    orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

    image, orig_img_size = detect_preprocessing(orig_image)

    #создание запроса
    input_img = httpclient.InferInput("images",image.shape,"FP32")

    #запись данных в запрос
    input_img.set_data_from_numpy(image,binary_data=True)

    #отправка и получение ответа
    response = client.infer("detection",[input_img])

    print(response.get_response())

    res = detect_postprocessing(response=response,orig_img_size=orig_img_size,probability=0.6)

    img = Image.fromarray(orig_image)
    draw = ImageDraw.Draw(img)
    for i in res:
        x1,y1,x2,y2,class_id,prob = i
        draw.rectangle((x1,y1,x2,y2),None,"#ff0000",width=3)
    
    img.save("result.jpg") 


if __name__ == "__main__":
    main()