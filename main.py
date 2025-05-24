import asyncio
import websockets
import binascii
from io import BytesIO
import cv2 as cv
import numpy as np
import logging

from concurrent.futures import ThreadPoolExecutor

from PIL import Image, UnidentifiedImageError

executor = ThreadPoolExecutor()
count_lock = asyncio.Lock()
count = 0

net = cv.dnn.readNetFromDarknet("Resources/yolov4-tiny.cfg", 
                                    "Resources/yolov4-tiny.weights")
layers_names = net.getLayerNames()
out_layers_indexes = net.getUnconnectedOutLayers()
out_layers = [layers_names[index - 1] for index in out_layers_indexes]

with open("Resources/coco.names.txt") as file:
    classes = file.read().split("\n")
    
#look_for = input("What we are looking for: ").split(',')
look_for = "traffic light"

list_look_for = []
for look in look_for:
    list_look_for.append(look.strip())
    
classes_to_look_for = list_look_for


def is_valid_image(image_bytes):
    imgnp = np.frombuffer(image_bytes, np.uint8)
    image = cv.imdecode(imgnp, cv.IMREAD_COLOR)
    return image is not None

async def handle_connection(websocket):
    global count
    loop = asyncio.get_event_loop()
    frame_count = 0
    PROCESS_EVERY_NTH_FRAME = 3  # ← обрабатывать каждый n-й кадр
    while True:
        try:
            message = await websocket.recv()
            frame_count += 1
            print(len(message))
            if len(message) > 5000:
                  if frame_count % PROCESS_EVERY_NTH_FRAME != 0:
                    continue
                  if is_valid_image(message):
                        #print(message)
                        
                        imgnp=np.array(bytearray(message),dtype=np.uint8)
                        im = cv.imdecode(imgnp,-1)
                        result_image, response_messages = apply_yolo_object_detection(im)
                        #im = await loop.run_in_executor(executor, 
                                                        #apply_yolo_object_detection, im)
                        apply_yolo_object_detection(im)
                        if response_messages and count > 0: #возможно убрать count
                            for msg in response_messages:
                                await websocket.send(msg)
                        else:
                            await websocket.send("Not found")

                        cv.imshow("Video Capture", result_image)
                        cv.waitKey(1)
                        # with open("image.jpg", "wb") as f:
                        #         f.write(message)
            await asyncio.sleep(0.001)
            print()
            async with count_lock:
                count = 0
            
        except websockets.exceptions.ConnectionClosed:
            break

import cv2 as cv
import numpy as np

def find_black_rectangle(image):
    """Функция для поиска черного прямоугольника в изображении"""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # Преобразование в градации серого
    _, thresh = cv.threshold(gray, 10, 255, cv.THRESH_BINARY_INV)  # Применение пороговой обработки
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # Поиск контуров
    
    black_boxes = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if w * h > 500:  # Фильтрация по размеру
            black_boxes.append((x, y, w, h))
    
    return black_boxes

def apply_yolo_object_detection(image):
    """Основная функция с добавлением поиска черного прямоугольника"""
    global count
    height, width = image.shape[:2]

    black_boxes = find_black_rectangle(image)  # Поиск черного прямоугольника

    blob = cv.dnn.blobFromImage(image, 1 / 255.0, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_layers)

    class_indexes = []
    class_scores = []
    boxes = []

    for detection in np.vstack(outs):  
        scores = detection[5:]
        class_index = np.argmax(scores)
        confidence = scores[class_index]

        if confidence > 0:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = center_x - w // 2
            y = center_y - h // 2

            boxes.append([x, y, w, h])
            class_indexes.append(class_index)
            class_scores.append(float(confidence))

    if not boxes:
        return image, [] 

    indices = cv.dnn.NMSBoxes(boxes, class_scores, score_threshold=0.0, nms_threshold=0.4)
    if len(indices) == 0:
        return image, []

    result = image
    count = 0
    messages = []

    for i in indices.flatten():
        class_id = class_indexes[i]
        if classes[class_id] in classes_to_look_for:
            count += 1
            x, y, w, h = boxes[i]
            result = draw_object_bounding_box(result, class_id, boxes[i])
            
            roi = image[max(y, 0):y + h, max(x, 0):x + w]
            color_result = detect_color_red_or_green(roi)

            if color_result and color_result not in messages:
                messages.append(color_result)

    # Отрисовка найденных черных прямоугольников
    for x, y, w, h in black_boxes:
        cv.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return draw_object_count(result, count), messages


def detect_color_red_or_green(roi):
    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

    # Маска красного (двойной диапазон)
    red_mask1 = cv.inRange(hsv, (0, 70, 50), (10, 255, 255))
    red_mask2 = cv.inRange(hsv, (160, 70, 50), (180, 255, 255))
    red_mask = cv.bitwise_or(red_mask1, red_mask2)

    # Маска зелёного
    green_mask = cv.inRange(hsv, (40, 70, 50), (90, 255, 255))

    # Подсчёт количества пикселей каждого цвета
    red_pixels = cv.countNonZero(red_mask)
    green_pixels = cv.countNonZero(green_mask)

    # Определение средней яркости каждого цвета
    red_brightness = np.mean(hsv[:, :, 2][red_mask > 0]) if red_pixels > 0 else 0
    green_brightness = np.mean(hsv[:, :, 2][green_mask > 0]) if green_pixels > 0 else 0

    if red_pixels > 50 and green_pixels > 50:
        # Если оба цвета есть, но красный ярче - выбираем красный
        if red_brightness > green_brightness:
            return "found red"
        else:
            return "found green"
    elif red_pixels > 50:
        return "found red"
    elif green_pixels > 50:
        return "found green"
    else:
        return None

def draw_object_bounding_box(image_to_process, index, box):
    x, y, w, h = box
    start = (x, y)
    end = (x + w, y + h)
    color = (0, 255, 0)
    width = 2
    final_image = cv.rectangle(image_to_process, start, end, color, width)

    start = (x, y - 10)
    font_size = 1
    font = cv.FONT_HERSHEY_SIMPLEX
    width = 2
    text = classes[index]
    final_image = cv.putText(final_image, text, start, font, font_size, color, width, cv.LINE_AA)

    return final_image

def draw_object_count(image_to_process, objects_count):
    start = (10, 120)
    font_size = 1.5
    font = cv.FONT_HERSHEY_SIMPLEX
    width = 3
    text = "Objects found: " + str(objects_count)

    white_color = (255, 255, 255)
    black_outline_color = (0, 0, 0)
    final_image = cv.putText(image_to_process, text, start, font, font_size, black_outline_color, width * 3, cv.LINE_AA)
    final_image = cv.putText(final_image, text, start, font, font_size, white_color, width, cv.LINE_AA)

    return final_image

# TODO: оно есть, трогать не надо
# def start_video_object_detection(video: str):
#     logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
#                         level=logging.DEBUG)
#     while True:
#         try:
#             video_camera_capture = cv.VideoCapture(video)

#             while video_camera_capture.isOpened():
                
#                 imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
#                 im = cv.imdecode(message,-1)  
#                 ret, frame = video_camera_capture.read()
#                 #if not ret:
#                     #break

#                 apply_yolo_object_detection(im)

#                 #frame = cv.resize(frame, (1920 // 2, 1080 // 2))
#                 cv.imshow("Video Capture", im)
#                 cv.waitKey(1)
#                 key = cv.waitKey(5)
#                 if key == ord('q'):
#                     break

#             video_camera_capture.release()
#             cv.destroyAllWindows()
        
#         except KeyboardInterrupt:
#             pass

# if __name__ == '__main__':
#     #logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
#                         #level=logging.DEBUG)
#     net = cv.dnn.readNetFromDarknet("Resources/yolov4-tiny.cfg", 
#                                     "Resources/yolov4-tiny.weights")
#     layers_names = net.getLayerNames()
#     out_layers_indexes = net.getUnconnectedOutLayers()
#     out_layers = [layers_names[index - 1] for index in out_layers_indexes]

#     with open("Resources/coco.names.txt") as file:
#         classes = file.read().split("\n")

#     #video = url 
#     look_for = input("What we are looking for: ").split(',')

#     list_look_for = []
#     for look in look_for:
#         list_look_for.append(look.strip())
    
#     classes_to_look_for = list_look_for

#     #start_video_object_detection(video)

async def main():
    server = await websockets.serve(handle_connection, '0.0.0.0', 3001)
    await server.wait_closed()

asyncio.run(main())