import asyncio
import websockets
import binascii
from io import BytesIO
import cv2 as cv
import numpy as np
import logging
from art import tprint
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, UnidentifiedImageError

executor = ThreadPoolExecutor()
count = 0

def is_valid_image(image_bytes):
    imgnp = np.frombuffer(image_bytes, np.uint8)
    image = cv.imdecode(imgnp, cv.IMREAD_COLOR)
    return image is not None

async def handle_connection(websocket):
    loop = asyncio.get_event_loop()
    frame_count = 0
    PROCESS_EVERY_NTH_FRAME = 2  # ← обрабатывать каждый n-й кадр
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
                        #processed_image, object_count = apply_yolo_object_detection(im)
                        #im = await loop.run_in_executor(executor, 
                                                        #apply_yolo_object_detection, im)
                        apply_yolo_object_detection(im)
                        if count > 0:
                            await websocket.send("Detected")
                        else:
                            await websocket.send("Not Detected")
                        cv.imshow("Video Capture", im)
                        cv.waitKey(1)
                        # with open("image.jpg", "wb") as f:
                        #         f.write(message)
            #await asyncio.sleep(0.001)
            print()
        except websockets.exceptions.ConnectionClosed:
            break

def apply_yolo_object_detection(image):
    global count
    height, width = image.shape[:2]

    blob = cv.dnn.blobFromImage(image, 1 / 255.0, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_layers)

    class_indexes = []
    class_scores = []
    boxes = []

    for detection in np.vstack(outs):  # объединение всех слоёв в один массив
        scores = detection[5:]
        class_index = np.argmax(scores)
        confidence = scores[class_index]

        if confidence > 0:  # при необходимости подними порог
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
        return image  # Быстрый выход

    indices = cv.dnn.NMSBoxes(boxes, class_scores, score_threshold=0.0, nms_threshold=0.4)
    if len(indices) == 0:
        return image

    result = image
    count = 0

    for i in indices.flatten():
        class_id = class_indexes[i]
        if classes[class_id] in classes_to_look_for:
            count += 1
            result = draw_object_bounding_box(result, class_id, boxes[i])
            
        # if classes[class_index] not in classes_to_look_for:
        #     continue

    return draw_object_count(result, count)

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

if __name__ == '__main__':
    #logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                        #level=logging.DEBUG)
    net = cv.dnn.readNetFromDarknet("Resources/yolov4-tiny.cfg", 
                                    "Resources/yolov4-tiny.weights")
    layers_names = net.getLayerNames()
    out_layers_indexes = net.getUnconnectedOutLayers()
    out_layers = [layers_names[index - 1] for index in out_layers_indexes]

    with open("Resources/coco.names.txt") as file:
        classes = file.read().split("\n")

    #video = url 
    look_for = input("What we are looking for: ").split(',')

    list_look_for = []
    for look in look_for:
        list_look_for.append(look.strip())
    
    classes_to_look_for = list_look_for

    #start_video_object_detection(video)

async def main():
    server = await websockets.serve(handle_connection, '0.0.0.0', 3001)
    await server.wait_closed()

asyncio.run(main())