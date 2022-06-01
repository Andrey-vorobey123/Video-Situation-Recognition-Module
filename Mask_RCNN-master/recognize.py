import cv2
import mrcnn.config
import os
from datetime import datetime
import mrcnn.visualize
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path
import click

def recognizing(path, flag, msk):
    # Configuration that will be used by the Mask-RCNN library
    import mrcnn.config
    class ObjectDetectorConfig(mrcnn.config.Config):
        NAME = "custom_object"
        IMAGES_PER_GPU = 1
        GPU_COUNT = 1
        DETECTION_MIN_CONFIDENCE = 0.8
        NUM_CLASSES = 81  # custom object + background class

    click.echo('\n\n\n----------------------------ВЫПОЛНЕНА КОНФИГУРАЦИЯ МОДЕЛИ ДЛЯ РАСПОЗНАВАНИЯ--------------------------------- \n\n\n')
    def mask_image(image, masks, ids, colors):
        # Loop over each detected object's mask
        for i in range(masks.shape[2]):
            # Draw the mask for the current object
            classID = ids[i]
            import numpy as np
            color_elem = colors[classID]
            color = [int(c) for c in np.array(color_elem) * 255]
            mask = masks[:, :, i]
            #color = (1.0, 0.0, 0.0) # Red
            image = mrcnn.visualize.apply_mask(image, mask, color, alpha=0.8)

        return image

    def contur_image(image, masks, boxes, class_ids, colors):
        import numpy as np
        bgr_image = image[:, :, ::-1]

        #import mrcnn.visualize

        CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        #COLORS = mrcnn.visualize.random_colors(len(CLASS_NAMES))

        for i in range(boxes.shape[0]):        
            
            y1, x1, y2, x2 = boxes[i]
            classID = class_ids[i]            
            label = CLASS_NAMES[classID]
            color_elem = colors[classID]
            color_mas = np.array(color_elem)
            font = cv2.FONT_HERSHEY_DUPLEX
            color = [int(c) for c in np.array(color_elem) * 255]
            text = label
            size = 0.4
            width = 1
            #mask = masks[:, :, i]  # берем срез
            #bgr_image = mrcnn.visualize.apply_mask(bgr_image, mask, color, alpha=0.6) # рисование маски
            #bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            cv2.rectangle(bgr_image, (x1, y1), (x2, y2), color, width)
            cv2.putText(bgr_image, text, (x1, y1-10), font, size, color, width)
            
        
        return bgr_image


    # Root directory of the project
    # ROOT_DIR = Path(".")
    MODEL_DIR = "training_logs"

    # Local path to trained weights file 
    TRAINED_MODEL_PATH = "mask_rcnn_coco.h5"
    if not os.path.exists(TRAINED_MODEL_PATH):
        mrcnn.utils.download_trained_weights(TRAINED_MODEL_PATH)

    # Video file to process 
    SOURCE_VIDEO_FILE = "C:/Progs/Clear_3/mask_rcnn-master/test_images/" + path

    # Create a Mask-RCNN model in inference mode
    model = MaskRCNN(mode="inference", model_dir="logs", config=ObjectDetectorConfig())
    click.echo('\n\n\n----------------------------СОЗДАНА МОДЕЛЬ ДЛЯ РАСПОЗНАВАНИЯ--------------------------------- \n\n\n')
    # Load pre-trained model
    model.load_weights(TRAINED_MODEL_PATH, by_name=True)

    # COCO Class names
    #class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    import mrcnn.visualize

    CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    cvetos = mrcnn.visualize.random_colors(len(CLASS_NAMES))
    # Load the video file we want to run detection on
    video_capture = cv2.VideoCapture(SOURCE_VIDEO_FILE)
    click.echo('\n\n\n----------------------------ОТКРЫТО ВИДЕО, НАЧАТ ПРОЦЕСС ЧТЕНИЯ КАДРОВ--------------------------------- \n\n\n')
    put = os.path.join(os.getcwd(), 'RECOGNIZED_VIDEOS')## + '\\' + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    #os.mkdir(put)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    PATH_TO_NEW_VID = put + "\\" + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + ".avi"
    w = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    out = cv2.VideoWriter(PATH_TO_NEW_VID, fourcc, video_capture.get(cv2.CAP_PROP_FPS), (int(w),int(h)))

    # NOTE:
    # This is just a simple demo to show how to process a video, not production quality.
    # You could make this run faster by processing more than one image at a time though the model.
    # To do that, you need to edit ObjectDetectorConfig to increase "Images per GPU" and then
    # pass in batches of frames instead of single frames. video_capture.get(cv2.CAP_PROP_FPS)
    # put + '\\' + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".avi"
    # However, that would require a GPU with more RAM so it may not work for you if you don't
    # have a high-end GPU with 12gb of RAM.
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # Convert the image from BGR color (which OpenCV uses) to RGB color
        rgb_image = frame[:, :, ::-1]

        # Run the image through the model
        results = model.detect([rgb_image], verbose=1)
        click.echo('\n----------------------------ОБЪЕКТЫ НА КАДРЕ РАСПОЗНАНЫ--------------------------------- \n')
        # Visualize results
        r = results[0]
        if flag: 
            print(r)
        
        if msk:
            rgb_image = mask_image(rgb_image, r['masks'], r['class_ids'], cvetos)

        rgb_image = contur_image(rgb_image, r['masks'], r['rois'], r['class_ids'], cvetos)

        # Convert the image back to BGR
        bgr_image = rgb_image#[:, :, ::-1]

        scale_percent = 140 # percent of original size
        width = int(bgr_image.shape[1] * scale_percent / 100)
        height = int(bgr_image.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(bgr_image, dim, interpolation = cv2.INTER_AREA)

        # cv2.namedWindow("main", cv2.WINDOW_NORMAL)
        cv2.imshow('Video', resized)
        #bgr_image = bgr_image[:, :, ::-1]
        out.write(bgr_image)
        #out.write(bgr_image.astype('uint8'))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    video_capture.release()
    cv2.destroyAllWindows()

    return PATH_TO_NEW_VID

import sys
import click

@click.group()
@click.version_option("1.0.0")
def main():
    
    print("\n\n\n")
    print("---------------------------------РАСПОЗНАВАНИЕ ОБЪЕКТОВ НА ВИДЕО----------------------------------\n\n\n")
    pass

@main.command()
@click.argument('name', required=True)
@click.option('--res', is_flag=True, help='Отображать доп. распознанную информацию по каждому кадру (rois, ids, scores, masks) [FLAG]')
@click.option('--mask', is_flag=True, help='Отображать цветную маску каждого распознанного образа в итоговом видео [FLAG]')
def main(**kwargs):
    """
    Приложение для распознавания объектов на видео.

    NAME - название ведеофайла ["<>.mp4"]
    """
    click.echo('\n\n\n----------------------------НАЧАТ ПРОЦЕСС РАСПОЗНАВАНИЯ ВИДЕО--------------------------------- \n\n\n')
    details = recognizing(kwargs.get("name"), kwargs.get("res"), kwargs.get("mask"))
    click.echo('\n\n\n-----------------------------ОБЪЕКТЫ НА ВИДЕО РАСПОЗНАНЫ--------------------------------- \n\n\n')
    click.echo(f'----------------------------РАСПОЗНАННОЕ ВИДЕО СОХРАНЕНО ПО АДРЕСУ--------------------------------\n\n{details}\n')
    #click.echo(f'Description \n\n{details["description"]}\n')
    #click.echo(f'References \n\n{details["references"]}\n')
    #click.echo(f'Assigning CNA \n\n{details["assigning cna"]}\n')
    #click.echo(f'Date Entry \n\n{details["date entry created"]}')

if __name__ == '__main__':
    args = sys.argv
    if "--help" in args or len(args) == 1:
        print("\n\nHELP\n\n")
    main()
