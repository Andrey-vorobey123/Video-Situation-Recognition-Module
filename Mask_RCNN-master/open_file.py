from mimetypes import init
from tkinter import Frame, Label, Tk, BOTH, Text, Menu, END, BOTH, X, N, LEFT
from tkinter import filedialog
import tkinter.font as tkFont
import cv2
import mrcnn.config
import os
from datetime import datetime
import mrcnn.visualize
import mrcnn.utils
from mrcnn.model import MaskRCNN
from recognize import recognizing, ObjectDetectorConfig

 
class Example(Frame):
 
    def __init__(self):
        super().__init__()
        self.initUI()
 
    def initUI(self):
        self.master.title("Окно для выбора файла")
        self.pack(fill=BOTH, expand=1)
 
        menubar = Menu(self.master)
        self.master.config(menu=menubar)
 
        fileMenu = Menu(menubar)
        fileMenu.add_command(label="Открыть", command=self.onOpen)
        menubar.add_cascade(label="Файл", menu=fileMenu)
 
        #self.txt = Text(self)
        #self.txt.pack(fill=BOTH, expand=1)
 
    

    def onOpen(self):
        ftypes = [('Videos', '*.mp4'), ('Все файлы', '*')]
        dlg = filedialog.Open(self, filetypes = ftypes)
        fl = dlg.show()
 
        if fl != '':
            #text = self.readFile(fl)
            import mrcnn.config
            class ObjectDetectorConfig(mrcnn.config.Config):
                NAME = "custom_object"
                IMAGES_PER_GPU = 1
                GPU_COUNT = 1
                DETECTION_MIN_CONFIDENCE = 0.8
                NUM_CLASSES = 81  # custom object + background class

            #Label_1.pack()
            Label_1 = Label(self.master, text='Сконфигурированы настройки модели')
            fontExample = tkFont.Font(family="Consolas", size=14, weight="bold", slant="roman")
            Label_1.configure(font=fontExample)
            Label_1.pack(side=LEFT, padx=5, pady=5)
            
            def mask_image(image, masks):
                # Loop over each detected object's mask
                for i in range(masks.shape[2]):
                    # Draw the mask for the current object
                    mask = masks[:, :, i]
                    color = (1.0, 0.0, 0.0) # Red
                    image = mrcnn.visualize.apply_mask(image, mask, color, alpha=0.2)

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
                    bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
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
            SOURCE_VIDEO_FILE = fl#"C:/Progs/Clear_3/mask_rcnn-master/test_images/parking_2.mp4"

            # Create a Mask-RCNN model in inference mode
            model = MaskRCNN(mode="inference", model_dir="logs", config=ObjectDetectorConfig())

            # Load pre-trained model
            model.load_weights(TRAINED_MODEL_PATH, by_name=True)

            # COCO Class names
            #class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
            import mrcnn.visualize

            CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

            cvetos = mrcnn.visualize.random_colors(len(CLASS_NAMES))
            # Load the video file we want to run detection on
            video_capture = cv2.VideoCapture(SOURCE_VIDEO_FILE)

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

                # Visualize results
                r = results[0]
                #rgb_image = mask_image(rgb_image, r['masks'])
                rgb_image = contur_image(rgb_image, r['masks'], r['rois'], r['class_ids'], cvetos)

                # Convert the image back to BGR
                bgr_image = rgb_image[:, :, ::-1]

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
            
 
    #def readFile(self, filename):
        #with open(filename, "r") as f:
            #text = f.read()
 
        #return text
 
 
def main():
    root = Tk()
    ex = Example()
    root.geometry("300x250+300+300")
    root.mainloop()
 
 
if __name__ == '__main__':
    main()