import os
import cv2
import numpy as np
import mrcnn.config
import mrcnn
from mrcnn.model import MaskRCNN
from pathlib import Path
from datetime import datetime

class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.8 # минимальный процент отображения прямоугольника
    NUM_CLASSES = 81

import mrcnn.utils

DATASET_FILE = "mask_rcnn_coco.h5"
if not os.path.exists(DATASET_FILE):
    mrcnn.utils.download_trained_weights(DATASET_FILE)

model = MaskRCNN(mode="inference", model_dir="logs", config=MaskRCNNConfig())
model.load_weights(DATASET_FILE, by_name=True)
image_directory = input('Please, enter name of folder of images to recognition:...')
IMAGE_DIR = os.path.join(os.getcwd(), image_directory)

##------------------------------------------------------------------------------
def visualize_detections(image, masks, boxes, class_ids, scores):
    import numpy as np
    bgr_image = image#[:, :, ::-1]

    import mrcnn.visualize

    CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    COLORS = mrcnn.visualize.random_colors(len(CLASS_NAMES))

    for i in range(boxes.shape[0]):        
        y1, x1, y2, x2 = boxes[i]

        classID = class_ids[i]            
        label = CLASS_NAMES[classID]
        font = cv2.FONT_HERSHEY_DUPLEX
        color = [int(c) for c in np.array(COLORS[classID]) * 255]
        text = "{}: {:.3f}".format(label, scores[i])
        size = 0.4
        width = 1
        mask = masks[:, :, i]  # берем срез
        bgr_image = mrcnn.visualize.apply_mask(bgr_image, mask, color, alpha=0.6) # рисование маски
        #bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        cv2.rectangle(bgr_image, (x1, y1), (x2, y2), color, width)
        cv2.putText(bgr_image, text, (x1, y1-10), font, size, color, width)
        
    #ВЫЗОВ ФУНКЦИИ СИТУАЦИИ
    resik = situation(class_ids)
    print(resik)
    font = cv2.FONT_HERSHEY_COMPLEX
    colorik = (0,0,255)
    size = 0.5
    width = 1
    text_color = (0,0,0)
    #bgr_image = bgr_image[:, :, ::-1]
    height_img, width_img = bgr_image[:2]
    overlay = bgr_image.copy()
    text_width, text_height = cv2.getTextSize(resik, font, size, width)
    text_coord = (5,text_height+20)
    if resik != '':
        cv2.rectangle(overlay, 
              (text_coord[0]-5, text_coord[1]+text_height),
              (text_width[0]+10, 0),
              (0, 255, 0),
              -1)
        opacity = 0.6
        cv2.addWeighted(overlay, opacity, bgr_image, 1 - opacity, 0, bgr_image)
    cv2.putText(bgr_image, resik, text_coord, font, size, text_color, width)
    #cv2.putText(bgr_image, text, (x1, y1-10), font, size, color, width)
    #bgr_image = bgr_image[:, :, ::-1]

    return bgr_image
#---------------------------------------------------------------------------------------------------------------

put = os.path.join(os.getcwd(), 'RECOGNIZED') + '\\' + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.mkdir(put)

#---------------------------------------------------------------------------------------------------------------

def situation(in_list):
    #на вход подается - r['rois'] - массив с айдишниками распознанных на кадре объектов
    CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    import numpy as np
    list_of_reco = []

    for i in range(in_list.size):
        classID = in_list[i]            
        label = CLASS_NAMES[classID]
        #print(f'\n\n{label}')
        list_of_reco.append(label)


    animal = []
    sravn = ['car', 'person', 'potted plant']
    out_one = ""
    #СИТУАЦИЯ
    #if ( list(set(CLASS_NAMES) - set(list_of_reco)) == ['c'])
    if ('car' in list_of_reco) and ('person' not in list_of_reco) and (list_of_reco.count('car') == 1):
        out_one = 'Машина на парковке/едет по дороге'
    elif ('car' in list_of_reco) and ('person' not in list_of_reco) and (list_of_reco.count('car') >= 1):
        out_one = 'Машины на парковке/едут по дороге'
    elif (list_of_reco.count('person') >= 3) and ('car' in list_of_reco) and (list_of_reco.count('car') <= 2):
        out_one = 'Группа людей на городской улице'
    elif (list_of_reco.count('person') >= 3) and ('car' in list_of_reco) and (list_of_reco.count('car') > 2):
        out_one = 'Городская улица'
    elif (list_of_reco.count('person') < 3) and ('car' in list_of_reco) and (list_of_reco.count('car') > 2):
        out_one = 'Машины на парковке/едут по дороге'
    elif ('car' in list_of_reco) and ('person' in list_of_reco) and (list_of_reco.count('person') < 3):
        out_one = 'Человек паркует машину'
    elif ('car' not in list_of_reco) and ('person' in list_of_reco) and (list_of_reco.count('person') > 1) and ( ('umbrella' in list_of_reco) or ('fire hydrant' in list_of_reco) or ('dining table' not in list_of_reco) or ('chair' not in list_of_reco)):
        out_one = 'Группа людей на улице'
    
    
    #выгул животных
    if ('person' in list_of_reco) and ('dog' in list_of_reco):
        out_one = out_one + ' человек выгуливает собаку' #adding to existing
    elif (list_of_reco.count('sheep') > 1) or (list_of_reco.count('cow') > 1):
        out_one = out_one + ' домашние животные пасутся' #adding to ...
    #elif ('person' in list_of_reco) or ('person' in list_of_reco)

    #спорт
    if ('person' in list_of_reco) and ('skateboard' in list_of_reco):
        out_one = 'человек катается на скейтборде' #adding to existing
    elif ('person' in list_of_reco) and ('tennis racket' in list_of_reco):
        out_one = 'человек играет в теннис' #adding to existing
    elif ('person' in list_of_reco) and ('snowboard' in list_of_reco):
        out_one = 'человек катается на сноуборде' #adding to existing

    #интерьер
    if ('person' in list_of_reco) and (('dining table' in list_of_reco) or ('chair' in list_of_reco)) and (list_of_reco.count('person') > 1):
        out_one = out_one + ' / люди сидят за столами' #adding to existing
    elif ('person' in list_of_reco) and (('dining table' in list_of_reco) or ('chair' in list_of_reco)):
        out_one = out_one + ' / человек сидит за столом' #adding to existing
    

    #МЕСТО
    #if 

    #ЖИВОТНЫЕ
    if ('elephant' in list_of_reco):
        animal.append('слон')
    if ('zebra' in list_of_reco):
        animal.append('зебра')
    if ('giraffe' in list_of_reco):
        animal.append('жираф')
    out_str = ' дикая природа, здесь: '
    out_str = out_str + ', '.join(animal)
    if out_str != ' дикая природа, здесь: ': 
        out_one = out_one + ' /' + out_str

    font = cv2.FONT_HERSHEY_DUPLEX
    color = (0, 0, 0)


    return out_one
#-----------------------------------------------------------------------------------------------------------------------------------------


for filename in os.listdir(IMAGE_DIR):
    image = cv2.imread(os.path.join(IMAGE_DIR, filename))
    rgb_image = image[:, :, ::-1]
    detections = model.detect([rgb_image], verbose=1)[0]
    cv2.imwrite(put + '\\' + filename, visualize_detections(image, detections['masks'], detections['rois'], detections['class_ids'], detections['scores']))





##print(detections)

