CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
test = input('0 - узнать по id класс, 1 - узнать по классу id')
print(test)
if test == '0':
    ids = input('введи номер айдишника, я дам название')
    print(CLASS_NAMES[int(ids)])
elif test == '1':
    names = input('введи имя класс, я верну айдишник')
    print(CLASS_NAMES.index(str(names)))

#3 - машина
#1 - человек
#14 - скамейка
#17 - dog
#19 - sheep, 20 - cow
#37 - skate, 32 - snow, 39 - tennis
#61 - dining table
#24 - giraffe, 21 - elephant, 23 - zebra
#
#