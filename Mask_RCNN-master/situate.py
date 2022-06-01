import numpy as np

def situate(in_list):
    CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    list_of_reco = []
    #size = len(in_list)
    for element in in_list:
        #alem = in_list[element]
        #label = CLASS_NAMES[alem]
        print(CLASS_NAMES[int(element)])
        list_of_reco.append(CLASS_NAMES[element])
    print(list_of_reco)

    animal = []
    sravn = ['car', 'person', 'potted plant']
    out_one = ''
    #СИТУАЦИЯ
    #if ( list(set(CLASS_NAMES) - set(list_of_reco)) == ['c'])
    if ('car' in list_of_reco) and ('person' not in list_of_reco) and (list_of_reco.count('car') == 1):
        print('the car is parked/drives on the road')
    elif ('car' in list_of_reco) and ('person' not in list_of_reco) and (list_of_reco.count('car') >= 1):
        print('cars is parked/drives on the road')
    elif (list_of_reco.count('person') >= 3) and ('car' in list_of_reco) and (list_of_reco.count('car') <= 2):
        print('group of people on a city street')   
    elif (list_of_reco.count('person') < 3) and ('car' in list_of_reco) and (list_of_reco.count('car') > 2):
        print('cars is parked/drives on the road')
    elif ('car' in list_of_reco) and ('person' in list_of_reco) and (list_of_reco.count('person') < 3):
        print('man parking a car')
    
    
    #выгул животных
    if ('person' in list_of_reco) and ('dog' in list_of_reco):
        print('man walking the dog') #adding to existing
    elif (list_of_reco.count('sheep') > 1) or (list_of_reco.count('cow') > 1):
        print('domestic animals on grazing') #adding to ...
    #elif ('person' in list_of_reco) or ('person' in list_of_reco)

    #спорт
    if ('person' in list_of_reco) and ('skateboard' in list_of_reco):
        print('man riding a skateboard') #adding to existing
    elif ('person' in list_of_reco) and ('tennis racket' in list_of_reco):
        print('man playing tennis') #adding to existing
    elif ('person' in list_of_reco) and ('snowboard' in list_of_reco):
        print('man snowboarding') #adding to existing

    #интерьер
    if ('person' in list_of_reco) and ('dining table' in list_of_reco) and (list_of_reco.count('person') > 1):
        print('people sit at tables') #adding to existing
    elif ('person' in list_of_reco) and ('dining table' in list_of_reco):
        print('man sitting at the table') #adding to existing
    

    #МЕСТО
    #if 

    #ЖИВОТНЫЕ
    if ('elephant' in list_of_reco):
        animal.append('elephant')
    if ('zebra' in list_of_reco):
        animal.append('zebra')
    if ('giraffe' in list_of_reco):
        animal.append('giraffe')
    out_str = 'wild nature, there is: '
    out_str = out_str + ', '.join(animal)
    if out_str != 'wild nature, there is: ': 
        print(out_str)


in_list = [1, 1, 1, 61, 1, 1]
situate(in_list)