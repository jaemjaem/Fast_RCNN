import os
import random
import xml.etree.ElementTree as ET

VOC_ROOT_DIR ="/mnt/VOC2012/"
ANNO_DIR = os.path.join(VOC_ROOT_DIR, "Annotations") 
IMAGE_DIR = os.path.join(VOC_ROOT_DIR, "JPEGImages") 

xml_file1 = os.path.join(ANNO_DIR, '2007_000027.xml')
xml_file2 = os.path.join(ANNO_DIR, '2007_000032.xml')
xml_file3 = os.path.join(ANNO_DIR, '2007_000033.xml')
xml_file4 = os.path.join(ANNO_DIR, '2007_000039.xml')
xml_file5 = os.path.join(ANNO_DIR, '2007_000042.xml')

xml_files = [xml_file1, xml_file2, xml_file3, xml_file4, xml_file5]


print(xml_files)

full_objects_list = []

for xml_file in xml_files:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    image_name = root.find('filename').text
    full_image_name = os.path.join(IMAGE_DIR, image_name)
    image_size = root.find('size')
    image_width = int(image_size.find('width').text)
    image_height = int(image_size.find('height').text)

    obj_list = []
    for obj in root.findall('object'):
        xmlbox = obj.find('bndbox')
        x1 = int(xmlbox.find('xmin').text) / image_width
        y1 = int(xmlbox.find('ymin').text) / image_height
        x2 = int(xmlbox.find('xmax').text) / image_width
        y2 = int(xmlbox.find('ymax').text) / image_height

        bndbox_pos = (x1, y1, x2, y2)
        class_name=obj.find('name').text
        object_dict={'class_name': class_name, 'bndbox_pos':bndbox_pos} #딕셔너리
        obj_list.append(object_dict)
    full_objects_list.append(obj_list)

for object_list in full_objects_list:
    print(object_list)