# The conversion file
# ODGT --> YOLO


import xml.etree.ElementTree as ET
import glob
import os
import cv2
import json


#ODGT
def image_shape(ID, image_dir):
    print(ID)
    jpg_path = image_dir+ID+'.jpg'
    print(jpg_path)
    img = cv2.imread(jpg_path)
    return img.shape


def txt_line(cls, bbox, img_w, img_h):
    """Generate 1 line in the txt file."""
    x, y, w, h = bbox
    x = max(int(x), 0)
    y = max(int(y), 0)
    w = min(int(w), img_w - x)
    h = min(int(h), img_h - y)

    cx = (x + w / 2.) / img_w
    cy = (y + h / 2.) / img_h
    nw = float(w) / img_w
    nh = float(h) / img_h
    return '%d %.6f %.6f %.6f %.6f\n' % (cls, cx, cy, nw, nh)


def process(set_='valid', annotation_filename=None,
            output_dir=None): 
    jpgs = []
    with open(annotation_filename, 'r') as fanno:
        for raw_anno in fanno.readlines():
            anno = json.loads(raw_anno)
            ID = anno['ID']  # e.g. '273271,c9db000d5146c15'
            print('Processing ID: %s' % ID)
            try:
                img_h, img_w, img_c = image_shape(ID, f'G:/tensorGO/dataset/human detection/{set_}/images/')
                assert img_c == 3  # should be a BGR image
                txt_path = f'../dataset/human detection/{set_}/labels/'+ID+'.txt'
                # write a txt for each image
                with open(txt_path, 'w') as ftxt:
                    for obj in anno['gtboxes']:
                        if obj['tag'] == 'mask':
                            continue  # ignore non-human
                        assert obj['tag'] == 'person'
                        if 'hbox' in obj.keys():  # head
                            line = txt_line(1, obj['hbox'], img_w, img_h)
                            if line:
                                ftxt.write(line)
                        if 'fbox' in obj.keys():  # full body
                            line = txt_line(0, obj['fbox'], img_w, img_h)
                            if line:
                                ftxt.write(line)
                jpgs.append('../dataset/human detection/train/images/'+ID+'.jpg')
            except:
                pass