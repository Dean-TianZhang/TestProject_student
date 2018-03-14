#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Tian Zhang, based on code from Ross Girshick, Xinlei Chen
# --------------------------------------------------------

"""
Demo script showing detections in video.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse
import scipy.io as sio
import argparse
import time

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

import csv
from sort.sort import Sort
# from deep_sort import generate_detections
# from deep_sort.deep_sort import nn_matching
# from deep_sort.deep_sort.tracker import Tracker

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

def process_box(bbox, score, h, w, threshold):
    max_indx = np.argmax(score)
    max_prob = score[max_indx]
    if max_prob > threshold:

        left =  int (bbox[0])
        right = int (bbox[0] + (bbox[2] - bbox[0]))
        top = int (bbox[1])
        bot = int (bbox[1] + (bbox[3] - bbox[1]))
        
        mess = 'person'
        return (left, right, top, bot, mess, max_indx, max_prob)
    return None

def vis_detections_video(im, class_name, dets, csv_file, csv, frame_id, encoder, tracker, thresh=0.6):
    """Draw detected bounding boxes."""
    detections = []
    scores = []
    h, w, _ = im.shape
    thick = int((h + w) // 300)
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return im
    for i in inds:
        scores.append(dets[i, -1])

    for i in inds:
        bbox = dets[i, :4]
        boxResults = process_box(bbox, scores, h, w, thresh)
        if boxResults is None:
            continue
        left, right, top, bot, mess, max_indx, confidence = boxResults
        detections.append(np.array([left,top,right,bot]).astype(np.float64))
        
        
        
    detections = np.array(detections)
    trackers = tracker.update(detections)
    for track in trackers:
        bbox = [int(track[0]),int(track[1]),int(track[2]),int(track[3])]
        id_num = str(int(track[4]))
        csv.writerow([frame_id,id_num,int(bbox[0]),int(bbox[1]),int(bbox[2])-int(bbox[0]),int(bbox[3])-int(bbox[1])])
        csv_file.flush()
        cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(0,255,255), thick//3)
        if((bbox[2]-bbox[0]) * (bbox[3]-bbox[1]) > 30000):
            cv2.putText(im, "teacher" + id_num,(int(bbox[0]), int(bbox[1]) - 12),0, 1e-3 * h, (0,255,255),thick//6)
        else:
            cv2.putText(im, id_num,(int(bbox[0]), int(bbox[1]) - 12),0, 1e-3 * h, (255,255,255),thick//6)
        # cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),2)
        # cv2.rectangle(im,(int(bbox[0]),int(bbox[1])-10),(int(bbox[0]+200),int(bbox[1])+10),(10,10,10),-1)
        # cv2.putText(im, id_num,(int(bbox[0]),int(bbox[1]-2)),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255))#,cv2.CV_AA)
    return im

def demo_video(sess, net, im, csv_file, csv, frame_id, encoder, tracker):
    """Detect object classes in an image using pre-computed object proposals."""
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    # print ('Detection took {:.3f}s for '
    #        '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.5

    NMS_THRESH = 0.2
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]

        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        if(cls == 'person'):
            im=vis_detections_video(im, cls, dets, csv_file, csv, frame_id, encoder, tracker, thresh=CONF_THRESH)
    #cv2.imwrite(os.path.join('output',str(time.time())+'.jpg'),im)
    # cv2.imshow('ret',im)
    
    # cv2.waitKey(20)
    return im

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()
    cv2.namedWindow('ret',0)

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])


    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture("TEST", 21,
                          tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    
    ### Load Video File
  
    videoFilePath = "/Users/deanzhang/Desktop/learnable.ai_project/tf-faster-rcnn/test_data/cut_4.mp4"
    videoFile = cv2.VideoCapture(videoFilePath)

    w = videoFile.get(cv2.CAP_PROP_FRAME_WIDTH);
    h = videoFile.get(cv2.CAP_PROP_FRAME_HEIGHT); 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('cut_4_output.mp4',fourcc, 15.0, (int(w),int(h)))
    #fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    #store the position of bounding box
    f = open('{}.csv'.format(videoFilePath),'w')
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['frame_id', 'track_id', 'x', 'y', 'w', 'h'])
    f.flush()

    # loading deep_sort/sort tracker
    encoder = None
    tracker = Sort()
    # metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, 100)
    # tracker = Tracker(metric)
    # encoder = generate_detections.create_box_encoder("/Users/deanzhang/Desktop/learnable.ai_project/tf-faster-rcnn/tools/deep_sort/resources/networks/mars-small128.ckpt-68577")

    frame_id = 0
    while True:
        frame_id += 1;
        ret, image = videoFile.read()
        im = demo_video(sess, net, image, f, writer, frame_id, encoder,tracker)
        out.write(im)

    videoFile.release()
    out.release()
    cv2.destroyAllWindows()
