from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import os
import json
from .ddd_utils import compute_box_3d, project_to_image, draw_box_3d
from utils.image import to_polar_coords, to_cartesian_coords,to_absolut_ref
class Debugger(object):
  def __init__(self, ipynb=False, theme='black',
               num_classes=-1, dataset=None, down_ratio=4):
    self.ipynb = ipynb
    if not self.ipynb:
      import matplotlib.pyplot as plt
      self.plt = plt
    self.imgs = {}
    self.theme = theme
    colors = [(color_list[_]).astype(np.uint8) \
            for _ in range(len(color_list))]
    self.colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)
    if self.theme == 'white':
      self.colors = self.colors.reshape(-1)[::-1].reshape(len(colors), 1, 1, 3)
      self.colors = np.clip(self.colors, 0., 0.6 * 255).astype(np.uint8)
    self.dim_scale = 1
    if dataset == 'coco_hp':
      self.names = ['p']
      self.num_class = 1
      self.num_joints = 17
      self.edges = [[0, 1], [0, 2], [1, 3], [2, 4], 
                    [3, 5], [4, 6], [5, 6], 
                    [5, 7], [7, 9], [6, 8], [8, 10], 
                    [5, 11], [6, 12], [11, 12], 
                    [11, 13], [13, 15], [12, 14], [14, 16]]
      self.ec = [(255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255), 
                 (255, 0, 0), (0, 0, 255), (255, 0, 255),
                 (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255),
                 (255, 0, 0), (0, 0, 255), (255, 0, 255),
                 (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255)]
      self.colors_hp = [(255, 0, 255), (255, 0, 0), (0, 0, 255), 
        (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
        (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
        (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
        (255, 0, 0), (0, 0, 255)]
    elif num_classes == 80 or dataset == 'coco':
      self.names = coco_class_name
    elif dataset == 'kidpath' or dataset == 'kidpath_old':
      self.names = kidpath_class_name
    elif dataset == 'kidmouse':
      self.names = kidpath_class_name
    elif dataset == 'monuseg':
      self.names = monuseg_class_name
    elif dataset == 'nucls':
        self.names = nucls_class_name
    elif num_classes == 20 or dataset == 'pascal':
      self.names = pascal_class_name
    elif dataset == 'gta':
      self.names = gta_class_name
      self.focal_length = 935.3074360871937
      self.W = 1920
      self.H = 1080
      self.dim_scale = 3
    elif dataset == 'viper':
      self.names = gta_class_name
      self.focal_length = 1158
      self.W = 1920
      self.H = 1080
      self.dim_scale = 3
    elif num_classes == 3 or dataset == 'kitti':
      self.names = kitti_class_name
      self.focal_length = 721.5377
      self.W = 1242
      self.H = 375
    elif num_classes == 1 or dataset == 'grapes':
      self.names = grapes_class_name
    elif num_classes == 1 or dataset == 'grapes_with_occ_reg':
      self.names = grapes_wo_class_name
    elif num_classes == 1 or dataset == 'polygons':
      self.names = polygons_class_name

    # num_classes = len(self.names)
    self.down_ratio=down_ratio
    # for bird view
    self.world_size = 64
    self.out_size = 384

  def add_img(self, img, img_id='default', revert_color=False):
    if revert_color:
      img = 255 - img
    self.imgs[img_id] = img.copy()
  
  def add_mask(self, mask, bg, imgId = 'default', trans = 0.8):
    self.imgs[imgId] = (mask.reshape(
      mask.shape[0], mask.shape[1], 1) * 255 * trans + \
      bg * (1 - trans)).astype(np.uint8)
  
  def show_img(self, pause = False, imgId = 'default'):
    cv2.imshow('{}'.format(imgId), self.imgs[imgId])
    if pause:
      cv2.waitKey()
  
  def add_blend_img(self, back, fore, img_id='blend', trans=0.7):
    if self.theme == 'white':
      fore = 255 - fore
    if fore.shape[0] != back.shape[0] or fore.shape[0] != back.shape[1]:
      fore = cv2.resize(fore, (back.shape[1], back.shape[0]))
    if len(fore.shape) == 2:
      fore = fore.reshape(fore.shape[0], fore.shape[1], 1)
    self.imgs[img_id] = (back * (1. - trans) + fore * trans)
    self.imgs[img_id][self.imgs[img_id] > 255] = 255
    self.imgs[img_id][self.imgs[img_id] < 0] = 0
    self.imgs[img_id] = self.imgs[img_id].astype(np.uint8).copy()

  '''
  # slow version
  def gen_colormap(self, img, output_res=None):
    # num_classes = len(self.colors)
    img[img < 0] = 0
    h, w = img.shape[1], img.shape[2]
    if output_res is None:
      output_res = (h * self.down_ratio, w * self.down_ratio)
    color_map = np.zeros((output_res[0], output_res[1], 3), dtype=np.uint8)
    for i in range(img.shape[0]):
      resized = cv2.resize(img[i], (output_res[1], output_res[0]))
      resized = resized.reshape(output_res[0], output_res[1], 1)
      cl = self.colors[i] if not (self.theme == 'white') \
           else 255 - self.colors[i]
      color_map = np.maximum(color_map, (resized * cl).astype(np.uint8))
    return color_map
    '''

  
  def gen_colormap(self, img, output_res=None):
    img = img.copy()
    c, h, w = img.shape[0], img.shape[1], img.shape[2]
    if output_res is None:
      output_res = (h * self.down_ratio, w * self.down_ratio)
    img = img.transpose(1, 2, 0).reshape(h, w, c, 1).astype(np.float32)
    colors = np.array(
      self.colors, dtype=np.float32).reshape(-1, 3)[:c].reshape(1, 1, c, 3)
    if self.theme == 'white':
      colors = 255 - colors
    color_map = (img * colors).max(axis=2).astype(np.uint8)
    color_map = cv2.resize(color_map, (output_res[0], output_res[1]))
    return color_map
    
  '''
  # slow
  def gen_colormap_hp(self, img, output_res=None):
    # num_classes = len(self.colors)
    # img[img < 0] = 0
    h, w = img.shape[1], img.shape[2]
    if output_res is None:
      output_res = (h * self.down_ratio, w * self.down_ratio)
    color_map = np.zeros((output_res[0], output_res[1], 3), dtype=np.uint8)
    for i in range(img.shape[0]):
      resized = cv2.resize(img[i], (output_res[1], output_res[0]))
      resized = resized.reshape(output_res[0], output_res[1], 1)
      cl =  self.colors_hp[i] if not (self.theme == 'white') else \
        (255 - np.array(self.colors_hp[i]))
      color_map = np.maximum(color_map, (resized * cl).astype(np.uint8))
    return color_map
  '''
  def gen_colormap_hp(self, img, output_res=None):
    c, h, w = img.shape[0], img.shape[1], img.shape[2]
    if output_res is None:
      output_res = (h * self.down_ratio, w * self.down_ratio)
    img = img.transpose(1, 2, 0).reshape(h, w, c, 1).astype(np.float32)
    colors = np.array(
      self.colors_hp, dtype=np.float32).reshape(-1, 3)[:c].reshape(1, 1, c, 3)
    if self.theme == 'white':
      colors = 255 - colors
    color_map = (img * colors).max(axis=2).astype(np.uint8)
    color_map = cv2.resize(color_map, (output_res[0], output_res[1]))
    return color_map


  def add_rect(self, rect1, rect2, c, conf=1, img_id='default'): 
    cv2.rectangle(
      self.imgs[img_id], (rect1[0], rect1[1]), (rect2[0], rect2[1]), c, 2)
    if conf < 1:
      cv2.circle(self.imgs[img_id], (rect1[0], rect1[1]), int(10 * conf), c, 1)
      cv2.circle(self.imgs[img_id], (rect2[0], rect2[1]), int(10 * conf), c, 1)
      cv2.circle(self.imgs[img_id], (rect1[0], rect2[1]), int(10 * conf), c, 1)
      cv2.circle(self.imgs[img_id], (rect2[0], rect1[1]), int(10 * conf), c, 1)

  def add_coco_bbox(self, bbox, cat, conf=1, show_txt=True, img_id='default'): 
    bbox = np.array(bbox, dtype=np.int32)
    # cat = (int(cat) + 1) % 80
    cat = int(cat)
    # print('cat', cat, self.names[cat])
    # c = self.colors[cat][0][0].tolist()
    # if self.theme == 'white':
    #   c = (255 - np.array(c)).tolist()
    c = (0, 255, 0) # hardcode to green
    txt = '{}{:.1f}'.format(self.names[cat], conf)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
    cv2.rectangle(
      self.imgs[img_id], (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 2)
    # if show_txt:
    #   cv2.rectangle(self.imgs[img_id],
    #                 (bbox[0], bbox[1] - cat_size[1] - 2),
    #                 (bbox[0] + cat_size[0], bbox[1] - 2), c, -1)
    #   cv2.putText(self.imgs[img_id], txt, (bbox[0], bbox[1] - 2),
    #               font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

  def add_coco_circle(self, circle, cat=0, conf=1, show_txt=False, img_id='default'):
    circle = np.array(circle, dtype=np.int32)
    cat=0
    # cat = (int(cat) + 1) % 80
    cat = int(cat)
    # print('cat', cat, self.names[cat])
    # c = self.colors[cat][0][0].tolist()
    # if self.theme == 'white':
    #   c = (255 - np.array(c)).tolist()
    c = (0, 255, 0)
    # c = (0, 255, 0)  # hardcode to green
    txt = '{}{:.1f}'.format(self.names[cat], conf)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
    # cv2.rectangle(
    #   self.imgs[img_id], (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 2)
    cv2.circle(self.imgs[img_id], (circle[0], circle[1]), circle[2], c, 1)

  def add_circle_and_occlusion(self, circle, occ, cat=0, conf=1, show_txt=False, img_id='default'):
    circle = np.array(circle, dtype=np.int32)
    cat=0
    occ = round(occ, 2)

    # cat = (int(cat) + 1) % 80
    cat = int(cat)
    # print('cat', cat, self.names[cat])
    # c = self.colors[cat][0][0].tolist()
    # if self.theme == 'white':
    #   c = (255 - np.array(c)).tolist()
    c = (0, 255, 0)
    # c = (0, 255, 0)  # hardcode to green
    txt = '{}{:.1f}'.format(self.names[cat], conf)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
    # cv2.rectangle(
    #   self.imgs[img_id], (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 2)
    cv2.circle(self.imgs[img_id], (circle[0], circle[1]), circle[2], c, 1)

    org = (circle[0], circle[1])
    fontScale = 0.3
    color = (0, 255, 0)
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(self.imgs[img_id], str(occ), org, font,
              fontScale, color, thickness, cv2.LINE_AA)
  def add_coco_polygon(self, polygon, cat=0, conf=1, show_txt=False, img_id='default'):
    polygon = np.array(polygon, dtype=np.int32)
    cat = 0
    center = (polygon[0], polygon[1])
    polygon = polygon[2:10]
    # cat = (int(cat) + 1) % 80
    cat = int(cat)
    c = (0, 255, 0)
    txt = '{}{:.1f}'.format(self.names[cat], conf)
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
    # cv2.rectangle(
    #   self.imgs[img_id], (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 2)
    isClosed = True
    polygon = to_cartesian_coords(polygon)
    polygon = to_absolut_ref(polygon, center)
    polygon = np.int32(np.rint(np.asarray([[polygon[i], polygon[i+1]] for i in range(0,len(polygon)-1, 2)])))
    # cv2.imshow('',self.imgs[img_id])
    # cv2.waitKey(0)
    cv2.polylines(self.imgs[img_id], [polygon], isClosed, c, thickness)
    # cv2.imshow('', self.imgs[img_id])
    # cv2.waitKey(0)
    if show_txt:
      bbox =  np.array(np.zeros((4)), dtype=np.int32)
      # bbox[0] = circle[0] - circle[2]
      # bbox[1] = circle[1] - circle[2]
      cv2.rectangle(self.imgs[img_id],
                    (bbox[0], bbox[1] - cat_size[1] - 2),
                    (bbox[0] + cat_size[0], bbox[1] - 2), c, -1)
      cv2.putText(self.imgs[img_id], txt, (bbox[0], bbox[1] - 2),
                  font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)


  def add_coco_hp(self, points, img_id='default'): 
    points = np.array(points, dtype=np.int32).reshape(self.num_joints, 2)
    for j in range(self.num_joints):
      cv2.circle(self.imgs[img_id],
                 (points[j, 0], points[j, 1]), 3, self.colors_hp[j], -1)
    for j, e in enumerate(self.edges):
      if points[e].min() > 0:
        cv2.line(self.imgs[img_id], (points[e[0], 0], points[e[0], 1]),
                      (points[e[1], 0], points[e[1], 1]), self.ec[j], 2,
                      lineType=cv2.LINE_AA)





  def add_points(self, points, img_id='default'):
    num_classes = len(points)
    # assert num_classes == len(self.colors)
    for i in range(num_classes):
      for j in range(len(points[i])):
        c = self.colors[i, 0, 0]
        cv2.circle(self.imgs[img_id], (points[i][j][0] * self.down_ratio, 
                                       points[i][j][1] * self.down_ratio),
                   5, (255, 255, 255), -1)
        cv2.circle(self.imgs[img_id], (points[i][j][0] * self.down_ratio,
                                       points[i][j][1] * self.down_ratio),
                   3, (int(c[0]), int(c[1]), int(c[2])), -1)

  def show_all_imgs(self,image_or_path_or_tensor, pause=False, time=0):
    if not self.ipynb:
      for i, v in self.imgs.items():
        # cv2.imshow('{}'.format(i), v)
        path, image_name = os.path.split(image_or_path_or_tensor)
        cv2.imwrite(self.opt.inference_folder + image_name, v)
      # if cv2.waitKey(0 if pause else 1) == 27:
      #   import sys
      #   sys.exit(0)
    else:
      self.ax = None
      nImgs = len(self.imgs)
      fig=self.plt.figure(figsize=(nImgs * 10,10))
      nCols = nImgs
      nRows = nImgs // nCols
      for i, (k, v) in enumerate(self.imgs.items()):
        fig.add_subplot(1, nImgs, i + 1)
        if len(v.shape) == 3:
          self.plt.imshow(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))
        else:
          self.plt.imshow(v)
      self.plt.show()

  def save_img(self, imgId='default', path='./cache/debug/'):
    cv2.imwrite(path + '{}.png'.format(imgId), self.imgs[imgId])
    
  def save_all_imgs(self, path='./cache/debug/', prefix='', genID=False):
    if genID:
      try:
        idx = int(np.loadtxt(path + '/id.txt'))
      except:
        idx = 0
      prefix=idx
      np.savetxt(path + '/id.txt', np.ones(1) * (idx + 1), fmt='%d')
    for i, v in self.imgs.items():
      cv2.imwrite(path + '/{}{}.png'.format(prefix, i), v)

  def save_detect_all_to_json(self, simg_big, detect_all, json_file, opt, simg, cat=0):

      try:
          start_x = np.int(simg.properties['openslide.bounds-x'])
          start_y = np.int(simg.properties['openslide.bounds-y'])
          width_x = np.int(simg.properties['openslide.bounds-width'])
          height_y = np.int(simg.properties['openslide.bounds-height'])
      except:
          start_x = 0
          start_y = 0
          width_x = np.int(simg.properties['aperio.OriginalWidth'])
          height_y = np.int(simg.properties['aperio.OriginalHeight'])
      down_rate = simg.level_downsamples[opt.lv]



      data = {}
      data['version'] = '3.16.7'
      data['flags'] = {}
      data['flags']['@MicronsPerPixel'] = simg.properties['openslide.mpp-x']
      data['flags']['@Level'] = opt.lv
      data['flags']['@DownRate'] = down_rate
      data['flags']['@start_x'] = start_x
      data['flags']['@start_y'] = start_y
      data['flags']['@width_x'] = width_x
      data['flags']['@height_y'] = height_y
      if 'leica.device-model' in simg.properties:
          data['flags']['@Device'] = 'leica.device-model'
      else:
          data['flags']['@Device'] = 'aperio.Filename'

      data['shapes'] = []
      data['lineColor'] = [0, 255, 0, 128]
      data['fillColor'] = [255, 0, 0, 128]
      data['imagePath'] = os.path.basename(json_file).replace('json', 'png')
      data['imageData'] = None
      data['imageHeight'] = simg_big.shape[0]
      data['imageWidth'] = simg_big.shape[1]
      label_name = self.names[cat]
      shapes = []
      for circle in detect_all:
          pt0 = [circle[0], circle[1]]
          pt1 = [circle[0], circle[1]+circle[2]]
          shape = {}
          shape['label'] = label_name
          shape['line_color'] = None
          shape['fill_color'] = None
          shape['points'] = [pt0, pt1]
          shape['shape_type'] = 'circle'
          shape['flags'] = {}
          shape['flags']['prob'] = circle[3]
          shapes.append(shape)
      data['shapes'] = shapes
      with open(json_file, 'w') as output_json_file:
          json.dump(data, output_json_file)


  def remove_side(self, img_id, img):
    if not (img_id in self.imgs):
      return
    ws = img.sum(axis=2).sum(axis=0)
    l = 0
    while ws[l] == 0 and l < len(ws):
      l+= 1
    r = ws.shape[0] - 1
    while ws[r] == 0 and r > 0:
      r -= 1
    hs = img.sum(axis=2).sum(axis=1)
    t = 0
    while hs[t] == 0 and t < len(hs):
      t += 1
    b = hs.shape[0] - 1
    while hs[b] == 0 and b > 0:
      b -= 1
    self.imgs[img_id] = self.imgs[img_id][t:b+1, l:r+1].copy()

  def project_3d_to_bird(self, pt):
    pt[0] += self.world_size / 2
    pt[1] = self.world_size - pt[1]
    pt = pt * self.out_size / self.world_size
    return pt.astype(np.int32)

  def add_ct_detection(
    self, img, dets, show_box=False, show_txt=True, 
    center_thresh=0.5, img_id='det'):
    # dets: max_preds x 5
    self.imgs[img_id] = img.copy()
    if type(dets) == type({}):
      for cat in dets:
        for i in range(len(dets[cat])):
          if dets[cat][i, 2] > center_thresh:
            cl = (self.colors[cat, 0, 0]).tolist()
            ct = dets[cat][i, :2].astype(np.int32)
            if show_box:
              w, h = dets[cat][i, -2], dets[cat][i, -1]
              x, y = dets[cat][i, 0], dets[cat][i, 1]
              bbox = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2],
                              dtype=np.float32)
              self.add_coco_bbox(
                bbox, cat - 1, dets[cat][i, 2], 
                show_txt=show_txt, img_id=img_id)
    else:
      for i in range(len(dets)):
        if dets[i, 2] > center_thresh:
          # print('dets', dets[i])
          cat = int(dets[i, -1])
          cl = (self.colors[cat, 0, 0] if self.theme == 'black' else \
                                       255 - self.colors[cat, 0, 0]).tolist()
          ct = dets[i, :2].astype(np.int32) * self.down_ratio
          cv2.circle(self.imgs[img_id], (ct[0], ct[1]), 3, cl, -1)
          if show_box:
            w, h = dets[i, -3] * self.down_ratio, dets[i, -2] * self.down_ratio
            x, y = dets[i, 0] * self.down_ratio, dets[i, 1] * self.down_ratio
            bbox = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2],
                            dtype=np.float32)
            self.add_coco_bbox(bbox, dets[i, -1], dets[i, 2], img_id=img_id)


  def add_3d_detection(
    self, image_or_path, dets, calib, show_txt=False, 
    center_thresh=0.5, img_id='det'):
    if isinstance(image_or_path, np.ndarray):
      self.imgs[img_id] = image_or_path
    else: 
      self.imgs[img_id] = cv2.imread(image_or_path)
    for cat in dets:
      for i in range(len(dets[cat])):
        cl = (self.colors[cat - 1, 0, 0]).tolist()
        if dets[cat][i, -1] > center_thresh:
          dim = dets[cat][i, 5:8]
          loc  = dets[cat][i, 8:11]
          rot_y = dets[cat][i, 11]
          # loc[1] = loc[1] - dim[0] / 2 + dim[0] / 2 / self.dim_scale
          # dim = dim / self.dim_scale
          if loc[2] > 1:
            box_3d = compute_box_3d(dim, loc, rot_y)
            box_2d = project_to_image(box_3d, calib)
            self.imgs[img_id] = draw_box_3d(self.imgs[img_id], box_2d, cl)

  def compose_vis_add(
    self, img_path, dets, calib,
    center_thresh, pred, bev, img_id='out'):
    self.imgs[img_id] = cv2.imread(img_path)
    # h, w = self.imgs[img_id].shape[:2]
    # pred = cv2.resize(pred, (h, w))
    h, w = pred.shape[:2]
    hs, ws = self.imgs[img_id].shape[0] / h, self.imgs[img_id].shape[1] / w
    self.imgs[img_id] = cv2.resize(self.imgs[img_id], (w, h))
    self.add_blend_img(self.imgs[img_id], pred, img_id)
    for cat in dets:
      for i in range(len(dets[cat])):
        cl = (self.colors[cat - 1, 0, 0]).tolist()
        if dets[cat][i, -1] > center_thresh:
          dim = dets[cat][i, 5:8]
          loc  = dets[cat][i, 8:11]
          rot_y = dets[cat][i, 11]
          # loc[1] = loc[1] - dim[0] / 2 + dim[0] / 2 / self.dim_scale
          # dim = dim / self.dim_scale
          if loc[2] > 1:
            box_3d = compute_box_3d(dim, loc, rot_y)
            box_2d = project_to_image(box_3d, calib)
            box_2d[:, 0] /= hs
            box_2d[:, 1] /= ws
            self.imgs[img_id] = draw_box_3d(self.imgs[img_id], box_2d, cl)
    self.imgs[img_id] = np.concatenate(
      [self.imgs[img_id], self.imgs[bev]], axis=1)

  def add_2d_detection(
    self, img, dets, show_box=False, show_txt=True, 
    center_thresh=0.5, img_id='det'):
    self.imgs[img_id] = img
    for cat in dets:
      for i in range(len(dets[cat])):
        cl = (self.colors[cat - 1, 0, 0]).tolist()
        if dets[cat][i, -1] > center_thresh:
          bbox = dets[cat][i, 1:5]
          self.add_coco_bbox(
            bbox, cat - 1, dets[cat][i, -1], 
            show_txt=show_txt, img_id=img_id)

  def add_bird_view(self, dets, center_thresh=0.3, img_id='bird'):
    bird_view = np.ones((self.out_size, self.out_size, 3), dtype=np.uint8) * 230
    for cat in dets:
      cl = (self.colors[cat - 1, 0, 0]).tolist()
      lc = (250, 152, 12)
      for i in range(len(dets[cat])):
        if dets[cat][i, -1] > center_thresh:
          dim = dets[cat][i, 5:8]
          loc  = dets[cat][i, 8:11]
          rot_y = dets[cat][i, 11]
          rect = compute_box_3d(dim, loc, rot_y)[:4, [0, 2]]
          for k in range(4):
            rect[k] = self.project_3d_to_bird(rect[k])
            # cv2.circle(bird_view, (rect[k][0], rect[k][1]), 2, lc, -1)
          cv2.polylines(
              bird_view,[rect.reshape(-1, 1, 2).astype(np.int32)],
              True,lc,2,lineType=cv2.LINE_AA)
          for e in [[0, 1]]:
            t = 4 if e == [0, 1] else 1
            cv2.line(bird_view, (rect[e[0]][0], rect[e[0]][1]),
                    (rect[e[1]][0], rect[e[1]][1]), lc, t,
                    lineType=cv2.LINE_AA)
    self.imgs[img_id] = bird_view

  def add_bird_views(self, dets_dt, dets_gt, center_thresh=0.3, img_id='bird'):
    alpha = 0.5
    bird_view = np.ones((self.out_size, self.out_size, 3), dtype=np.uint8) * 230
    for ii, (dets, lc, cc) in enumerate(
      [(dets_gt, (12, 49, 250), (0, 0, 255)), 
       (dets_dt, (250, 152, 12), (255, 0, 0))]):
      # cc = np.array(lc, dtype=np.uint8).reshape(1, 1, 3)
      for cat in dets:
        cl = (self.colors[cat - 1, 0, 0]).tolist()
        for i in range(len(dets[cat])):
          if dets[cat][i, -1] > center_thresh:
            dim = dets[cat][i, 5:8]
            loc  = dets[cat][i, 8:11]
            rot_y = dets[cat][i, 11]
            rect = compute_box_3d(dim, loc, rot_y)[:4, [0, 2]]
            for k in range(4):
              rect[k] = self.project_3d_to_bird(rect[k])
            if ii == 0:
              cv2.fillPoly(
                bird_view,[rect.reshape(-1, 1, 2).astype(np.int32)],
                lc,lineType=cv2.LINE_AA)
            else:
              cv2.polylines(
                bird_view,[rect.reshape(-1, 1, 2).astype(np.int32)],
                True,lc,2,lineType=cv2.LINE_AA)
            # for e in [[0, 1], [1, 2], [2, 3], [3, 0]]:
            for e in [[0, 1]]:
              t = 4 if e == [0, 1] else 1
              cv2.line(bird_view, (rect[e[0]][0], rect[e[0]][1]),
                      (rect[e[1]][0], rect[e[1]][1]), lc, t,
                      lineType=cv2.LINE_AA)
    self.imgs[img_id] = bird_view


kitti_class_name = [
  'p', 'v', 'b'
]

gta_class_name = [
  'p', 'v'
]

pascal_class_name = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", 
  "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", 
  "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

coco_class_name = [
     'person', 'bicycle', 'car', 'motorcycle', 'airplane',
     'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
     'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
     'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
     'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
     'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
     'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
     'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
     'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
     'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
     'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
     'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

kidpath_class_name = ['glomerulus']

monuseg_class_name = ['nuclei']

nucls_class_name = [
    'tumor', 'fibroblast', 'lymphocyte', 'plasma_cell', 'macrophage',
    'mitotic_figure', 'vascular_endothelium', 'myoepithelium', 'neutrophil',
    'apoptotic_body', 'ductal_epithelium', 'eosinophil'
]

grapes_class_name = ['grape']
grapes_wo_class_name = ['grape']
polygons_class_name = ['polygon']

color_list = np.array(
        [
            1.000, 1.000, 1.000,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            0.000, 0.447, 0.741,
            0.50, 0.5, 0
        ]
    ).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255