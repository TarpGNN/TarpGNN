import numpy as np
import skimage.io
import skimage.transform
import os

import torch
import torchvision.transforms as transforms
from torch.utils import data
import torchvision.models as models

from PIL import Image
import random
from collections import defaultdict

import sys

"""
Reference:
https://github.com/cvlab-epfl/social-scene-understanding/blob/master/volleyball.py
"""

ACTIVITIES = ['r_set', 'r_spike', 'r-pass', 'r_winpoint',
              'l_set', 'l-spike', 'l-pass', 'l_winpoint']

NUM_ACTIVITIES = 8

ACTIONS = ['blocking', 'digging', 'falling', 'jumping',
           'moving', 'setting', 'spiking', 'standing',
           'waiting']
NUM_ACTIONS = 9


def volley_read_annotations(cfg, path):
    """
    reading annotations for the given sequence
    """
    annotations = {}

    gact_to_id = {name: i for i, name in enumerate(ACTIVITIES)}

    activity_dict = dict()
    with open(path + '/annotations.txt') as f:
        for l in f.readlines():
            values = l[:-1].split(' ')
            file_name = int(values[0].split('.')[0])
            activity_dict[file_name] = gact_to_id[values[1]]

    file_list = [int(i) for i in os.listdir(path) if ".txt" not in i]
    file_list.sort()
    for file_name in file_list:
        result_path = path + '/' + str(file_name) + '/results.txt'
        activity = activity_dict[file_name]
        with open(result_path, 'r') as fr:
            annotation = defaultdict(list)
            for line in fr.readlines():
                info = line.split(',')[1:7]
                x, y, w, h = map(int, info[2:])
                temp_box = [x / (cfg.image_size[1]*10), y / (cfg.image_size[0]*10),
                            (x + w) / (cfg.image_size[1]*10), (y + h) / (cfg.image_size[0]*10)]
                annotation[int(info[0].split('.')[0])].append(temp_box)
        annotations[file_name] = {
            'group_activity': activity,
            'bboxes': annotation,
        }

    return annotations


def volley_read_dataset(cfg, seqs):
    data = {}
    for sid in seqs:
        data[sid] = volley_read_annotations(cfg, cfg.data_path + '/%d' % sid)
    return data


def load_tracks(anns):
    tracks = {}
    for sid, ann in anns.items():
        for src_fid, values in ann.items():
            temp_track = {}
            for fid, box_info in values["bboxes"].items():
                temp_track[fid] = np.array(box_info)
            tracks[(sid, src_fid)] = temp_track
    return tracks
    pass


def volley_all_frames(data):
    frames = []
    for sid, anns in data.items():
        for fid, ann in anns.items():
            frames.append((sid, fid))
    return frames


class VolleyballDataset(data.Dataset):
    """
    Characterize volleyball dataset for pytorch
    """

    def __init__(self, anns, tracks, frames, images_path, image_size, feature_size, num_boxes=12, num_before=4,
                 num_after=4, is_training=True, is_finetune=False):
        self.anns = anns
        self.tracks = tracks
        self.frames = frames
        self.images_path = images_path
        self.image_size = image_size
        self.feature_size = feature_size

        self.num_boxes = num_boxes
        self.num_before = num_before
        self.num_after = num_after

        self.is_training = is_training
        self.is_finetune = is_finetune

    def __len__(self):
        """
        Return the total number of samples
        """
        return len(self.frames)

    def __getitem__(self, index):
        """
        Generate one sample of the dataset
        """
        sample = self.load_samples_sequence(self.frames[index])
        return sample

    def volley_frames_sample(self, src_fid):

        if self.is_finetune:
            if self.is_training:
                fid = random.randint(src_fid - self.num_before, src_fid + self.num_after)
                return [fid]
            else:
                return [fid for fid in range(src_fid - self.num_before, src_fid + self.num_after + 1)]
        else:
            sample_frames = range(src_fid - self.num_before, src_fid + self.num_after + 1)  #
            return [fid for fid in sample_frames]

    def get_location_change(self, boxes):
        location_change = []
        for i, location in enumerate(boxes[1:], start=1):
            temp_loc = []
            for j in range(self.num_boxes):
                pre_x = (boxes[i - 1][j][2] - boxes[i - 1][j][0] / 2)
                pre_y = (boxes[i - 1][j][3] - boxes[i - 1][j][1] / 2)
                cur_x = (boxes[i][j][2] - boxes[i][j][0] / 2)
                cur_y = (boxes[i][j][3] - boxes[i][j][1] / 2)
                temp_loc.append(np.sqrt((cur_x - pre_x) ** 2) + (cur_y - pre_y) ** 2)
            location_change.append(temp_loc)
        return np.stack(location_change)

    def load_samples_sequence(self, frames):
        """
        load samples sequence

        Returns:
            pytorch tensors
        """
        images, boxes = [], []
        activities, location_re = [], []

        sid, src_fid = frames
        select_frames = self.volley_frames_sample(src_fid)

        OH, OW = self.feature_size

        activities.append(self.anns[sid][src_fid]['group_activity'])

        for i, fid in enumerate(select_frames):

            img = Image.open(self.images_path + '/%d/%d/%d.jpg' % (sid, src_fid, fid))

            img = transforms.functional.resize(img, self.image_size)
            img = np.array(img)

            # H,W,3 -> 3,H,W
            img = img.transpose(2, 0, 1)
            images.append(img)

            temp_boxes = np.ones_like(self.tracks[(sid, src_fid)][fid])
            for i, track in enumerate(self.tracks[(sid, src_fid)][fid]):
                y1, x1, y2, x2 = track
                w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
                temp_boxes[i] = np.array([w1, h1, w2, h2])

            boxes.append(temp_boxes)

            if len(boxes[-1]) != self.num_boxes:
                boxes[-1] = np.vstack([boxes[-1], boxes[-1][:self.num_boxes - len(boxes[-1])]])

        images = np.stack(images)[1:, :, :]
        activities = np.array(activities, dtype=np.int32)
        location_re = self.get_location_change(boxes)
        bboxes = np.stack(boxes)[1:, :, :]

        # convert to pytorch tensor
        images = torch.from_numpy(images).float()
        bboxes = torch.from_numpy(bboxes).float()
        location_re = torch.from_numpy(location_re).float()
        activities = torch.from_numpy(activities).long()
        return images, bboxes, activities, location_re
