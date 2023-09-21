"""
Read the log file and draw the loss curve.
"""

import re
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt


def draw_one_curve(log_file, label, color, line_type):
    """Draw one curve."""
    """
    loss_mask: 0.637474 loss_rpn_cls: 0.055456 loss_rpn_reg: 0.069791 loss_bbox_cls: 0.068547 loss_bbox_reg: 0.090937 loss: 0.922204
    """
    loss = []
    loss_mask = []
    loss_rpn_cls = []
    loss_rpn_reg = []
    loss_bbox_cls = []
    loss_bbox_reg = []

    with open(log_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if 'loss:' not in line:
                continue
            # 正则匹配出 loss_mask、loss_rpn_cls、loss_rpn_reg、loss_bbox_cls、loss_bbox_reg、loss
            loss_mask.append(float(re.findall(r'loss_mask: (\d+\.\d+)', line)[0]))
            loss_rpn_cls.append(float(re.findall(r'loss_rpn_cls: (\d+\.\d+)', line)[0]))
            loss_rpn_reg.append(float(re.findall(r'loss_rpn_reg: (\d+\.\d+)', line)[0]))
            loss_bbox_cls.append(float(re.findall(r'loss_bbox_cls: (\d+\.\d+)', line)[0]))
            loss_bbox_reg.append(float(re.findall(r'loss_bbox_reg: (\d+\.\d+)', line)[0]))
            loss.append(float(re.findall(r'loss: (\d+\.\d+)', line)[0]))
    plt.plot(loss, label=label, color=color, linestyle=line_type)

if __name__ == '__main__':
    folder_list = ['baseline', 'amp_and_nhwc']
    color_list = ['red', 'blue']
    line_type_list = ['-', '--']
    log_file_list = [f'{item}/{item}_train.log' for item in folder_list]
    
    for i in range(len(folder_list)):
        draw_one_curve(log_file_list[i], folder_list[i], color_list[i], line_type_list[i])
    plt.legend()
    plt.show()