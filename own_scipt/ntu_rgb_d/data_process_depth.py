import os
import sys
import cv2
import numpy as np
import shutil

def process(path):
    sample_info_file = os.path.join(path, 'rgb_sample_info.txt')
    target_depth = os.path.join(path, 'depth_sample')
    if not os.path.exists(target_depth):
        os.makedirs(target_depth)
    sample_info_list = []
    with open(sample_info_file, 'r') as f:
        for line in f:
            info = line.strip().split('\t')
            if len(info) != 2:
                continue
            sample_info_list.append(info)
    
    depth_path = os.path.join(path, 'depth', 'nturgb+d_depth_masked')
    for sample in sample_info_list:
        source_path = os.path.join(depth_path, sample[0].split('_')[0])
        target_path_sample = ''
        if sample[1] == '1':
            target_path_sample = os.path.join(target_depth, sample[0].split('_')[0]  + '_depth_fall')
        else:
            target_path_sample = os.path.join(target_depth, sample[0].split('_')[0]  + '_depth_adl')
        if not os.path.exists(target_path_sample):
            os.makedirs(target_path_sample)
        
        for file in os.listdir(source_path):

            source_file = os.path.join(source_path, file)
            if not os.path.exists(source_file):
                continue

            target_flie = os.path.join(target_path_sample, 'img_{:04}.png'.format(int(file.split('.')[0].split('-')[1])))
            shutil.copyfile(source_file, target_flie)


def main(argv):
	for dirs in argv[1:]:
		process(dirs)

if __name__=='__main__':
	argv = sys.argv
	main(argv)
