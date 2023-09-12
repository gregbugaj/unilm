import argparse
import io
import json
import multiprocessing as mp
import numpy as np
import copy

from concurrent.futures.thread import ThreadPoolExecutor
from functools import lru_cache
from multiprocessing import Pool


import json
import sys

'''
Usage: just type in order the .json files that you want to merge.
Eg. $python3 coco1.json coco2.json coco3.json outputfile.json
'''

#global_table
categoryIdTable = dict()	#category ids are taken from the first file.
currentCatId = 1


class cocoFile():
	def __init__(self, path):
		self.imageIdTable = dict()
		self.annIdTable = dict()
		self.catIdTable = dict()
		self.file = open(path)
		self.file = json.load(self.file)
		print(path)

	def updateImageTable(self, start_index):
		print("image Table,", start_index)
		for i in range(len(self.file['images'])):
			self.imageIdTable[int(self.file['images'][i]['id'])] = start_index + i
			self.file['images'][i]['id'] = start_index + i
		return start_index + i + 1
	
	def updateAnnTable(self, start_index):
		print("ann table,",start_index)
		for i in range(len(self.file['annotations'])):
			self.file['annotations'][i]['id'] = start_index + i
			#print(self.file['annotations'][i]['id'])
			self.file['annotations'][i]['image_id'] = self.imageIdTable[int(self.file['annotations'][i]['image_id'])]
			self.file['annotations'][i]['category_id'] = self.catIdTable[self.file['annotations'][i]['category_id']]
		return start_index + i + 1
	
	def updateCategoryIds(self, categoryIdTable, currentCatId):
		#first, update the category id table
		for i in range(len(self.file['categories'])):
			if not self.file['categories'][i]['name'] in categoryIdTable:
				categoryIdTable[self.file['categories'][i]['name']] = currentCatId
				currentCatId += 1
			
			#now updating khud ka category id table:
			self.catIdTable[self.file['categories'][i]['id']] = categoryIdTable[self.file['categories'][i]['name']]

		return categoryIdTable, currentCatId


input_files = []
for i in range(1, len(sys.argv) - 1):
	input_files.append(sys.argv[i])

print("input files:", input_files)
output_file = sys.argv[-1]

print("output file:", output_file)
cocoFiles = []
start_index_imageid = 0
start_index_annid = 0

for i in range(len(input_files)):
	c = cocoFile(input_files[i])
	print("processing file:", input_files[i])
	print("\tstart_index_imageid:", start_index_imageid)
	print("\tstart_index_annid:", start_index_annid)
	
	start_index_imageid = c.updateImageTable(start_index_imageid)
	categoryIdTable, currentCatId = c.updateCategoryIds(categoryIdTable, currentCatId)
	start_index_annid = c.updateAnnTable(start_index_annid)
	print("done processing file:", input_files[i])	
	cocoFiles.append(c)
	
main = cocoFiles[0].file

for cocoFile in cocoFiles[1:]:
    f = cocoFile.file
    
    for i in f['images']:
        main['images'].append(i)
    
    for i in f['annotations']:
        main['annotations'].append(i)

#Saving this to outputfile:
with open(output_file, 'w') as outfile:
    json.dump(main, outfile)
	
# python ./coco_dataset_merger.py /home/greg/datasets/funds_dit/raw/job_2-2023_09_09_05_32_27-coco_1.0/annotations/instances_default.json /home/greg/datasets/funds_dit/raw/job_3-2023_09_11_22_04_35-coco_1.0/annotations/instances_default.json /tmp/instances_default_merged.json


if False:
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Converts COCO annotations to DIT format")
        parser.add_argument(
            "--coco_annoations_file",
            type=str,
            help="Path to the COCO annotations file",
            required=True,
        )

        args = parser.parse_args()

        process(args.coco_annoations_file)
