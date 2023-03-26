# Divide data for chained model
import numpy as np
import cv2
import sys
import os
import shutil
IMG_SIZE=100

list1=["0-2","4-6","8-13","15-20","25-32","38-43","48-53","60-"]
list2=["male","female","child"]

for i in list2:
	if os.path.isdir(i):
		shutil.rmtree(i)

	os.mkdir(i)

	for j in list1:

		folder_name=i+"/train_data/"+j
		if not os.path.exists(folder_name):
			os.makedirs(folder_name)

		folder_name=i+"/validation_and_test_data/"+j
		if not os.path.exists(folder_name):
			os.makedirs(folder_name)


def divide(P,x):
	train_X=[]
	train_Y_AGE=[]
	train_Y_GENDER=[]
	text_file_name="data/fold_"+str(x)+"_data.txt"
	text_file = open(text_file_name, "r")
	lines = text_file.readlines()
	for line in lines:
		ls=line.split("\t")
		file_name='coarse_tilt_aligned_face.'+ls[2]+"."+ls[1]
		temp=ls[3].replace(" ","")[1:-1].split(",")
		valid_age=-1
		# 0-2, 4-6, 8-13, 15-20, 25-32, 38-43, 48-53, 60-
		l_Age_1=[0,4,8,15,25,38,48,60]
		l_Age_2=[2,6,13,20,32,43,53,100]
		for i in range(8):
			if ls[3].isdigit() and (int(ls[3])>=l_Age_1[i] and int(ls[3])<=l_Age_2[i]):
				valid_age=i
			elif temp[0].isdigit() and l_Age_1[i]==int(temp[0]):
				valid_age=i
			elif (temp[0].isdigit() and temp[1].isdigit()):
				avg=(int(temp[0])+int(temp[1]))/2.0
				if(avg>=l_Age_1[i] and avg<=l_Age_2[i]):
					valid_age=i

		valid_gender=-1
		valid2=["m","f","u"]
		for i in range(len(valid2)):
			if valid2[i]==ls[4]:
				valid_gender=i

		img = cv2.imread("data/faces/"+ls[0]+"/"+file_name)
		if valid_gender!=-1 and valid_age!=-1 and (img is not None) and img.shape[0]==227 and img.shape[1]==227:
			path=list2[valid_gender]+P+list1[valid_age]+"/"+file_name
			# print(path)
			cv2.imwrite(path,img)

	text_file.close()

for fold_id in range(4):#5th one is for testng porpose
	print("processing training fold ... ",fold_id)
	divide("/train_data/",fold_id)

print("processing validation and testing fold ... 4")
divide("/validation_and_test_data/",4)
