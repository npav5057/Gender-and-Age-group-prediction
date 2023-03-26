#Preprocess image which include precise aligning and Cropping
# in last also copy all five fold text files
import cv2
import numpy as np
from PIL import Image
import dlib
import os
import shutil

IMG_SIZE=227

def distance(a, b):
	return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def shape_to_normal(shape):
	shape_normal = []
	for i in range(0, 5):
		shape_normal.append((shape.part(i).x, shape.part(i).y))
	return shape_normal

def angle_opposite_to_line3(length_line1, length_line2, length_line3):
	cos_value = (length_line1**2 + length_line2**2 - length_line3**2) / (2*length_line2*length_line1)
	return np.arccos(cos_value)

def align_and_resize_image(img):
	##### align image ############################
	predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
	detector = dlib.get_frontal_face_detector()

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	rects = detector(img, 0)

	if(len(rects)==0):
		return -1

	rect_idx=-1
	max_area=0
	for i in range(len(rects)):
		x1 = rects[i].left()
		y1 = rects[i].top()
		x2 = rects[i].right()
		y2 = rects[i].bottom()
		if(max_area<(abs(x1-x2)*abs(y1-y2))):
			max_area=(abs(x1-x2)*abs(y1-y2))
			rect_idx=i
		# img = cv2.circle(img, (x1,y1), 3, (0,0,255), 3)
		# img = cv2.circle(img, (x2,y2), 3, (0,0,255), 3)
		# cv2.imshow("window_name", img)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

	shape = predictor(gray, rects[rect_idx])
	shape = shape_to_normal(shape)
	# shape has 5 point 4 nose 2 3 left eye 0 1 right eye
	# https://www.pyimagesearch.com/2018/04/02/faster-facial-landmark-detector-with-dlib/
	nose = shape[4]
	left_eye = [int(shape[2][0]+shape[3][0])/2,int(shape[2][1]+shape[3][1])/2]
	right_eye = [int(shape[0][0]+shape[1][0])/2,int(shape[0][1]+shape[1][1])/2]

	center_of_forehead = ((left_eye[0] + right_eye[0])/2, (left_eye[1] + right_eye[1])/2)

	center_image_top = [nose[0],0]
	length_line1 = distance(center_of_forehead, nose)
	length_line2 = distance(center_image_top, nose)
	length_line3 = distance(center_image_top, center_of_forehead)

	angle = abs(angle_opposite_to_line3(length_line1, length_line2, length_line3))
	if center_of_forehead[0]<=nose[0]:
		angle = -1*(np.degrees(angle))
	else:
		angle = np.degrees(angle)

	# print("angle is ",angle)
	x1 = rects[rect_idx].left()
	y1 = rects[rect_idx].top()
	x2 = rects[rect_idx].right()
	y2 = rects[rect_idx].bottom()
	img=img[y1:y2, x1:x2, :]
	try:
		img = Image.fromarray(img)
		img = np.array(img.rotate(angle))
		# ########################### Cropping of aligned image ###################################
		img=cv2.resize(img, (IMG_SIZE, IMG_SIZE))
		# print("===crpoing=",img.shape)
		return img
	except Exception as e:
		return -1



# img=cv2.imread("i.jpg")
# img=align_and_resize_image(img)
# cv2.imshow("window_name", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

shutil.rmtree('copy')
if not os.path.isdir( 'copy' ) :
	os.mkdir("copy")
if not os.path.isdir( 'copy/faces' ) :
	os.mkdir("copy/faces")

count_good=0
count_bad=0
for i in [0,1,2,3,4]:
	l=[]
	text_file_name="./data/fold_"+str(i)+"_data.txt"
	text_file = open(text_file_name, "r")
	lines = text_file.readlines()
	for line in lines:
		ls=line.split("\t")
		file_name="./data/faces/"+ls[0]+'/coarse_tilt_aligned_face.'+ls[2]+"."+ls[1]
		if not os.path.isdir( 'copy/faces/'+ls[0] ) :
			os.mkdir("copy/faces/"+ls[0])
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
		valid2=["m","f","u"] #earlier it was valid2=["m","f","u"]
		for i in range(len(valid2)):
			if valid2[i]==ls[4]:
				valid_gender=i

		if not (valid_age==-1 or valid_gender==-1):
			img = cv2.imread(file_name)
			img=align_and_resize_image(img)
			try:
				dummy=img[0][0]
				file_name=file_name.replace("data", "copy")
				cv2.imwrite(file_name, img)
				count_good+=1
				print("GOOD COUNT => ",count_good)
			except Exception as e:
				count_bad+=1
				print(img,"BAD COUNT => ",count_bad)

	text_file.close()

#now copy all five fold text files
