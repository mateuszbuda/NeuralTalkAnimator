#!/usr/bin/python
__author__ = 'Samim.io'

import argparse
import os
import subprocess
from os import listdir
from os.path import isfile, join
import json
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


def drawOverlay(image, text):
	print 'drawOverlay: ' + image + ' text: ' + text
	img = Image.open(image)
	draw = ImageDraw.Draw(img)
	font = ImageFont.truetype("font.ttf", 46)
	draw.text((20, 10), text, (255, 255, 255), font=font)
	img.save(image)


def createImageOverlay(inputdir):
	print '(5/5) createImageOverlay: ' + inputdir
	foundimages = []
	with open(inputdir + '/result_struct.json') as data_file:
		data = json.load(data_file)
		images = data['imgblobs']
		for image in images:
			imgpath = image['img_path']
			imgtext = image['candidate']['text']
			newImage = [imgpath, imgtext]
			foundimages.append(newImage)

	print 'Creating Image Overlay'

	for image in foundimages:
		drawOverlay(inputdir + '/' + image[0], image[1])

	frames = [f for f in listdir(inputdir) if isfile(join(inputdir, f))]
	captionedFrames = [image[0] for image in foundimages]

	for frame in frames:
		if frame.endswith('.jpg') or frame.endswith('.png'):
			if frame in captionedFrames:
				continue
			else:
				try:
					os.remove(inputdir + '/' + frame)
				except OSError, e:
					print(e)


def getImageSentence(inputdir, framerate, debug):
	print '(4/5) getImageSentence: ' + inputdir
	command = 'python predict_on_images.py cv/model_checkpoint_coco_visionlab43.stanford.edu_lstm_11.14.p -r ' + inputdir + ' -f ' + framerate + ' -d ' + str(debug)
	print(command)
	proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,
							universal_newlines=True)
	while proc.poll() is None:
		line = proc.stdout.readline()
		print(line)

	createImageOverlay(inputdir)


def getImageFeatures(inputdir, framerate, debug):
	print '(3/5) getImageFeatures: ' + inputdir
	command = 'python python_features/extract_features.py --caffe /caffe --model_def python_features/deploy_features.prototxt --model python_features/VGG_ILSVRC_16_layers.caffemodel --files ' + inputdir + '/tasks.txt --out ' + inputdir + '/features'
	print(command)
	proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,
							universal_newlines=True)
	while proc.poll() is None:
		line = proc.stdout.readline()
		print(line)

	getImageSentence(inputdir, framerate, debug)


def addToList(inputdir, frameFreq, framerate, debug):
	print '(2/5) addToList: ' + inputdir
	frames = [f for f in listdir(inputdir) if isfile(join(inputdir, f))]
	counter = frameFreq
	open(inputdir + '/tasks.txt', 'w').close()
	for frame in frames:
		if frame.endswith('.jpg') or frame.endswith('.png'):
			if counter >= frameFreq:
				# print frame
				with open(inputdir + '/tasks.txt', 'a') as textfile:
					textfile.write(frame + '\n')
				counter = 0
			counter += 1

	getImageFeatures(inputdir, framerate, debug)


def extractVideo(inputdir, outputdir, framefreq, debug):
	if not os.path.exists(outputdir):
		os.makedirs(outputdir)
	print('(1/5) extractVideo: ' + inputdir + ' To: ' + outputdir)

	# get framerate
	command = 'ffmpeg -i ' + inputdir + ' 2>&1 | sed -n "s/.*, \(.*\) fp.*/\\1/p"'
	print(command)
	framerate = '24'
	proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,
							universal_newlines=True)
	while proc.poll() is None:
		line = proc.stdout.readline()
		if len(line) > 1:
			framerate = line.rstrip('\n')

	print 'framerate: ' + framerate

	# get video
	command = 'ffmpeg -i ' + inputdir + ' -framerate ' + str(
		framerate) + ' -y -f image2 ' + outputdir + '/frame-%06d.jpg'
	print(command)
	proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,
							universal_newlines=True)
	while proc.poll() is None:
		line = proc.stdout.readline()
		# print(line + '\n')

		print('extracting audio: ' + inputdir + ' To: ' + outputdir)

		# get audio
		command = 'ffmpeg -i ' + inputdir + ' -y -map 0:1 -vn -acodec copy ' + outputdir + '/output.aac'
		print(command)
		proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,
								universal_newlines=True)

		while proc.poll() is None:
			line = proc.stdout.readline()

	addToList(outputdir, framefreq, framerate, debug)


def cleanup(inputdir):
	files = [f for f in listdir(inputdir) if isfile(join(inputdir, f))]

	for file in files:
		if file != 'result_struct.json':
			try:
				os.remove(inputdir + '/' + file)
			except OSError, e:
				print(e)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='VideoCaptionGenerator')
	parser.add_argument('-f', '--captionfrequency', help='Caption Creation Frequency Per Frame.', type=int, default=120)
	parser.add_argument('-d', '--debug', help='In debug mode additional html file with overlaid images is created', type=int, default=0)

	args = parser.parse_args()

	debug = args.debug
	captionfrequency = args.captionfrequency

	print '***************************************'
	print '******** GENERATING CAPTIONS **********'
	print '***************************************'
	print 'captionfrequency: ' + str(captionfrequency)

	mypath = 'videos/'
	foldername = ''

	videos = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	for video in videos:
		if video.endswith('.mp4') or video.endswith('.mov') or video.endswith('.avi'):
			print 'Processing: ' + video
			foldername = os.path.splitext(video)[0]
			extractVideo(mypath + video, mypath + foldername, captionfrequency, debug)

	if debug == 0:
		cleanup(mypath + foldername)

	print ''
	print '********* PROCESSED ALL ************'
	print ''
