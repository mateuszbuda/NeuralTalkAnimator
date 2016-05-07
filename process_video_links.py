__author__ = 'mateusz'

from pytube import YouTube
import argparse
import subprocess
import os

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--debug', help='In debug mode additional html file with overlaid images is created', type=int, default=0)

	args = parser.parse_args()
	debug = args.debug
	videos = './videos/'

	with open('videos/video_links.txt') as f:
		links = f.readlines()

	for link in links:
		try:
			yt = YouTube(link)
		except Exception as e:
			print e
			continue

		yt.set_filename(link.rstrip().split('=')[1])

		video = yt.filter('mp4')[-1]
		video.download(videos)

		command = 'python extract.py -d ' + str(debug)
		print(command)
		proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,
							universal_newlines=False)

		while proc.poll() is None:
			line = proc.stdout.readline()
			print(line)

		try:
			os.remove(videos + yt.filename + '.mp4')
		except OSError, e:
			print(e)
