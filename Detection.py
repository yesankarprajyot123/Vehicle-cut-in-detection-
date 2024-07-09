import cv2
from ultralytics import YOLO
import numpy as np
import statistics
import os
import numpy as np
import time
import sys

model = YOLO("model/best.pt")

def get_red_green_color(value):
	inverted_value = 1 - value
	blue = 0
	red = int(255 * inverted_value)
	green = int(255 * value)

	return (blue, green, red)

def normalize_value(value, min_val=0.5, max_val=2):
	return (value - min_val) / (max_val - min_val)

def find_distance(height_px, vehicle_class):
	real_height = {0: 1000, 1:1600, 2:4000} # In mm
	sensor_height = 3.02
	focal_length = 2.12
	frame_height = 1080
	distance = (focal_length*real_height[int(vehicle_class)]*frame_height)/(height_px*sensor_height)
	return distance/1000

try:
	image_directory = sys.argv[1]
except IndexError:
	print("Please provide path to image_directory")
	print("\nUsage: python cut_in_detection.py <image_directory_name>")
	print("Example: python cut_in_detection.py ./images")
	sys.exit(1)

if image_directory[-1] == '/':
	image_directory = image_directory[:-1]

files = os.listdir(image_directory)


# Frame dimensions and lane markings
frame_width = 1920
frame_height = 1080
center_frame_w = frame_width/2 - 80
center_frame_h = frame_height/2
left_lane_p1 = (int(center_frame_w - frame_width*0.33), frame_height)
left_lane_p2 = (int(center_frame_w - frame_width*0.02), int(frame_height*0.5))
right_lane_p1 = (int(center_frame_w + frame_width*0.33), frame_height)
right_lane_p2 = (int(center_frame_w + frame_width*0.02), int(frame_height*0.5))
slope_l1 = abs((left_lane_p2[1]-left_lane_p1[1])/(left_lane_p2[0]-left_lane_p1[0]))
slope_l2 = abs((right_lane_p2[1]-right_lane_p1[1])/(right_lane_p2[0]-right_lane_p1[0]))

distance = dict()
vel_angle = dict()
warnings = dict()

cv2.namedWindow('Cut-In Detection', cv2.WINDOW_NORMAL)
# cv2.setWindowProperty('Cut-In Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Use slicing files[:] to loop through small batch of files
for file in files:
	start_time = time.time()
	frame = cv2.imread(f'{image_directory}/{file}')
	resized_img = cv2.resize(frame, (640, 384))
	try:
		results = model.track(resized_img, conf=0.70, persist=True, iou=0.90)
	except np.linalg.LinAlgError:
		continue
	
	cv2.line(frame, left_lane_p1, left_lane_p2, (255, 255, 255), 2, cv2.LINE_AA)
	cv2.line(frame, right_lane_p1, right_lane_p2, (255, 255, 255), 2, cv2.LINE_AA)

	for r in results:
		bounding = r.boxes.xywhn # Coordinates of bounding boxes # Tensor
		bounding[:,0] *= frame_width
		bounding[:,2] *= frame_width
		bounding[:,1] *= frame_height
		bounding[:,3] *= frame_height

		box = r.boxes.id # Tensor
		vehicle_class = r.boxes.cls

		# If no object is detected clear up distance and vel_angle
		if box == None:
			distance = dict()
			vel_angle = dict()
			continue

		temp = list(distance.keys())

		for key in temp:
			if key not in box:
				distance.pop(key)
				if key in warnings:
					del warnings[key]

		for j in range(bounding.shape[0]):
			x, y, w, h = bounding[j,:]

			# Initialize distance of object
			if int(box[j]) not in distance.keys():
				distance[int(box[j])] = [0, 0]
				distance[int(box[j])][1] = float(find_distance(h, vehicle_class[j]))-1.5
				continue
			else:
				distance[int(box[j])][0] = distance[int(box[j])][1]
				distance[int(box[j])][1] = float(find_distance(h, vehicle_class[j]))-1.5


				v = abs(distance[int(box[j])][1]-distance[int(box[j])][0])*15 # Time interval is 1/15, so distance/time_interval


				if x+w/2<center_frame_w: # Find difference between consecutive angles to determine if a car is cutting in
					m = (y-(h/2)-left_lane_p1[1])/(x+(w/2)-left_lane_p1[0])
					angle = float(((np.arctan(abs(m))-(np.arctan(slope_l1)))*180/np.pi))
				else:
					m = (y-(h/2)-right_lane_p1[1])/(x-(w/2)-right_lane_p1[0])
					angle = float(((np.arctan(abs(m))-(np.arctan(slope_l2)))*180/np.pi))

				# Initialize angle with the reference line
				if int(box[j]) not in vel_angle.keys():
					vel_angle[int(box[j])] = list()
					vel_angle[int(box[j])].append([v])
					vel_angle[int(box[j])].append([angle])

				else:
					# Using previous 5 frames to determine ideal velocity and angle_diff values
					if len(vel_angle[int(box[j])][0]) == 5:
						vel = statistics.median(vel_angle[int(box[j])][0])
						angle_stdev = statistics.stdev(vel_angle[int(box[j])][1])
						ttc = distance[int(box[j])][1]/vel
						
						angle_diff = vel_angle[int(box[j])][1][-1]-vel_angle[int(box[j])][1][-2]

						vel_angle[int(box[j])][0].pop(0) # Remove the first stored value in the velocity array, i.e. the value at index 0
						vel_angle[int(box[j])][1].pop(0) # Remove the first stored value in the angle array, i.e. the value at index 0

						if ttc<3 and angle_diff<0 and angle_stdev > 0.5 and angle<5:
							if ttc<0.8 and angle_stdev > 1.5:
								warnings[int(box[j])] = [int(x), int(y), int(w), int(h), ttc, 30, 1]
							elif int(box[j]) in warnings.keys() and warnings[int(box[j])][6] == 1:
								warnings[int(box[j])] = [int(x), int(y), int(w), int(h), ttc, 30, 1]
							elif distance[int(box[j])][1] < 5:
								warnings[int(box[j])] = [int(x), int(y), int(w), int(h), ttc, 30, 0]

						elif int(box[j]) in warnings.keys():
							warnings[int(box[j])][0:5] = [int(x), int(y), int(w), int(h), ttc]

					vel_angle[int(box[j])][0].append(v)
					vel_angle[int(box[j])][1].append(angle)


	# Print the cut-in warning and track vehicle for next 30 frames
	for key, value in warnings.items():
		font = cv2.FONT_HERSHEY_SIMPLEX
		if value[5] > 0:
			text = 'Cut-In'
			textsize = cv2.getTextSize(text, font, 1, 2)[0]
			textX = int(value[0] - (textsize[0] / 2))
			textY = int(value[1] + (textsize[1] / 2))

			if value[6] == 1:
				cv2.putText(frame, f'Cut-In', (textX, textY), font, 1, (0,200,255), 2, cv2.LINE_AA)
			
			pts = np.array([[value[0], value[1]+(value[3]/2)], [value[0]-25, value[1]+(value[3]/2)+20], [value[0]+25, value[1]+(value[3]/2)+20]], np.int32)
			cv2.fillPoly(frame, [pts], get_red_green_color(normalize_value(value[4])))
			value[5] -= 1
		else:
			del warnings[key]
			break

	cv2.putText(frame, f"FPS: {int(1.0 / (time.time() - start_time))}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
	cv2.putText(frame, "Press q to exit", (int(frame_width-250), 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
	cv2.imshow("Cut-In Detection", frame)

	# Press q to exit
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	if cv2.getWindowProperty('Cut-In Detection', cv2.WND_PROP_VISIBLE) < 1:
		break

cv2.destroyAllWindows()
