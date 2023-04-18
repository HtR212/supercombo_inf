import cv2
import json
import numpy as np
import onnxruntime
import pandas as pd
import math
import copy
import random

from matplotlib import pyplot as plt
from adam import AdamOpt

X_IDXS = np.array([ 0. ,   0.1875,   0.75  ,   1.6875,   3.    ,   4.6875,
         6.75  ,   9.1875,  12.    ,  15.1875,  18.75  ,  22.6875,
        27.    ,  31.6875,  36.75  ,  42.1875,  48.    ,  54.1875,
        60.75  ,  67.6875,  75.    ,  82.6875,  90.75  ,  99.1875,
       108.    , 117.1875, 126.75  , 136.6875, 147.    , 157.6875,
       168.75  , 180.1875, 192.])

def parse_image(frame):
	H = (frame.shape[0]*2)//3
	W = frame.shape[1]
	parsed = np.zeros((6, H//2, W//2), dtype=np.uint8)

	parsed[0] = frame[0:H:2, 0::2]
	parsed[1] = frame[1:H:2, 0::2]
	parsed[2] = frame[0:H:2, 1::2]
	parsed[3] = frame[1:H:2, 1::2]
	parsed[4] = frame[H:H+H//4].reshape((-1, H//2,W//2))
	parsed[5] = frame[H+H//4:H+H//2].reshape((-1, H//2,W//2))

	return parsed

def seperate_points_and_std_values(df):
	points = df.iloc[lambda x: x.index % 2 == 0]
	std = df.iloc[lambda x: x.index % 2 != 0]
	points = pd.concat([points], ignore_index = True)
	std = pd.concat([std], ignore_index = True)

	return points, std

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def main():
	model = "supercombo_server3.onnx"
	
	# cap = cv2.VideoCapture('data/cropped_mini.mp4')
	cap = cv2.VideoCapture('video.hevc')
	parsed_images = []

	width = 512
	height = 256
	dim = (width, height)
	
	plan_start_idx = 0
	plan_end_idx = 4955
	
	lanes_start_idx = plan_end_idx
	lanes_end_idx = lanes_start_idx + 528
	
	lane_lines_prob_start_idx = lanes_end_idx
	lane_lines_prob_end_idx = lane_lines_prob_start_idx + 8
	
	road_start_idx = lane_lines_prob_end_idx
	road_end_idx = road_start_idx + 264

	lead_start_idx = road_end_idx
	lead_end_idx = lead_start_idx + 255

	lead_prob_start_idx = lead_end_idx
	lead_prob_end_idx = lead_prob_start_idx + 3

	desire_start_idx = lead_prob_end_idx
	desire_end_idx = desire_start_idx + 8

	meta_start_idx = desire_end_idx
	meta_end_idx = meta_start_idx + 64

	pose_start_idx = meta_end_idx
	pose_end_idx = pose_start_idx + 12

	rnn_start_idx = pose_end_idx
	rnn_end_idx = rnn_start_idx + 512

# 	lead_start_idx = road_end_idx
# 	lead_end_idx = lead_start_idx + 55
# 	
# 	lead_prob_start_idx = lead_end_idx
# 	lead_prob_end_idx = lead_prob_start_idx + 3
# 	
# 	desire_start_idx = lead_prob_end_idx
# 	desire_end_idx = desire_start_idx + 72
# 	
# 	meta_start_idx = desire_end_idx
# 	meta_end_idx = meta_start_idx + 32
# 	
# 	desire_pred_start_idx = meta_end_idx
# 	desire_pred_end_idx = desire_pred_start_idx + 32
# 	
# 	pose_start_idx = desire_pred_end_idx
# 	pose_end_idx = pose_start_idx + 12
# 	
# 	rnn_start_idx = pose_end_idx
# 	rnn_end_idx = rnn_start_idx + 908

	imgs = []

	i = 0
	session = onnxruntime.InferenceSession(model, None)
	while(True):
		i += 1
		print(i)

		ret, frame = cap.read()
		if (ret == False):
			break

		if frame is not None:
			f = frame.copy()
			# for h in range(350,550):
			# 	for w in range(100,400):
			# 		f[h][w] = [0,0,0]
			cv2.imwrite("sample_original.png", f)
			imgs.append(frame)
			img = cv2.resize(frame, dim)
			img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
			parsed = parse_image(img_yuv)
	
		if (len(parsed_images) >= 2):
			del parsed_images[0]

		parsed_images.append(parsed)

		if (len(parsed_images) == 2):

			# optimization setup
			iter = 1000
			# height_p = len(imgs[0])
			# width_p = len(imgs[0][0])
			thres = 10

			prev_prob_avg = 0

			patch = None
			prev_patch = None
			patch_height = 200
			patch_width = 900
			# grad = np.zeros((height_p, width_p, 1)).astype('float64')
			grad = np.zeros((patch_height, patch_width, 1)).astype('float64')

			adam = AdamOpt(grad.shape, lr = 1)

			imgs[0] = imgs[0].astype('float64')
			imgs[1] = imgs[1].astype('float64')

			for it in range(iter):
				print("iter:", it)

				if it == 0:
					prev_patch = thres * np.random.rand(patch_height, patch_width, 1).astype('float64')
					patch = thres * np.random.rand(patch_height, patch_width, 1).astype('float64')
					tmp_imgs = copy.deepcopy(imgs)
					tmp_imgs[0][350:350+patch_height,100:100+patch_width] += np.repeat(prev_patch,3,axis=2)
					tmp_imgs[1][350:350+patch_height,100:100+patch_width] += np.repeat(prev_patch,3,axis=2)
					parsed_images[0] = parse_image(cv2.cvtColor(cv2.resize(np.clip(tmp_imgs[0],0,255).astype(np.uint8), dim), cv2.COLOR_BGR2YUV_I420))
					parsed_images[1] = parse_image(cv2.cvtColor(cv2.resize(np.clip(tmp_imgs[1],0,255).astype(np.uint8), dim), cv2.COLOR_BGR2YUV_I420))

				else:
					# add patch
					patch = np.clip(patch, -thres, thres)
					tmp_imgs = copy.deepcopy(imgs)
					tmp_imgs[0][350:350+patch_height,100:100+patch_width] += np.repeat(patch,3,axis=2)
					tmp_imgs[1][350:350+patch_height,100:100+patch_width] += np.repeat(patch,3,axis=2)
					parsed_images[0] = parse_image(cv2.cvtColor(cv2.resize(np.clip(tmp_imgs[0],0,255).astype(np.uint8), dim), cv2.COLOR_BGR2YUV_I420))
					parsed_images[1] = parse_image(cv2.cvtColor(cv2.resize(np.clip(tmp_imgs[1],0,255).astype(np.uint8), dim), cv2.COLOR_BGR2YUV_I420))

					if it%10 == 0:
						cv2.imwrite("sample_original.png", np.clip(imgs[1],0,255).astype(np.uint8))
						cv2.imwrite("sample_with_patch.png", np.clip(tmp_imgs[1],0,255).astype(np.uint8))

					# img_with_patch = np.clip(imgs[0]+patch,0,255).astype(np.uint8)
					# cv2.imshow('img_with_patch', img_with_patch)

					# if cv2.waitKey(1) & 0xFF == ord('q'):
					# 	break

				parsed_arr = np.array(parsed_images)
				parsed_arr.resize((1,12,128,256))

				data = json.dumps({'data': parsed_arr.tolist()})
				data = np.array(json.loads(data)['data']).astype('float32')
				
				input_imgs = session.get_inputs()[0].name
				desire = session.get_inputs()[1].name
				initial_state = session.get_inputs()[2].name
				traffic_convention = session.get_inputs()[3].name
				output_name = session.get_outputs()[0].name
				
				desire_data = np.array([0]).astype('float32')
				desire_data.resize((1,8))
				
				traffic_convention_data = np.array([0]).astype('float32')
				traffic_convention_data.resize((1,512))
				
				initial_state_data = np.array([0]).astype('float32')
				initial_state_data.resize((1,2))

				result = session.run([output_name], {input_imgs: data,
													desire: desire_data,
													traffic_convention: traffic_convention_data,
													initial_state: initial_state_data
													})

				res = np.array(result)

				plan = res[:,:,plan_start_idx:plan_end_idx]
				lanes = res[:,:,lanes_start_idx:lanes_end_idx]
				lane_lines_prob = res[:,:,lane_lines_prob_start_idx:lane_lines_prob_end_idx]
				lane_road = res[:,:,road_start_idx:road_end_idx]
				lead = res[:,:,lead_start_idx:lead_end_idx]
				lead_prob = res[:,:,lead_prob_start_idx:lead_prob_end_idx]
				desire_state = res[:,:,desire_start_idx:desire_end_idx]
				meta = res[:,:,meta_start_idx:meta_end_idx]
				# desire_pred = res[:,:,desire_pred_start_idx:desire_pred_end_idx]
				pose = res[:,:,pose_start_idx:pose_end_idx]
				recurrent_layer = res[:,:,rnn_start_idx:rnn_end_idx]

				lead_prob = lead_prob.flatten()
				lead_prob = [sigmoid(lead_prob[0]), sigmoid(lead_prob[1]), sigmoid(lead_prob[2])]
				print(lead_prob)

				lead = lead.flatten()
				speed = lead[0:24]
				accel = lead[24:48]
				prob_024 = lead[48:51]
				print(speed)
				print(accel)
				print(sigmoid(prob_024[0]), sigmoid(prob_024[1]), sigmoid(prob_024[2]))
				print()

				if it == 0:
					prev_prob_avg = sum(lead_prob)/3					

				else:
					prob_avg = sum(lead_prob)/3
					diff_prob = prob_avg - prev_prob_avg
					prev_prob_avg = prob_avg
					zero_diff_mask = np.zeros_like(patch)

					for h in range(patch_height):
						for w in range(patch_width):
							for c in range(1):
								diff_patch_pixel = patch[h][w][c] - prev_patch[h][w][c]
								if diff_patch_pixel == 0:
									zero_diff_mask[h][w][c] += 0.1
									grad[h][w][c] = 0
								else:
									zero_diff_mask[h][w][c] = 0
									grad[h][w][c] = diff_prob/diff_patch_pixel
					
					# update patch
					prev_patch = patch
					patch -= adam.update(grad)

					# deal with unchanged pixels
					patch += zero_diff_mask * random.choice([-1,1])

			break

# 			lanes_flat = lanes.flatten()
# 			df_lanes = pd.DataFrame(lanes_flat)

# 			ll_t = df_lanes[0:66]
# 			ll_t2 = df_lanes[66:132]
# 			points_ll_t, std_ll_t = seperate_points_and_std_values(ll_t)
# 			points_ll_t2, std_ll_t2 = seperate_points_and_std_values(ll_t2)

# 			l_t = df_lanes[132:198]
# 			l_t2 = df_lanes[198:264]
# 			points_l_t, std_l_t = seperate_points_and_std_values(l_t)
# 			points_l_t2, std_l_t2 = seperate_points_and_std_values(l_t2)

# 			r_t = df_lanes[264:330]
# 			r_t2 = df_lanes[330:396]
# 			points_r_t, std_r_t = seperate_points_and_std_values(r_t)
# 			points_r_t2, std_r_t2 = seperate_points_and_std_values(r_t2)

# 			rr_t = df_lanes[396:462]
# 			rr_t2 = df_lanes[462:528]
# 			points_rr_t, std_rr_t = seperate_points_and_std_values(rr_t)
# 			points_rr_t2, std_rr_t2 = seperate_points_and_std_values(rr_t2)

# 			road_flat = lane_road.flatten()
# 			df_road = pd.DataFrame(road_flat)

# 			roadr_t = df_road[0:66]
# 			roadr_t2 = df_road[66:132]
# 			points_road_t, std_ll_t = seperate_points_and_std_values(roadr_t)
# 			points_road_t2, std_ll_t2 = seperate_points_and_std_values(roadr_t2)

# 			roadl_t = df_road[132:198]
# 			roadl_t2 = df_road[198:264]
# 			points_roadl_t, std_rl_t = seperate_points_and_std_values(roadl_t)
# 			points_roadl_t2, std_rl_t2 = seperate_points_and_std_values(roadl_t2)

# 			middle = points_ll_t2.add(points_l_t, fill_value=0) / 2

# 			plt.scatter(middle, X_IDXS, color = "g")

# # 			plt.scatter(points_ll_t, X_IDXS, color = "b", marker = "*")
# 			plt.scatter(points_ll_t2, X_IDXS, color = "y")

# 			plt.scatter(points_l_t, X_IDXS, color = "y")
# # 			plt.scatter(points_l_t2, X_IDXS, color = "y", marker = "*")

# 			plt.scatter(points_road_t, X_IDXS, color = "r")
# 			plt.scatter(points_road_t2, X_IDXS, color = "r")

# 			plt.title("Raod lines")
# 			plt.xlabel("red - road lines | green - predicted path | yellow - lane lines")
# 			plt.ylabel("Range")
# 			# plt.show()
# 			# plt.savefig("results/sample_{}.png".format(i))
# 			plt.savefig("test_hevc.png")
# 			# plt.pause(0.1)
# 			plt.clf()

		# frame = cv2.resize(frame, (900, 500))
		# cv2.imshow('frame', frame)

		# if cv2.waitKey(1) & 0xFF == ord('q'):
		# 	break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
