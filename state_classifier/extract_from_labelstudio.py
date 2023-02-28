import collections
import shutil

import cv2
# Import the SDK and the client module
import json
import os

# Connect to the Label Studio API and check the connection
# todo: parameterize for 4 tasks, changing file i/o (anything else?)
# access original data from labeled examples (split video name to start at user)
import numpy as np
import tqdm
from keras.utils import to_categorical

AGENT_RESOLUTION = (128, 128)
def resize_image(img, target_resolution):
    # For your sanity, do not resize with any function than INTER_LINEAR
    img = cv2.resize(img, target_resolution, interpolation=cv2.INTER_LINEAR)
    return img

def process_labels(label_filepath, task_name):
    print(f"Processing {task_name}")
    with open(label_filepath, 'r') as labels_file:
        house_labels = json.load(labels_file)

    videos_directory = f"data/{task_name}"
    labels_directory = f"data/labels/{task_name}"
    if os.path.exists(labels_directory):
        shutil.rmtree(labels_directory)
    os.makedirs(labels_directory)

    video_id_to_video_filepaths = collections.defaultdict(list)
    for filename in os.listdir(videos_directory):
        if filename.endswith(".mp4"):
            video_id = "-".join(filename.split("-")[0:4])
            video_id_to_video_filepaths[video_id].append(filename)
            video_id_to_video_filepaths[video_id] = sorted(video_id_to_video_filepaths[video_id])

    # Example video name: cheeky-cornflower-setter-0a9ad3ddd136-20220726-193610.mp4
    # Upload name: bbb54b05-woozy-ruby-ostrich-0e36ad30dada-20220725-144831.mp4
    per_video_per_frame_labels = []

    for video_label in house_labels:
        video_id = video_label["id"]
        original_video_filename = "-".join(video_label["file_upload"].split("-")[1:])
        original_video_filepath = os.path.join(videos_directory, original_video_filename)

        vidcap = cv2.VideoCapture(original_video_filepath)
        success, image = vidcap.read()
        video_fps = vidcap.get(cv2.CAP_PROP_FPS)

        per_frame_labels = []

        good_trial = True

        annotations = video_label["annotations"]
        for annotation in annotations:
            results = annotation["result"]
            for result in results:
                result_id = result["id"]
                value = result["value"]
                value_start = value["start"]
                value_end = value["end"]
                value_labels = value["labels"]
                if len(value_labels) > 1:
                    print(f"There is a multilabel on upload id {id} at result {result_id}")
                    continue

                start_frame = int(value_start * video_fps)
                end_frame = int(value_end * video_fps)
                frame_label = value_labels[0]

                if frame_label == "BAD TRIAL":
                    good_trial = False
                    break

                per_frame_labels.append((start_frame, end_frame, frame_label))
            if not good_trial:
                break

        if not good_trial:
            continue

        per_video_per_frame_labels.append({
            "video_file_name": original_video_filename,
            "id": video_id,
            "labels": per_frame_labels,
        })

    # print(per_video_per_frame_labels)

    # Turn string labels into integers
    label_ids = set()
    for video_frame_labels in per_video_per_frame_labels:
        for _, _, frame_label in video_frame_labels["labels"]:
            label_ids.add(frame_label)

    label_to_id_dict = {v: i for i, v in enumerate(label_ids)}
    print(label_to_id_dict)

    # Construct per_video_label_indices
    video_id_to_labels = collections.defaultdict(list)
    for video_frame_labels in per_video_per_frame_labels:
        video_filename = video_frame_labels["video_file_name"]
        video_id = "-".join(video_filename.split("-")[0:4])
        video_id_to_labels[video_id].append(video_frame_labels)
        video_id_to_labels[video_id] = sorted(video_id_to_labels[video_id], key=lambda x: x["video_file_name"])

    video_extraction_tasks = collections.defaultdict(dict)
    for k, v in video_id_to_labels.items():
        temp_tasks = []
        all_labeled = True
        for gt_filename in video_id_to_video_filepaths[k]:
            video_labeled = False
            video_id = None
            for _v in v:
                if _v["video_file_name"] == gt_filename:
                    video_labeled = True
                    video_id = _v["id"]
                    video_labels = _v["labels"]
                    break

            if video_labeled:
                # print(f"\t[LABELED] {video_id} | {gt_filename}")
                temp_tasks.append((gt_filename, video_labels))
            else:
                all_labeled = False
                # print(f"\t[UNLABELED] {video_id} | {gt_filename}")
                # redownload labels json, this should not happen anymore
        if all_labeled:
            video_extraction_tasks[k] = temp_tasks

    corrected_extraction_tasks = []
    for video_id, label_tasks in video_extraction_tasks.items():
        all_frame_labels = []
        current_index = 0
        for gt_filename, video_labels in label_tasks:
            video_frame_index = 0
            for start_frame, end_frame, frame_label in video_labels:
                frame_label = label_to_id_dict[frame_label]
                if len(all_frame_labels) == 0:
                    for i in range(start_frame):
                        all_frame_labels.append((current_index, video_id, gt_filename, -1, video_frame_index))
                        current_index += 1
                        video_frame_index += 1
                if video_frame_index < start_frame:
                    for i in range(video_frame_index, start_frame + 1):
                        all_frame_labels.append((current_index, video_id, gt_filename, frame_label, video_frame_index))
                        current_index += 1
                        video_frame_index += 1
                for i in range(video_frame_index, end_frame):
                    all_frame_labels.append((current_index, video_id, gt_filename, frame_label, video_frame_index))
                    current_index += 1
                    video_frame_index += 1
        corrected_extraction_tasks.append(all_frame_labels)

    for corrected_extraction_task in tqdm.tqdm(corrected_extraction_tasks):
        current_gt_filename = ""
        current_video = None
        current_video_frame_index = 0
        for frame_index, video_id, gt_filename, frame_label, video_frame_index in corrected_extraction_task:
            if gt_filename != current_gt_filename:
                current_gt_filename = gt_filename
                # print(os.path.join(videos_directory, current_gt_filename), os.path.exists(os.path.join(videos_directory, current_gt_filename)))
                current_video = cv2.VideoCapture(os.path.join(videos_directory, current_gt_filename))
                current_video_frame_index = 0
            while current_video_frame_index < video_frame_index:
                success, image = current_video.read()
                current_video_frame_index += 1

            success, image = current_video.read()
            current_video_frame_index += 1
            if success:
                output_folder = os.path.join(labels_directory, video_id)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                image = resize_image(image, AGENT_RESOLUTION)
                cv2.imwrite(os.path.join(output_folder, f"{frame_index:05d}.png"), image)
                np.save(os.path.join(output_folder, f"{frame_index:05d}.npy"), to_categorical(frame_label, num_classes=len(label_ids)))


def main():
    process_labels("labels/cave.json", "MineRLBasaltFindCave-v0")
    process_labels("labels/waterfall.json", "MineRLBasaltMakeWaterfall-v0")
    process_labels("labels/animal.json", "MineRLBasaltCreateVillageAnimalPen-v0")
    process_labels("labels/house.json", "MineRLBasaltBuildVillageHouse-v0")


if __name__ == "__main__":
    main()

# Dictionary whose key is the name of the video
# The value is a list where the index into the list is the frame ID, and the value is the label at that frame.

# Write out frames and labels as files

# file structure? , individual frames like below?
# for gt_filename in house_video_id_to_video_filepaths[k]:
#     os.mkdir(gt_filename)
#     os.mkdir(gt_filename + '/Done')  # etc per subtask folders
#     for video_frame_labels in per_video_per_frame_labels:
#         for i in range(len(label[start:end])):
#           cv2.imwrite("frame%d.jpg" % i)