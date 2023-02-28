import os
from collections import defaultdict

people = [
    "David",
    "V",
    "Nick",
    "Ellen",
    "Josh"
]

data_directory = "/media/david/research_data/basalt2022"
task_names = os.listdir(data_directory)
assigned_tasks = defaultdict(list)


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


for task_name in task_names:
    task_path = os.path.join(data_directory, task_name)
    all_videos_paths = [os.path.join(task_name, f) for f in os.listdir(task_path) if f.endswith(".mp4")]
    split_video_paths = split(all_videos_paths, len(people))
    for person, video_paths in zip(people, split_video_paths):
        assigned_tasks[person].extend(video_paths)


if not os.path.isdir("split_tasks"):
    os.makedirs("split_tasks")

for person in people:
    with open(os.path.join("split_tasks", f"{person}.txt"), 'w') as outfile:
        for path in assigned_tasks[person]:
            outfile.write(f"{path}\n")

