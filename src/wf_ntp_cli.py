import os
import numpy as np
from WF_NTP_script import run_tracker  # from WF_NTP.WF_NTP_script
import json
import argparse
import pickle
import pandas as pd
import cv2

def process_dict(arg):
    if arg == '{}':
        return {}
    else:
        return dict(arg)

parser = argparse.ArgumentParser()
# Positional arguments
parser.add_argument('video_filename', type=str, help='Path to the video file')
parser.add_argument('save_as', type=str, help='Path to the folder where the results will be saved')

# General arguments
parser.add_argument('--downsampled_fps', type=int,
                    help='Reduce the fps of the video to speed up tracking')
parser.add_argument('--start_frame', type=int, default=0)
parser.add_argument('--limit_images_to', type=int, default=1800)
parser.add_argument('--fps', type=float, default=30.0)
parser.add_argument('--px_to_mm', type=float, default=0.04)
parser.add_argument('--darkfield', type=bool, default=False)

# Locating parameters
parser.add_argument('--keep_paralyzed_method', type=bool, default=False)
parser.add_argument('--std_px', type=int, default=64)
parser.add_argument('--threshold', type=int, default=9)
parser.add_argument('--opening', type=int, default=1)
parser.add_argument('--closing', type=int, default=3)
parser.add_argument('--prune_size', type=int, default=0)
parser.add_argument('--skeletonize', type=bool, default=False)
parser.add_argument('--do_full_prune', type=bool, default=False)

# Filtering parameters
parser.add_argument('--min_size', type=int, default=30)
parser.add_argument('--max_size', type=int, default=2300)
parser.add_argument('--minimum_ecc', type=float, default=0.93)

# Other parameters
parser.add_argument('--use_average', type=bool, default=True)
parser.add_argument('--lower', type=int, default=0)
parser.add_argument('--upper', type=int, default=100)
parser.add_argument('--Bends_max', type=float, default=20.0)
parser.add_argument('--Speed_max', type=float, default=0.035)
parser.add_argument('--extra_filter', type=bool, default=False)
parser.add_argument('--cutoff_filter', type=bool, default=False)
parser.add_argument('--max_dist_move', type=int, default=10)
parser.add_argument('--min_track_length', type=int, default=50)
parser.add_argument('--memory', type=int, default=5)
parser.add_argument('--bend_threshold', type=float, default=2.1)
parser.add_argument('--minimum_bends', type=float, default=0.0)
parser.add_argument('--frames_to_estimate_velocity', type=int, default=49)
parser.add_argument('--maximum_bpm', type=float, default=0.5)
parser.add_argument('--maximum_velocity', type=float, default=0.1)
parser.add_argument('--regions', type=process_dict, default='{}')
parser.add_argument('--output_overlayed_images', type=int, default=0)
parser.add_argument('--font_size', type=int, default=8)
parser.add_argument('--scale_bar_size', type=float, default=1.0)
parser.add_argument('--scale_bar_thickness', type=int, default=7)
parser.add_argument('--max_plot_pixels', type=int, default=2250000)
parser.add_argument('--Z_skip_images', type=int, default=1)
parser.add_argument('--use_images', type=int, default=100)
parser.add_argument('--use_around', type=int, default=4)
parser.add_argument('--stop_after_example_output', type=bool, default=False)
parser.add_argument('--stdout prefix', type=str, default='[6]')


# Stores all hyperparameters specified via command line in a json file
def store_hyperparameters(json_file_path, args):
    argparse_dict = vars(args)
    with open(json_file_path, 'w') as f:
        json.dump(argparse_dict, f)
    print('JSON File successfully saved!')


def process_video(json_file_path):
    # Load job as specified in the json file
    print('Loading job')
    try:
        with open(json_file_path) as f:
            job = json.load(f)
    except Exception:
        raise Exception('Not a valid settings.json file')
    print('Job loaded')

    run_tracker(job)


# Post-processes the spine_unfiltered file to match the track file. Specifically, for each row in the track file,
# find the row in the spine file that is closest to the x and y coordinates of the target row.
def post_process_spine_file(parent_dir, video_filename, fps):
    # Load track file, the original output of the WF-NTP tracker
    with open(parent_dir + '/track.p', 'br') as f:
        tracks = pickle.load(f, encoding='latin-1')
    target_df = pd.DataFrame(tracks)

    video_name = os.path.basename(video_filename).split('.')[0]

    # Rename track.p file so that it can be uniquely identified with the video
    os.rename(parent_dir + '/track.p', parent_dir + f'/{video_name}_track.p')

    # Load the spine_unfiltered file, that was written by the code added to the WF_NTP_script.py file
    spine_df = pd.read_csv(parent_dir + f'/{video_name}.spine_unfiltered', sep=' ', header=None)

    # Rename columns; first 3 columns are date_time, larva_id, time;
    # the rest are spine points, x and y coordinates alternating
    columns_points = []
    for i in range(1, 12):
        columns_points.extend([f'spinepoint{i}_x', f'spinepoint{i}_y'])
    spine_df.columns = ['date_time', 'larva_id', 'frame'] + columns_points + ['centroid_x', 'centroid_y']

    # Create an empty dataframe that will contain the filtered spine data
    filtered_spine_df = pd.DataFrame(columns=['date_time', 'larva_id', 'frame'] + columns_points)

    # For each row in target_df, find the corresponding row in spine_df which is closest to the x and y coordinates
    for index, row in target_df.iterrows():
        frame_spine_df = spine_df[spine_df['frame'] == row['frame']]

        # Get x and y coordinates from target_df and find the entry in spine_df that is closest to these coordinates
        target_centroid_x, target_centroid_y = row['x'], row['y']
        min_dist = np.Inf
        min_index = None
        for spine_index, spine_row in frame_spine_df.iterrows():
            spine_centroid_x, spine_centroid_y = spine_row['centroid_x'], spine_row['centroid_y']
            dist = np.sqrt((target_centroid_x - spine_centroid_x) ** 2 + (target_centroid_y - spine_centroid_y) ** 2)
            if dist < min_dist:
                min_dist = dist
                min_index = spine_index
        # Add the row in spine_df that is the closest to the centroid of the target entry
        filtered_spine_df = filtered_spine_df.append(frame_spine_df.loc[min_index])

    # Add a time column to both dataframes
    for temp_df in [target_df, filtered_spine_df]:
        time = []
        for frame_nr in temp_df['frame']:
            time.append('{0:.3f}'.format(frame_nr / fps))
        temp_df.insert(loc=2, column='time_in_sec', value=time)

    # Reformat larva_id to have 5 digits
    filtered_spine_df['larva_id'] = filtered_spine_df['larva_id'].apply(lambda x: '{0:0>5}'.format(x))

    filtered_spine_df.drop(columns=['frame', 'centroid_x', 'centroid_y'], inplace=True)

    # Save filtered spine file and .spine file
    filtered_spine_df.to_csv(parent_dir + f'/{video_name}.spine', sep=' ', header=False, index=False)

    # Check if the spine file and the track file match
    if not check_spine_file(target_df, filtered_spine_df):
        raise Warning('Spines and tracks do not match!')


# Check if for each time point, the number of rows in the spine file is equal to the number of rows in the track file
def check_spine_file(target_df, spine_df):
    for time_stamp in target_df['time_in_sec'].unique():
        frame_target_df = target_df[target_df['time_in_sec'] == time_stamp]
        frame_spine_df = spine_df[spine_df['time_in_sec'] == time_stamp]
        if len(frame_target_df) != len(frame_spine_df):
            print(f'Number of rows in spine file at time {time_stamp} sec does not match number of rows in track file')
            return False
    return True


# Due to the long processing time of WF-NTP, there is an option to downsample the video before processing it,
# meaning reducing the number of frames that are processed for each video.
def downsample_video(video_path, old_fps, new_fps):
    print('Downsampling video...')
    # Extract all frames from original video
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    print(f'Number of frames in original video: {len(frames)}')

    # Downsample frames
    frames = frames[::int(old_fps / new_fps)]
    print(f'Number of frames in downsampled video: {len(frames)}')

    # Save frames as video
    size = (frames[0].shape[1], frames[0].shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    out = cv2.VideoWriter(f'{video_path.replace(".avi", "_downsampled.avi")}', fourcc, new_fps, size)
    for i in range(len(frames)):
        out.write(frames[i])
    out.release()


def main():
    args = parser.parse_args()

    # If downsample option is specified, downsample video
    if args.downsampled_fps is not None:
        # Check if downsampled video already exists, if not, downsample video
        if not os.path.exists(args.video_filename.replace('.avi', '_downsampled.avi')):
            downsample_video(args.video_filename, args.fps, args.downsampled_fps)
        args.video_filename = args.video_filename.replace('.avi', '_downsampled.avi')
        args.fps = args.downsampled_fps

    store_hyperparameters(os.path.join(args.save_as, 'settings.json'), args)

    # Process video with WF-NTP and post-process spine file
    # so that we obtain the .spine file used for downstream anylysis
    process_video(os.path.join(args.save_as, 'settings.json'))
    post_process_spine_file(args.save_as, args.video_filename, args.fps)


if __name__ == "__main__":
    main()
