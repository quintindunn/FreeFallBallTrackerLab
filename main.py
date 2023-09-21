import pathlib
import os

import cv2


# Credit: https://stackoverflow.com/a/58126805
def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dsize=dim, interpolation=inter)


def convert_video(path: pathlib.Path | str):
    """
    Converts a video from a supported ffmpeg format to a .mp4 file.
    ** IMPORTANT ** Don't let users pass whatever they please into path, it can easily be escaped for ACE.
    :param path: path to file to convert.
    :return: Path to the converted file.
    """
    path = pathlib.Path(path)
    output = '\\'.join(list(path.absolute().parts[:-1]) + path.parts[-1].split(".")[:-1]) + ".mp4"

    if not os.path.isfile(output):
        os.system(f"ffmpeg -i \"{path.absolute()}\" -q:v 0 {output}")

    return output


def process_video(path, csv_out="dt.csv", start_timestamp=0.0, ball_size=40, seed=457):
    """
    Processes the video (main function)
    :param path: path to file to process.
    :param csv_out: path for the csv output file.
    :param start_timestamp: timestamp in seconds of where to start the video.
    :param ball_size: Size of the ball in pixels for the tracker.
    :param seed: random seed used by opencv (mostly for tracker).
    :return: None
    """
    cap = cv2.VideoCapture(path)

    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    print("Frame rate:", frame_rate)

    def on_mouse(event, x, y, *_):
        nonlocal has_pos
        nonlocal s_x, s_y

        if event == cv2.EVENT_LBUTTONDOWN:
            has_pos = True
            s_x, s_y = x * 2, y * 2

    cv2.namedWindow("video")
    cv2.setMouseCallback("video", on_mouse)

    has_pos = False
    tracker = None
    s_x, s_y = 0, 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_timestamp * frame_rate))

    frame = None

    frame_count = 0

    timestamps = []

    while cap.isOpened():
        if frame is None or has_pos:
            # If the frame is None, or the init pos is picked
            # get the frame, this is used for the initial picking of position

            ret, frame = cap.read()
            frame_count += 1

            # If done reading video.
            if frame is None:
                break

            # Blur the frame for better results.
            blur = 2
            frame = cv2.blur(frame, (blur, blur))

        if has_pos and tracker is None:
            # If a position is picked but the tracker hasn't been instantiated.
            params = cv2.TrackerCSRT_Params()
            params.padding *= 2

            cv2.setRNGSeed(seed)
            tracker = cv2.TrackerCSRT.create(params)
            tracker.init(frame, (s_x - ball_size // 2, s_y - ball_size // 2, ball_size, ball_size))

        if tracker:
            # If the tracker is instantiated and running.
            ok, (x, y, w, h) = tracker.update(frame)

            if ok:
                # Found a new position.

                # Draw tracker area
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                # Draw avg location
                center_x, center_y = (x + w // 2), (y + h // 2)

                cv2.circle(frame, (center_x, center_y), 2, (0, 255, 0), -1)

                # Calculate the distance in pixels from the starting position of the ball that the user picked.
                distance_pixels = center_y - s_y

                timestamp = frame_count * 1 / frame_rate

                print(f"Distance (px): {distance_pixels} | Time: {timestamp:.2f} Frame #{frame_count}")

                timestamps.append({
                    "frame": frame_count,
                    "timestamp": timestamp,
                    "distance_px": distance_pixels,
                })

        # Resize the frame so it fits the screen correctly, this won't work for all monitor/resolutions but in my
        # case it did.

        height = float(cap.get(4))
        cv2.imshow("video", resize_with_aspect_ratio(frame, height=int(height / 2)))

        cv2.waitKey(1)

    cap.release()

    # Determine the furthest down pixel to calculate the size of 1m
    meter_conversion = sorted(timestamps, key=lambda x: x['distance_px'], reverse=True)[0]['distance_px']

    for timestamp in timestamps:
        # Add the calculated distances to the timestamp
        distance_m = timestamp['distance_px'] / meter_conversion
        timestamp['distance_m'] = distance_m
        timestamp['distance_cm'] = distance_m * 100
        timestamp['distance_mm'] = distance_m * 10

    # Log the timestamp
    for timestamp in timestamps:
        print(f"Time: {timestamp['timestamp']:.2f}, distance (m, cm, mm): {timestamp['distance_m']:.3f} "
              f"{timestamp['distance_cm']:.2f} {timestamp['distance_mm']:.2f}")

    # Setup CSV output for importing to excel/google sheets.
    csv_output = ["Time (s), Distance (m), Distance (cm), Distance (mm)"]
    for timestamp in timestamps:
        if timestamps[-1]['distance_m'] < timestamp['distance_m']:
            # If the tracker backtracked a bit, ignore this as it is physically impossible for that to happen
            continue

        # Add the data to the csv file.
        csv_output.append(f"{timestamp['timestamp']:.2f},{timestamp['distance_m']:.3f},{timestamp['distance_cm']:.2f},"
                          f"{timestamp['distance_mm']:.2f}")
        if timestamp['distance_m'] == 1:
            # If the ball has already gone as far down as possible it physically cannot backtrack so stop logging.
            break

    # Write the CSV data.
    with open(csv_out, 'w') as f:
        f.write('\n'.join(csv_output))


if __name__ == '__main__':
    file_path = convert_video(input("Input file: "))

    VIDEO_START = 0.5

    process_video(file_path, start_timestamp=VIDEO_START)
