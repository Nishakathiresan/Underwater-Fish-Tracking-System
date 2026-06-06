import cv2
import numpy as np
from ultralytics import YOLO
from norfair import Detection, Tracker
from collections import defaultdict, deque


MODEL_PATH = "best.pt"
VIDEO_PATH = "fish_video_20s.mp4"
OUTPUT_PATH = "Track+Count.mp4"


paused = False
selected_id = None
paths = defaultdict(lambda: deque(maxlen=200))


total_fish_ids = set()


def distance_function(detection, tracked_object):
    if tracked_object.last_detection is None:
        return 1e6
    return np.linalg.norm(
        detection.points - tracked_object.last_detection.points
    )


tracker = Tracker(
    distance_function=distance_function,
    distance_threshold=60,
    hit_counter_max=40,
    initialization_delay=3
)


model = YOLO(MODEL_PATH)


cap = cv2.VideoCapture(VIDEO_PATH)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)


def draw_grid(frame, step=80):
    for x in range(0, w, step):
        cv2.line(frame, (x, 0), (x, h), (60, 60, 60), 1)
    for y in range(0, h, step):
        cv2.line(frame, (0, y), (w, y), (60, 60, 60), 1)


def mouse_callback(event, x, y, flags, param):
    global selected_id
    if paused and event == cv2.EVENT_LBUTTONDOWN:
        for obj in tracker.tracked_objects:
            if obj.last_detection is None:
                continue
            cx, cy = obj.estimate[0].astype(int)
            if abs(cx - x) < 15 and abs(cy - y) < 15:
                selected_id = obj.id
                paths[selected_id].clear()
                print(f"🎯 Selected Fish ID: {selected_id}")
                break

cv2.namedWindow("Fish Tracking")
cv2.setMouseCallback("Fish Tracking", mouse_callback)


while cap.isOpened():

    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        detections = []

        
        results = model(frame, conf=0.3, verbose=False)[0]

        if results.boxes is not None:
            for box in results.boxes.xyxy:
                x1, y1, x2, y2 = box.cpu().numpy()
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                detections.append(
                    Detection(
                        points=np.array([[cx, cy]]),
                        scores=np.array([1.0])
                    )
                )

        
        tracked_objects = tracker.update(detections)

        
        for obj in tracked_objects:
            if obj.last_detection is None:
                continue

            cx, cy = obj.estimate[0].astype(int)
            paths[obj.id].append((cx, cy))

          
            total_fish_ids.add(obj.id)

    display = frame.copy()

   
    draw_grid(display)

   
    fish_count = len(total_fish_ids)

    cv2.putText(
        display,
        f"Total Fish Count: {fish_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    
    for obj in tracker.tracked_objects:
        if obj.last_detection is None:
            continue

        cx, cy = obj.estimate[0].astype(int)

        cv2.circle(display, (cx, cy), 3, (255, 0, 0), -1)
        cv2.putText(
            display,
            f"ID {obj.id}",
            (cx + 5, cy - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 0),
            1
        )

       
        if obj.id == selected_id:
            pts = paths[selected_id]
            for i in range(1, len(pts)):
                cv2.line(display, pts[i - 1], pts[i], (0, 0, 255), 2)

    
    if paused:
        cv2.putText(
            display,
            "PAUSED - Click fish to select",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    cv2.imshow("Fish Tracking", display)
    out.write(display)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        paused = not paused


cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Finished. Total Fish Count is stable.")
