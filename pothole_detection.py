import cv2

import math

import time

import numpy as np

from collections import deque, Counter

from ultralytics import YOLO



# ============== PROJECT SETTINGS ==============

MODEL_PATH = "best.pt"

VIDEO_SOURCE = "video_11.mp4"  # Change to your actual video filename

OUTPUT_VIDEO = "pothole_hud_output.mp4"



CONFIDENCE = 0.25

IMAGE_SIZE = 640



# Approximate camera geometry: calibrate for meaningful distances.

CAMERA_HEIGHT_M = 1.2

VERTICAL_FOV_DEG = 45.0

HORIZON_RATIO = 0.40



# ---- NEW: lane / direction settings ----

# Frame is split into 3 vertical zones: LEFT | CENTER (our vehicle) | RIGHT

LANE_SPLITS = (1 / 3, 2 / 3)

# A side lane must be at least this much safer than the center to suggest a move

# (0.6 = side danger must be below 60% of center danger)

SAFER_MARGIN = 0.6

# Number of recent frames used to vote on the direction (stops arrow flicker)

SMOOTH_FRAMES = 8



CYAN = (255, 220, 30)

BLUE = (255, 125, 20)

WHITE = (240, 250, 255)

GREEN = (70, 255, 170)

AMBER = (0, 175, 255)

RED = (55, 65, 255)

DARK = (15, 24, 30)





def text(img, value, pos, scale=0.55, color=WHITE, thick=1):

    cv2.putText(img, str(value), pos, cv2.FONT_HERSHEY_SIMPLEX,

                scale, color, thick, cv2.LINE_AA)





def panel(img, x1, y1, x2, y2, color=CYAN, alpha=0.76):

    """Dark translucent panel with a bright technical outline."""

    overlay = img.copy()

    cv2.rectangle(overlay, (x1, y1), (x2, y2), DARK, -1)

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    cv2.line(img, (x1, y1), (x1 + 28, y1), color, 3)

    cv2.line(img, (x1, y1), (x1, y1 + 20), color, 3)





def corner_box(img, x1, y1, x2, y2, color, phase):

    """Glowing-style targeting brackets, reticle and scan detail."""

    w = max(14, int((x2 - x1) * 0.24))

    h = max(14, int((y2 - y1) * 0.24))

    t = 3



    # Soft outer glow

    glow = img.copy()

    corners = [

        ((x1, y1), (x1 + w, y1)), ((x1, y1), (x1, y1 + h)),

        ((x2, y1), (x2 - w, y1)), ((x2, y1), (x2, y1 + h)),

        ((x1, y2), (x1 + w, y2)), ((x1, y2), (x1, y2 - h)),

        ((x2, y2), (x2 - w, y2)), ((x2, y2), (x2, y2 - h))

    ]



    for a, b in corners:

        cv2.line(glow, a, b, color, 8, cv2.LINE_AA)

    cv2.addWeighted(glow, 0.22, img, 0.78, 0, img)



    for a, b in corners:

        cv2.line(img, a, b, color, t, cv2.LINE_AA)



    # Targeting reticle

    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    radius = max(8, min(20, (x2 - x1) // 5))

    cv2.circle(img, (cx, cy), radius, CYAN, 1, cv2.LINE_AA)

    cv2.line(img, (cx - radius - 5, cy), (cx - radius + 3, cy),

            WHITE, 1, cv2.LINE_AA)

    cv2.line(img, (cx + radius - 3, cy), (cx + radius + 5, cy),

            WHITE, 1, cv2.LINE_AA)

    cv2.line(img, (cx, cy - radius - 5), (cx, cy - radius + 3),

            WHITE, 1, cv2.LINE_AA)

    cv2.line(img, (cx, cy + radius - 3), (cx, cy + radius + 5),

            WHITE, 1, cv2.LINE_AA)



    # Moving scan highlight

    scan_x = x1 + int((x2 - x1) * ((phase % 60) / 60))

    if x1 < scan_x < x2:

        cv2.line(img, (scan_x, y1), (scan_x, y2), CYAN, 1,

                cv2.LINE_AA)





def estimate_distance(y_bottom, frame_height):

    """

    Rough monocular geometry estimate.

    Requires a suitable ground-contact point and calibrated camera.

    Not physical pothole depth; inaccurate on slopes or changing pitch.

    """

    horizon_y = frame_height * HORIZON_RATIO

    gap = y_bottom - horizon_y



    if gap <= 5:

        return None



    focal_y = (frame_height / 2) / math.tan(

        math.radians(VERTICAL_FOV_DEG / 2)

    )

    return CAMERA_HEIGHT_M * focal_y / gap





# =====================================================================

# Keep every processed video at a consistent 16:9 canvas. The source is
# resized proportionally and letterboxed (never stretched or cropped), so the
# HUD stays aligned and all four corners remain inside the output frame.
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720

def fit_frame(frame, target_w=OUTPUT_WIDTH, target_h=OUTPUT_HEIGHT):
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


# NEW: LANE ANALYSIS + DIRECTION GUIDANCE

# =====================================================================

def get_lane(cx, w):

    """0 = LEFT, 1 = CENTER (our vehicle), 2 = RIGHT, based on box center x."""

    if cx < w * LANE_SPLITS[0]:

        return 0

    if cx < w * LANE_SPLITS[1]:

        return 1

    return 2





def compute_lane_stats(dets, w, h):

    """

    dets: list of (x1, y1, x2, y2, conf).

    Returns (counts, scores), each a list [left, center, right].



    counts = how many potholes in each lane.

    scores = danger per lane. Each pothole adds:

        conf * (0.3 + 0.7 * closeness)

    where closeness = y2 / h (0 = far at horizon, 1 = right at the bumper).

    So a near, confident pothole hurts more than a far, doubtful one.

    """

    counts = [0, 0, 0]

    scores = [0.0, 0.0, 0.0]

    horizon_y = h * HORIZON_RATIO



    for x1, y1, x2, y2, conf in dets:

        if y2 < horizon_y:        # above the horizon -> not on the road, ignore

            continue

        lane = get_lane((x1 + x2) / 2, w)

        closeness = min(1.0, y2 / h)

        counts[lane] += 1

        scores[lane] += conf * (0.3 + 0.7 * closeness)



    return counts, scores





def decide_direction(scores):
    """Choose the safest of left, center (current vehicle path), and right.

    scores = [left_danger, center_danger, right_danger]. Lower is safer.
    Ties favor CENTER so the system does not suggest an unnecessary lane change.
    """
    left, center, right = scores

    # Center is the best or tied-best path: continue straight.
    if center <= left and center <= right:
        return "CLEAR"

    # If a side is the uniquely safest path, show that direction.
    if left < center and left < right:
        return "LEFT"
    if right < center and right < left:
        return "RIGHT"

    # Left and right tie while both are safer than center: don't guess.
    return "BLOCKED"


def dashed_vline(img, x, y1, y2, color, dash=14, gap=10):

    for y in range(y1, y2, dash + gap):

        cv2.line(img, (x, y), (x, min(y + dash, y2)), color, 1,

                cv2.LINE_AA)





def draw_lane_overlay(frame, counts, decision):

    """Dashed lane dividers, lane labels, and a green tint on the suggested lane."""

    h, w = frame.shape[:2]

    top, bottom = 170, h - 46

    x1 = int(w * LANE_SPLITS[0])

    x2 = int(w * LANE_SPLITS[1])

    bounds = [(0, x1), (x1, x2), (x2, w)]



    # Tint the lane we are being told to move into

    if decision in ("LEFT", "RIGHT"):

        a, b = bounds[0] if decision == "LEFT" else bounds[2]

        overlay = frame.copy()

        cv2.rectangle(overlay, (a, top), (b, bottom), GREEN, -1)

        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)



    dashed_vline(frame, x1, top, bottom, CYAN)

    dashed_vline(frame, x2, top, bottom, CYAN)



    names = ["LEFT", "CENTER (YOU)", "RIGHT"]

    for i, (a, b) in enumerate(bounds):

        label = f"{names[i]}: {counts[i]}"

        color = GREEN if counts[i] == 0 else (AMBER if counts[i] < 3 else RED)

        label_x = min(max(a + 12, 8), max(8, b - 150))
        text(frame, label, (label_x, top + 24), 0.48, color, 1)





def draw_direction_arrow(frame, decision):
    """Responsive full-width direction banner with a compact road guide."""
    h, w = frame.shape[:2]
    cx = w // 2

    if decision == "LEFT":
        color, msg, arrow_dir = GREEN, "MOVE LEFT", "LEFT"
    elif decision == "RIGHT":
        color, msg, arrow_dir = GREEN, "MOVE RIGHT", "RIGHT"
    elif decision == "CLEAR":
        color, msg, arrow_dir = GREEN, "GO STRAIGHT", "STRAIGHT"
    else:
        color, msg, arrow_dir = RED, "SLOW DOWN - PATH UNCLEAR", "STOP"

    # A full-width banner avoids clipping long warnings on different resolutions.
    margin = max(12, int(w * 0.025))
    x1, x2 = margin, w - margin
    y1, y2 = 86, 153
    cv2.rectangle(frame, (x1, y1), (x2, y2), (10, 15, 22), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)

    acx, acy = x1 + 42, (y1 + y2) // 2
    if arrow_dir == "LEFT":
        cv2.arrowedLine(frame, (acx + 23, acy), (acx - 23, acy), color, 6, cv2.LINE_AA, tipLength=0.48)
    elif arrow_dir == "RIGHT":
        cv2.arrowedLine(frame, (acx - 23, acy), (acx + 23, acy), color, 6, cv2.LINE_AA, tipLength=0.48)
    elif arrow_dir == "STRAIGHT":
        cv2.arrowedLine(frame, (acx, acy + 22), (acx, acy - 22), color, 6, cv2.LINE_AA, tipLength=0.48)
    else:
        pts = np.array([[acx, acy - 22], [acx - 20, acy + 18], [acx + 20, acy + 18]], dtype=np.int32)
        cv2.polylines(frame, [pts], True, color, 4, cv2.LINE_AA)
        text(frame, "!", (acx - 5, acy + 12), 0.75, color, 2)

    # Center the message in the remaining banner space, keeping it within the frame.
    msg_scale = 0.72 if len(msg) <= 12 else 0.58
    text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, msg_scale, 2)[0]
    text_x = min(max(x1 + 86, (w - text_size[0]) // 2), x2 - text_size[0] - 12)
    text_y = acy + text_size[1] // 2
    text(frame, msg, (text_x, text_y), msg_scale, color, 2)

    # Compact road guide stays in the lower centre and is clamped to frame bounds.
    guide_top = max(390, h - 220)
    guide_bottom = min(h - 50, max(guide_top + 30, h - 55))
    if guide_bottom > guide_top + 20:
        road_half = max(35, min(75, w // 12))
        lx, rx = max(4, cx - road_half), min(w - 4, cx + road_half)
        cv2.line(frame, (min(w - 2, lx + 18), guide_top), (lx, guide_bottom), CYAN, 2, cv2.LINE_AA)
        cv2.line(frame, (max(2, rx - 18), guide_top), (rx, guide_bottom), CYAN, 2, cv2.LINE_AA)
        car_y = guide_bottom - 3
        cv2.rectangle(frame, (cx - 13, car_y - 22), (cx + 13, car_y), WHITE, 2, cv2.LINE_AA)
        cv2.rectangle(frame, (cx - 7, car_y - 17), (cx + 7, car_y - 5), CYAN, -1)
        start = (cx, car_y - 25)
        if arrow_dir == "LEFT":
            target = (max(5, cx - road_half), guide_top + 18)
            cv2.arrowedLine(frame, start, target, color, 4, cv2.LINE_AA, tipLength=0.4)
        elif arrow_dir == "RIGHT":
            target = (min(w - 5, cx + road_half), guide_top + 18)
            cv2.arrowedLine(frame, start, target, color, 4, cv2.LINE_AA, tipLength=0.4)
        elif arrow_dir == "STRAIGHT":
            cv2.arrowedLine(frame, start, (cx, guide_top + 5), color, 4, cv2.LINE_AA, tipLength=0.4)
        else:
            cv2.circle(frame, (cx, guide_top + 18), 15, RED, 3, cv2.LINE_AA)
            text(frame, "!", (cx - 5, guide_top + 24), 0.75, RED, 2)


def draw_hud(frame, detections_count, high_count, fps, infer_ms,

            frame_no, total_frames):

    h, w = frame.shape[:2]



    # Subtle dark top and bottom bands

    overlay = frame.copy()

    cv2.rectangle(overlay, (0, 0), (w, 76), (9, 18, 26), -1)

    cv2.rectangle(overlay, (0, h - 43), (w, h), (9, 18, 26), -1)

    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)



    cv2.line(frame, (0, 76), (w, 76), CYAN, 1)

    cv2.line(frame, (0, h - 43), (w, h - 43), CYAN, 1)



    # System title

    cv2.circle(frame, (32, 34), 19, CYAN, 1, cv2.LINE_AA)

    cv2.circle(frame, (32, 34), 11, CYAN, 1, cv2.LINE_AA)

    cv2.line(frame, (12, 34), (52, 34), CYAN, 1)

    cv2.line(frame, (32, 14), (32, 54), CYAN, 1)

    text(frame, "POTHOLE INTELLIGENCE SYSTEM", (62, 30),

        0.62, WHITE, 2)

    text(frame, "VISION  /  ROAD HAZARD ANALYSIS", (63, 53),

        0.38, CYAN, 1)



    # Metrics across the top

    start_x = max(360, w - 430)

    text(frame, f"DETECTIONS  {detections_count}",

        (start_x, 27), 0.48, CYAN, 1)

    text(frame, f"INFER  {infer_ms:.0f} ms",

        (start_x, 53), 0.43, WHITE, 1)

    text(frame, f"FPS {fps:.1f}", (w - 145, 27),

        0.48, GREEN, 1)

    text(frame, "TRACKING ON", (w - 145, 53),

        0.40, GREEN, 1)



    # Bottom alert and progress indicator

    if detections_count:

        alert = f"!! ROAD HAZARD DETECTED  /  {detections_count} TARGET(S)"

        alert_color = RED if high_count else AMBER

    else:

        alert = "SCANNING ROAD  /  NO TARGETS IN CURRENT FRAME"

        alert_color = GREEN



    text(frame, alert, (18, h - 16), 0.48, alert_color, 2)



    if total_frames > 0:

        progress = min(1.0, frame_no / total_frames)

        bar_y = h - 3

        cv2.line(frame, (0, bar_y), (w, bar_y), (50, 60, 65), 2)

        cv2.line(frame, (0, bar_y), (int(w * progress), bar_y),

                CYAN, 2)





def main():

    print("Loading trained YOLO model...")

    model = YOLO(MODEL_PATH)



    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():

        raise RuntimeError(

            f"Cannot open {VIDEO_SOURCE}. Check its filename and location."

        )



    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    source_fps = cap.get(cv2.CAP_PROP_FPS)

    if not source_fps or source_fps < 1:

        source_fps = 25.0



    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(

        OUTPUT_VIDEO, fourcc, source_fps, (OUTPUT_WIDTH, OUTPUT_HEIGHT)

    )



    if not writer.isOpened():

        cap.release()

        raise RuntimeError("Could not create output video file.")



    frame_no = 0

    processing_start = time.time()

    fps_smooth = 0.0

    direction_history = deque(maxlen=SMOOTH_FRAMES)   # NEW



    print(f"Processing video ({source_width}x{source_height} source -> {OUTPUT_WIDTH}x{OUTPUT_HEIGHT} output).")
    print("Output will be saved to:", OUTPUT_VIDEO)



    try:

        while True:

            ok, frame = cap.read()

            if not ok:

                break



            frame_no += 1

            # Preserve source aspect ratio and fit it inside the fixed output canvas.
            frame = fit_frame(frame)
            tick = time.time()

            h, w = frame.shape[:2]



            results = model.track(

                frame,

                persist=True,

                tracker="bytetrack.yaml",

                conf=CONFIDENCE,

                imgsz=IMAGE_SIZE,

                verbose=False

            )



            infer_ms = (time.time() - tick) * 1000

            fps_now = 1.0 / max(time.time() - tick, 0.001)

            fps_smooth = (0.85 * fps_smooth + 0.15 * fps_now

                        if fps_smooth else fps_now)



            boxes = results[0].boxes

            count = len(boxes) if boxes is not None else 0

            high_count = 0



            # ---------- PASS 1: collect detections ----------

            dets = []       # (x1, y1, x2, y2, conf)

            extras = []     # (track_id) kept in the same order as dets

            if boxes is not None:

                for det in boxes:

                    x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())

                    x1, x2 = max(0, x1), min(w - 1, x2)

                    y1, y2 = max(0, y1), min(h - 1, y2)

                    conf = float(det.conf[0])

                    track_id = int(det.id[0]) if det.id is not None else None

                    dets.append((x1, y1, x2, y2, conf))

                    extras.append(track_id)

                    if conf >= 0.60:

                        high_count += 1



            # ---------- NEW: lane stats + direction decision ----------

            counts, scores = compute_lane_stats(dets, w, h)

            raw_decision = decide_direction(scores)

            direction_history.append(raw_decision)

            decision = Counter(direction_history).most_common(1)[0][0]
            if frame_no % 30 == 0:
                print(f"Frame {frame_no}: potholes L/C/R={counts}, danger={ [round(v, 2) for v in scores] }, direction={decision}")



            # Lanes drawn first so boxes sit on top of them

            draw_lane_overlay(frame, counts, decision)



            # ---------- PASS 2: draw each detection ----------

            for (x1, y1, x2, y2, conf), track_id in zip(dets, extras):

                color = AMBER if conf >= 0.60 else CYAN

                ident = f"#{track_id:02d}" if track_id is not None else "#--"



                dist = estimate_distance(y2, h)

                dist_text = f"~{dist:.1f}m*" if dist else "N/A"



                corner_box(frame, x1, y1, x2, y2, color, frame_no)



                # Label panel anchored above the target where possible

                label_w = min(215, max(165, x2 - x1 + 40))

                label_h = 48

                lx = max(4, min(w - label_w - 4, x1))

                ly = y1 - label_h - 5

                if ly < 80:

                    ly = min(h - 50, y2 + 5)



                panel(frame, lx, ly, lx + label_w, ly + label_h,

                    color, 0.84)

                text(frame, f"TARGET {ident}  /  POTHOLE",

                    (lx + 8, ly + 18), 0.39, WHITE, 1)

                text(frame, f"CONF {conf:.0%}   DIST {dist_text}",

                    (lx + 8, ly + 38), 0.40, color, 1)



                # Connector from label to detection

                anchor_x = max(lx, min(lx + label_w, (x1 + x2) // 2))

                anchor_y = ly + label_h if ly < y1 else ly

                cv2.line(frame, (anchor_x, anchor_y),

                        ((x1 + x2) // 2, (y1 + y2) // 2),

                        color, 1, cv2.LINE_AA)



            draw_hud(

                frame, count, high_count, fps_smooth,

                infer_ms, frame_no, total_frames

            )



            # NEW: direction arrow drawn last so nothing covers it

            draw_direction_arrow(frame, decision)



            # Small tracking scan marker in top-right corner

            cv2.circle(frame, (w - 28, 93), 7, CYAN, 1)

            cv2.circle(frame, (w - 28, 93), 2, GREEN, -1)



            writer.write(frame)

            cv2.imshow("POTHOLE INTELLIGENCE | VIDEO ANALYSIS", frame)



            # Q/Esc exits early; Space pauses playback.

            key = cv2.waitKey(max(1, int(1000 / source_fps))) & 0xFF

            if key in (ord("q"), 27):

                print("Stopped early; output contains processed frames so far.")

                break

            if key == ord(" "):

                while True:

                    k = cv2.waitKey(30) & 0xFF

                    if k in (ord(" "), ord("q"), 27):

                        break

                if k in (ord("q"), 27):

                    break



    finally:

        cap.release()

        writer.release()

        cv2.destroyAllWindows()



    elapsed = time.time() - processing_start

    print(f"Processed {frame_no} frames in {elapsed:.1f} seconds.")

    print(f"Saved annotated video: {OUTPUT_VIDEO}")

    print("Note: *Distance is a rough, uncalibrated estimate, not pothole depth.")





if __name__ == "__main__":

    main()