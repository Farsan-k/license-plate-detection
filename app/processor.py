import cv2
from ultralytics import YOLO

from plate_recognition import recognize_plate
from plate_tracker import get_plate_history, plate_history, reset_history


CONF_THRESH = 0.30
OCR_INTERVAL = 5


def draw_box_and_id(frame, x1, y1, x2, y2, track_id):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(
        frame, f"ID: {track_id}",
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )


def draw_plate_text(frame, stable_text, x1, y1, x2, y2, width, height):

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.3
    thickness = 3

    (text_w, text_h), baseline = cv2.getTextSize(
        stable_text, font, font_scale, thickness
    )

    text_x = x1
    text_y = y1 - 15

    if text_y - text_h < 0:
        text_y = y2 + text_h + 15

    if text_x + text_w > width:
        text_x = max(0, width - text_w - 5)

    text_y = max(text_h + 5, min(text_y, height - 5))

    cv2.putText(
        frame, stable_text, (text_x, text_y),
        font, font_scale, (0, 0, 0), thickness + 3
    )
    cv2.putText(
        frame, stable_text, (text_x, text_y),
        font, font_scale, (255, 255, 255), thickness
    )


def process_frame(model, frame, frame_count, width, height):

    results = model.track(
        frame,
        persist=True,
        verbose=False,
        conf=CONF_THRESH,
        tracker="botsort.yaml"
    )

    for result in results:

        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:

            conf = float(box.conf.item())
            if conf < CONF_THRESH:
                continue

            if box.id is None:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            track_id = int(box.id.item())
            plate_crop = frame[y1:y2, x1:x2]

            if plate_crop.size == 0:
                continue

            text = ""

            if frame_count % OCR_INTERVAL == 0:
                text, _ = recognize_plate(plate_crop)

            stable_text = get_plate_history(track_id, text)

            if not stable_text and plate_history[track_id]:
                stable_text = plate_history[track_id][-1]

            draw_box_and_id(frame, x1, y1, x2, y2, track_id)

            if stable_text:
                draw_plate_text(frame, stable_text, x1, y1, x2, y2, width, height)

    return frame


def run(input_video, out_video, weights_path, show_preview=True):
    """
    show_preview=False must be used when calling this from a server
    (e.g. the FastAPI app) since there is no display available there
    and cv2.imshow/waitKey would error out or hang.
    """

    # Track IDs are only unique within a single video, so state from a
    # previous run (e.g. a previous API request) must not leak into this one.
    reset_history()

    model = YOLO(weights_path)

    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        print("ERROR: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Video FPS:", fps)
    print("Video size:", width, "x", height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_video, fourcc, fps, (width, height))

    frame_count = 0

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        frame = process_frame(model, frame, frame_count, width, height)

        out.write(frame)

        if show_preview:
            cv2.imshow("License Plate Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()
    if show_preview:
        cv2.destroyAllWindows()

    print("Processing completed!")
    print("Output:", out_video)