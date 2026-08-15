from collections import deque, defaultdict

from plate_recognition import is_valid_plate


plate_history = defaultdict(lambda: deque(maxlen=15))
plate_final = {}


def reset_history():
    """Clear all tracked plate state. Must be called before processing a
    new video (e.g. a new API request) since YOLO track IDs restart from
    scratch each run and would otherwise collide with leftover state from
    a previous video."""
    plate_history.clear()
    plate_final.clear()


def get_plate_history(track_id, new_text):

    if new_text:
        plate_history[track_id].append(new_text)

    history = plate_history[track_id]

    if not history:
        return plate_final.get(track_id, "")

    length_counts = defaultdict(int)
    for plate in history:
        length_counts[len(plate)] += 1

    target_length = max(length_counts, key=length_counts.get)

    valid_history = [p for p in history if len(p) == target_length]

    if not valid_history:
        return plate_final.get(track_id, "")

    character_result = []
    for position in range(target_length):
        counts = defaultdict(int)
        for plate in valid_history:
            counts[plate[position]] += 1
        best_char = max(counts, key=counts.get)
        character_result.append(best_char)

    final_plate = "".join(character_result)

    if is_valid_plate(final_plate) and len(valid_history) >= 3:
        plate_final[track_id] = final_plate

    return plate_final.get(track_id, "")