def count_fingers(landmarks):
    if not landmarks:
        return 0

    fingers = []

    # Thumb
    if landmarks[4][0] > landmarks[3][0]:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other 4 fingers
    tip_ids = [8, 12, 16, 20]
    for tip in tip_ids:
        if landmarks[tip][1] < landmarks[tip - 2][1]:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)