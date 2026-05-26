import cv2
import mediapipe as mp
import math
import time
import mido
impulse_count = 0
flash_frames = 0
# =========================
# MediaPipe setup
# =========================

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=2
)

mp_draw = mp.solutions.drawing_utils

# =========================
# Camera
# =========================

cap = cv2.VideoCapture(0)

# =========================
# MIDI
# =========================

midi_out = mido.open_output("GestureControl 1")

previous_midi_volume = -1

# =========================
# Volume smoothing
# =========================

smoothed_volume = 0
last_valid_volume = 0

# =========================
# Hand impulse detection
# =========================

prev_left_openness = None
prev_right_openness = None

impulse_threshold = 0.05
last_impulse_time = 0
impulse_cooldown = 0.25

# =========================
# Main loop
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # =========================
    # Pose processing
    # =========================

    pose_results = pose.process(rgb_frame)

    if pose_results.pose_landmarks:

        mp_draw.draw_landmarks(
            frame,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        landmarks = pose_results.pose_landmarks.landmark

        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Arm vectors
        left_dx = left_wrist.x - left_shoulder.x
        left_dy = left_wrist.y - left_shoulder.y

        right_dx = right_wrist.x - right_shoulder.x
        right_dy = right_wrist.y - right_shoulder.y

        # Angle from vertical
        left_angle = math.degrees(
            math.atan2(abs(left_dx), left_dy)
        )

        right_angle = math.degrees(
            math.atan2(abs(right_dx), right_dy)
        )

        # Clamp
        left_angle = min(left_angle, 90)
        right_angle = min(right_angle, 90)

        avg_angle = (left_angle + right_angle) / 2

        target_volume = int((avg_angle / 90) * 100)

        smoothed_volume = (
            smoothed_volume * 0.95
            + target_volume * 0.05
        )

        volume = int(smoothed_volume)

        last_valid_volume = volume
        # =========================
        # MIDI Volume
        # =========================

        midi_volume = int((volume / 100) * 127)

        if abs(midi_volume - previous_midi_volume) > 2:

            msg = mido.Message(
                "control_change",
                control=21,
                value=midi_volume
            )

            midi_out.send(msg)

            previous_midi_volume = midi_volume
        cv2.putText(
            frame,
            f"Volume: {volume}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    else:
        volume = last_valid_volume

    # =========================
    # Hand tracking
    # =========================

    hand_results = hands.process(rgb_frame)
    left_change = 0
    right_change = 0
    if hand_results.multi_hand_landmarks and pose_results.pose_landmarks:

        for i, hand_landmarks in enumerate(
            hand_results.multi_hand_landmarks
        ):

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Left / Right hand
            hand_label = (
                hand_results.multi_handedness[i]
                .classification[0]
                .label
            )

            # Wrist
            wrist = hand_landmarks.landmark[
                mp_hands.HandLandmark.WRIST
            ]

            # Middle fingertip
            middle_tip = hand_landmarks.landmark[
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP
            ]

            # Distance wrist -> fingertip
            finger_distance = math.sqrt(
                (middle_tip.x - wrist.x) ** 2
                + (middle_tip.y - wrist.y) ** 2
            )

            # =========================
            # Body-relative scaling
            # =========================

            left_elbow = landmarks[
                mp_pose.PoseLandmark.LEFT_ELBOW
            ]

            forearm_length = math.sqrt(
                (left_wrist.x - left_elbow.x) ** 2
                + (left_wrist.y - left_elbow.y) ** 2
            )

            # Normalized openness
            openness = finger_distance / forearm_length

            # =========================
            # Left hand
            # =========================

            if hand_label == "Left":

                if prev_left_openness is None:
                    prev_left_openness = openness

                left_change = (
                    prev_left_openness - openness
                )

                prev_left_openness = openness

            # =========================
            # Right hand
            # =========================

            elif hand_label == "Right":

                if prev_right_openness is None:
                    prev_right_openness = openness

                right_change = (
                    prev_right_openness - openness
                )

                prev_right_openness = openness

# =========================
# Synchronized impulse
# =========================

        current_time = time.time()

        if (
            left_change > impulse_threshold
            and right_change > impulse_threshold
            and current_time - last_impulse_time
            > impulse_cooldown
        ):

            impulse_count += 1

            flash_frames = 10

            print(f"IMPULSE {impulse_count}")
            note_on = mido.Message(
                "note_on",
                note=60,
                velocity=127
            )

            midi_out.send(note_on)
            last_impulse_time = current_time
    if flash_frames > 0:

        cv2.rectangle(
                frame,
                (0, 0),
                (frame.shape[1], frame.shape[0]),
                (255, 255, 255),
                20
            )

        flash_frames -= 1

        # Impulse counter
    cv2.putText(
            frame,
            f"Impulses: {impulse_count}",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
    cv2.imshow("Gesture Instrument", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()