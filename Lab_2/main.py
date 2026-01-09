import cv2
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Object tracking using Shi-Tomasi and Lucas-Kanade")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument(
        "--mode",
        type=str,
        default="bbox",
        choices=["bbox", "homography"],
        help="Tracking visualization mode"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Инициализация видео
    cap = cv2.VideoCapture(args.video)
    ret, first_frame = cap.read()
    if not ret:
        print("Ошибка загрузки видео")
        return

    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY) # Конвертируем в градации серого

    # Выбор ROI - пользователь выбирает сам прямоугольник
    bbox = cv2.selectROI("Select object", first_frame, fromCenter=False) # выбор bounding box
    x, y, w, h = map(int, bbox)

    roi_gray = first_gray[y:y + h, x:x + w]

    # Параметры Shi-Tomasi
    feature_params = dict(
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7
    )

    p0 = cv2.goodFeaturesToTrack(roi_gray, mask=None, **feature_params)
    p0[:, 0, 0] += x
    p0[:, 0, 1] += y

    # Параметры Lucas-Kanade
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    prev_gray = first_gray.copy()
    prev_pts = p0

    bbox_pts = np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h]
    ], dtype=np.float32).reshape(-1, 1, 2)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Текущий кадр

        # Вычисляем оптический поток
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, prev_pts, None, **lk_params
        )

        good_old = prev_pts[status == 1]
        good_new = next_pts[status == 1]

        if len(good_new) < 4:
            print("Недостаточно точек для трекинга")
            break

        if args.mode == "bbox":
            x_min, y_min = np.min(good_new, axis=0)
            x_max, y_max = np.max(good_new, axis=0)
            cv2.rectangle(
                frame,
                (int(x_min), int(y_min)),
                (int(x_max), int(y_max)),
                (0, 255, 0),
                2
            )

        elif args.mode == "homography":
            # Матрица перспективных преобразований
            H, _ = cv2.findHomography(good_old, good_new, cv2.RANSAC, 5.0)
            if H is not None:
                new_bbox = cv2.perspectiveTransform(bbox_pts, H)
                cv2.polylines(
                    frame,
                    [np.int32(new_bbox)],
                    True,
                    (255, 0, 0),
                    2
                )
                bbox_pts[:] = new_bbox

        # Красные кружки на отслеженных точках
        for p in good_new:
            cv2.circle(frame, tuple(p.astype(int)), 3, (0, 0, 255), -1)

        cv2.imshow("Tracking", frame)

        prev_gray = gray.copy()
        prev_pts = good_new.reshape(-1, 1, 2)

        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()