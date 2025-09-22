import cv2
import os
import time
import psutil
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


def match_images(img1, img2, detector="ORB"):
    if detector == "ORB":
        feature = cv2.ORB_create()
        norm = cv2.NORM_HAMMING
    elif detector == "SIFT":
        feature = cv2.SIFT_create()
        norm = cv2.NORM_L2
    else:
        raise ValueError("Detector must be 'ORB' or 'SIFT'")

    # Detect and compute
    kp1, des1 = feature.detectAndCompute(img1, None)
    kp2, des2 = feature.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0  # no matches possible

    bf = cv2.BFMatcher(norm, crossCheck=True)
    matches = bf.match(des1, des2)

    # Similarity = fraction of matches that are "good"
    good = [m for m in matches if m.distance < 50]
    return len(good) / max(1, len(matches))  # normalized score


def evaluate_pipeline(dataset_dir, detector="ORB", threshold=0.3):
    y_true = []
    y_pred = []

    times = []
    memories = []

    process = psutil.Process(os.getpid())

    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        files = sorted(os.listdir(folder_path))
        if len(files) < 2:
            continue

        img1 = cv2.imread(os.path.join(folder_path, files[0]), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(os.path.join(folder_path, files[1]), cv2.IMREAD_GRAYSCALE)

        # Ground truth
        label = 1 if folder.startswith("same") else 0

        # Measure time & memory
        start_time = time.time()
        score = match_images(img1, img2, detector)
        end_time = time.time()

        mem = process.memory_info().rss / (1024 * 1024)  # in MB

        times.append(end_time - start_time)
        memories.append(mem)

        # Prediction
        pred = 1 if score >= threshold else 0

        y_true.append(label)
        y_pred.append(pred)


    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "avg_time": np.mean(times),
        "avg_memory": np.mean(memories),
    }
    return metrics


def show_matches(folder_path, detector="ORB", top_n=30):
    # Get the two image file names
    files = sorted(os.listdir(folder_path))
    if len(files) < 2:
        print("Folder does not contain 2 images.")
        return

    # Load both fingerprint images in grayscale
    img1 = cv2.imread(os.path.join(folder_path, files[0]), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(os.path.join(folder_path, files[1]), cv2.IMREAD_GRAYSCALE)

    # Choose detector
    if detector == "ORB":
        feature = cv2.ORB_create()
        norm = cv2.NORM_HAMMING
    else:
        feature = cv2.SIFT_create()
        norm = cv2.NORM_L2

    # Compute keypoints & descriptors
    kp1, des1 = feature.detectAndCompute(img1, None)
    kp2, des2 = feature.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        print("No descriptors found in one of the images.")
        return

    # Match with BFMatcher
    bf = cv2.BFMatcher(norm, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw matches
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:top_n], None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Show with matplotlib
    plt.figure(figsize=(12, 6))
    plt.imshow(match_img, cmap="gray")
    plt.title(f"{detector} Matches (Top {top_n}) - {os.path.basename(folder_path)}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    dataset_dir = "data_check"  # change this if your dataset folder has another name

    print("Evaluating ORB...")
    orb_metrics = evaluate_pipeline(dataset_dir, detector="ORB")

    print("Evaluating SIFT...")
    sift_metrics = evaluate_pipeline(dataset_dir, detector="SIFT")

    # Print results
    print("\nComparison:")
    print(f"{'Metric':<15} {'ORB':<15} {'SIFT':<15}")
    print("-" * 40)
    for key in ["accuracy", "precision", "recall", "f1", "avg_time", "avg_memory"]:
        print(f"{key:<12} {orb_metrics[key]:<15.4f} {sift_metrics[key]:<15.4f}")

    show_matches("data_check/same_1", detector="ORB", top_n=30)

    show_matches("data_check/different_1", detector="ORB", top_n=30)
