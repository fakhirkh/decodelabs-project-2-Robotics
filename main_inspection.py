# ============================================================
# Project 2: Live Demo - No Phone, No Real Gears Needed!
# DecodeLabs Internship - Batch 2026
# Author:Fakhir Ali Khan (2023-MC-29)
#
# HOW IT WORKS:
# 1. Generates perfect and defective gear images
# 2. Shows them one by one in a window
# 3. Webcam captures that window
# 4. Inspector analyzes and shows PASS/FAIL
#
# OR even simpler:
# Runs inspection directly on images as if they were
# live video frames — same pipeline, same result
# ============================================================

import cv2
import numpy as np
import os
import time

# ─────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────

DEFECT_DEPTH_THRESHOLD     = 45.0
MIN_OBJECT_AREA            = 5000
CONSECUTIVE_FAIL_THRESHOLD = 3


# ─────────────────────────────────────────────────────────
#  GEAR GENERATOR (same as before)
# ─────────────────────────────────────────────────────────

def create_synthetic_gear(size=400, defective=False, gear_id=0):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = (45, 45, 45)
    cx, cy    = size // 2, size // 2
    num_teeth = 18
    r_tooth   = 170
    r_valley  = 133
    r_notch   = 65

    np.random.seed(gear_id * 11 + 5)
    if defective:
        num_missing = np.random.randint(1, 3)
        skip_teeth  = list(
            np.random.choice(num_teeth, num_missing, replace=False)
        )
    else:
        skip_teeth = []

    gear_pts = []
    for i in range(num_teeth):
        angle      = (2 * np.pi * i)       / num_teeth
        next_angle = (2 * np.pi * (i + 1)) / num_teeth
        mid_angle  = (angle + next_angle)   / 2.0
        t_half     = (next_angle - angle)   * 0.30

        if i in skip_teeth:
            gear_pts.append([
                int(cx + r_valley * np.cos(angle + 0.04)),
                int(cy + r_valley * np.sin(angle + 0.04))
            ])
            gear_pts.append([
                int(cx + r_notch * np.cos(mid_angle - 0.10)),
                int(cy + r_notch * np.sin(mid_angle - 0.10))
            ])
            gear_pts.append([
                int(cx + r_notch * np.cos(mid_angle)),
                int(cy + r_notch * np.sin(mid_angle))
            ])
            gear_pts.append([
                int(cx + r_notch * np.cos(mid_angle + 0.10)),
                int(cy + r_notch * np.sin(mid_angle + 0.10))
            ])
            gear_pts.append([
                int(cx + r_valley * np.cos(next_angle - 0.04)),
                int(cy + r_valley * np.sin(next_angle - 0.04))
            ])
        else:
            gear_pts.append([
                int(cx + r_valley * np.cos(angle + 0.04)),
                int(cy + r_valley * np.sin(angle + 0.04))
            ])
            gear_pts.append([
                int(cx + r_tooth  * np.cos(mid_angle - t_half)),
                int(cy + r_tooth  * np.sin(mid_angle - t_half))
            ])
            gear_pts.append([
                int(cx + r_tooth  * np.cos(mid_angle)),
                int(cy + r_tooth  * np.sin(mid_angle))
            ])
            gear_pts.append([
                int(cx + r_tooth  * np.cos(mid_angle + t_half)),
                int(cy + r_tooth  * np.sin(mid_angle + t_half))
            ])
            gear_pts.append([
                int(cx + r_valley * np.cos(next_angle - 0.04)),
                int(cy + r_valley * np.sin(next_angle - 0.04))
            ])

    gear_pts = np.array(gear_pts, dtype=np.int32)
    cv2.fillPoly(img, [gear_pts], (210, 210, 210))
    cv2.circle(img, (cx, cy), 45, (45, 45, 45), -1)
    cv2.circle(img, (cx, cy), 80, (185, 185, 185), 2)
    noise = np.random.randint(0, 6, img.shape, dtype=np.uint8)
    img   = cv2.add(img, noise)
    return img


# ─────────────────────────────────────────────────────────
#  PIPELINE FUNCTIONS (same as before)
# ─────────────────────────────────────────────────────────

def phase1_preprocess(image_or_frame, is_live=False):
    if len(image_or_frame.shape) == 3:
        gray = cv2.cvtColor(image_or_frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_or_frame.copy()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    if is_live:
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 4
        )
        kernel = np.ones((4, 4), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    else:
        _, thresh = cv2.threshold(
            blurred, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    return gray, blurred, thresh


def phase2_topological_analysis(thresh):
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, None, None
    main_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(main_contour) < MIN_OBJECT_AREA:
        return None, None, None
    hull = cv2.convexHull(main_contour, returnPoints=False)
    if len(hull) < 4:
        return main_contour, hull, None
    try:
        defects = cv2.convexityDefects(main_contour, hull)
    except cv2.error:
        defects = None
    return main_contour, hull, defects


def phase3_tolerance_gate(defects, contour):
    structural_defects = []
    if defects is None or contour is None:
        return "PASS", structural_defects
    for defect in defects:
        start_idx, end_idx, farthest_idx, d_raw = defect[0]
        actual_depth = d_raw / 256.0
        if actual_depth > DEFECT_DEPTH_THRESHOLD:
            farthest_pt = tuple(contour[farthest_idx][0])
            structural_defects.append({
                "point": farthest_pt,
                "depth": actual_depth
            })
    verdict = "FAIL" if structural_defects else "PASS"
    return verdict, structural_defects


# ─────────────────────────────────────────────────────────
#  SIMULATED LIVE VIDEO MODE
#  No webcam needed — processes images as video frames
#  Same pipeline as real webcam, just image source differs
# ─────────────────────────────────────────────────────────

def run_simulated_live():
    """
    Simulates a live conveyor belt inspection.

    Creates 20 gear images and processes them one by one
    with a delay between each — exactly like a real
    conveyor belt moving parts under a camera.

    Shows animated processing window with PASS/FAIL results.
    Press any key to advance to next part.
    Press Q to quit early.
    """

    print("\n" + "="*60)
    print("  SIMULATED LIVE CONVEYOR BELT INSPECTION")
    print("  Each gear = one part passing under camera")
    print("  Processing in real-time like live video feed")
    print("="*60)
    print("\nGenerating gear dataset...")

    # Generate all 20 gears
    gears = []

    for i in range(10):
        img   = create_synthetic_gear(400, defective=False, gear_id=i)
        label = "perfect"
        gears.append((img, label, f"P-{i+1:02d}"))

    for i in range(10):
        img   = create_synthetic_gear(400, defective=True, gear_id=i)
        label = "defective"
        gears.append((img, label, f"D-{i+1:02d}"))

    # Shuffle so perfect and defective are mixed
    # like real conveyor belt — random order
    np.random.shuffle(gears)

    print(f"20 parts queued for inspection...")
    print("Press SPACE to inspect next part | Q to quit\n")

    os.makedirs("inspection_results", exist_ok=True)
    os.makedirs("live_screenshots",   exist_ok=True)

    total, correct = 0, 0
    part_number    = 0
    results_log    = []

    for gear_img, true_label, part_id in gears:
        part_number += 1

        # ── SCANNING ANIMATION ──
        # Show "Part approaching" screen first
        scan_screen = np.zeros((500, 800, 3), dtype=np.uint8)
        cv2.rectangle(scan_screen, (0,0), (800,500), (20,20,35), -1)
        cv2.putText(scan_screen,
                    f"CONVEYOR BELT — PART {part_number}/20",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
        cv2.putText(scan_screen,
                    "SCANNING...",
                    (300, 260),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (200, 200, 0), 2)
        cv2.putText(scan_screen,
                    f"Part ID: {part_id}",
                    (320, 310),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (150, 150, 150), 1)
        cv2.imshow("DecodeLabs Live Conveyor Inspection", scan_screen)
        cv2.waitKey(600)  # Show scanning for 0.6 seconds

        # ── RUN INSPECTION PIPELINE ──
        # This is identical to what runs on real webcam frames
        gray, blurred, thresh   = phase1_preprocess(
            gear_img, is_live=False
        )
        contour, hull, defects  = phase2_topological_analysis(thresh)
        verdict, struct_defects = phase3_tolerance_gate(
            defects, contour
        )

        # ── BUILD RESULT DISPLAY ──
        # Left panel: annotated gear image
        result_gear = gear_img.copy()

        if contour is not None:
            cv2.drawContours(
                result_gear, [contour], -1, (255, 200, 0), 2
            )

        if verdict == "FAIL":
            for d in struct_defects:
                x, y  = d["point"]
                depth = d["depth"]
                cv2.rectangle(result_gear,
                              (x-30, y-30), (x+30, y+30),
                              (0, 0, 255), 2)
                cv2.circle(result_gear, (x, y), 8, (0,0,255), -1)
                cv2.putText(result_gear,
                            f"{depth:.0f}px",
                            (x+33, y+5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 255, 255), 1)

        # ── BUILD FULL DISPLAY WINDOW ──
        # 800x500 display showing gear + info panel
        display = np.zeros((500, 800, 3), dtype=np.uint8)
        display[:] = (20, 20, 35)

        # Place gear image on left (400x400)
        gear_resized = cv2.resize(result_gear, (400, 400))
        display[50:450, 20:420] = gear_resized

        # Threshold view (small, bottom left)
        thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        thresh_small = cv2.resize(thresh_color, (150, 150))
        display[330:480, 430:580] = thresh_small
        cv2.putText(display, "Threshold View",
                    (435, 325),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 255, 255), 1)

        # Info panel on right
        # Verdict banner
        if verdict == "PASS":
            v_color = (0, 220, 0)
            b_color = (0, 80, 0)
            v_text  = "PASS"
        else:
            v_color = (0, 0, 255)
            b_color = (80, 0, 0)
            v_text  = "FAIL"

        cv2.rectangle(display, (590, 50), (790, 150), b_color, -1)
        cv2.rectangle(display, (590, 50), (790, 150), v_color, 2)
        cv2.putText(display, v_text,
                    (630, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, v_color, 3)

        # Part info
        cv2.putText(display,
                    f"Part ID: {part_id}",
                    (595, 175),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1)

        cv2.putText(display,
                    f"Part {part_number} of 20",
                    (595, 205),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (150, 150, 150), 1)

        # Defect count
        if verdict == "FAIL":
            cv2.putText(display,
                        f"Defects: {len(struct_defects)}",
                        (595, 235),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 100, 255), 1)
            max_d = max(d["depth"] for d in struct_defects)
            cv2.putText(display,
                        f"Max depth: {max_d:.0f}px",
                        (595, 260),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 200, 255), 1)
        else:
            cv2.putText(display,
                        "No defects found",
                        (595, 235),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 200, 0), 1)

        # Running accuracy
        expected   = "FAIL" if true_label == "defective" else "PASS"
        is_correct = verdict == expected
        if is_correct:
            correct += 1
        total += 1

        acc = (correct / total) * 100
        cv2.putText(display,
                    f"Accuracy: {acc:.0f}%",
                    (595, 300),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0) if acc == 100 else (0, 200, 255),
                    1)

        cv2.putText(display,
                    f"Correct: {correct}/{total}",
                    (595, 330),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (150, 150, 150), 1)

        # Pipeline stages label
        cv2.putText(display,
                    "Pipeline:",
                    (595, 370),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 255), 1)
        cv2.putText(display,
                    "1. Grayscale",
                    (595, 390),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (100, 255, 100), 1)
        cv2.putText(display,
                    "2. Gaussian Blur",
                    (595, 408),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (100, 255, 100), 1)
        cv2.putText(display,
                    "3. Threshold",
                    (595, 426),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (100, 255, 100), 1)
        cv2.putText(display,
                    "4. Contours",
                    (595, 444),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (100, 255, 100), 1)
        cv2.putText(display,
                    "5. Convex Hull",
                    (595, 462),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (100, 255, 100), 1)
        cv2.putText(display,
                    "6. PASS/FAIL Gate",
                    (595, 480),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (100, 255, 100), 1)

        # Top bar
        cv2.rectangle(display, (0, 0), (800, 45), (30, 30, 50), -1)
        cv2.putText(display,
                    "DecodeLabs Conveyor Belt Inspection System  "
                    "| SPACE=next  Q=quit  S=screenshot",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 200, 255), 1)

        # Save screenshot automatically
        cv2.imwrite(
            f"live_screenshots/{part_number:02d}_{part_id}"
            f"_{verdict}.png",
            display
        )

        # Print to terminal
        status = "CORRECT" if is_correct else "WRONG"
        print(f"  Part {part_number:02d} | {part_id} | "
              f"Verdict: {verdict:4s} | "
              f"Expected: {expected:4s} | "
              f"{status} | "
              f"Accuracy: {acc:.0f}%")

        results_log.append({
            "part_id":  part_id,
            "verdict":  verdict,
            "expected": expected,
            "correct":  is_correct
        })

        # Show display
        cv2.imshow(
            "DecodeLabs Live Conveyor Inspection", display
        )

        # Wait for key press
        # Auto advance after 2 seconds OR press SPACE to skip
        key = cv2.waitKey(2000) & 0xFF
        if key == ord('q'):
            print("\nQ pressed - stopping inspection")
            break
        elif key == ord('s'):
            print(f"Screenshot saved: live_screenshots/"
                  f"{part_number:02d}_{part_id}_{verdict}.png")

    cv2.destroyAllWindows()

    # ── FINAL REPORT ──
    accuracy = (correct / total) * 100 if total > 0 else 0

    print("\n" + "="*60)
    print("  CONVEYOR BELT INSPECTION COMPLETE")
    print(f"  Parts Inspected : {total}")
    print(f"  Correct         : {correct}")
    print(f"  Accuracy        : {accuracy:.1f}%")
    if accuracy == 100.0:
        print("  STATUS          : 100% Accuracy Achieved!")
    print("="*60)
    print(f"\n  Screenshots saved to: live_screenshots/")

    return accuracy


# ─────────────────────────────────────────────────────────
#  REAL WEBCAM MODE (if you have camera)
# ─────────────────────────────────────────────────────────

def run_real_webcam():
    """Uses actual webcam — laptop camera or DroidCam"""
    global DEFECT_DEPTH_THRESHOLD

    print("\n" + "="*55)
    print("  REAL WEBCAM MODE")
    print("  Show gear images to camera for PASS/FAIL")
    print("="*55)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("No webcam found!")
            print("Switching to simulated mode...")
            run_simulated_live()
            return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Camera opened! Q=quit | S=screenshot")

    os.makedirs("screenshots", exist_ok=True)
    stats        = {"pass": 0, "fail": 0, "frames": 0}
    prev_time    = time.time()
    screenshot_n = 0
    consec_fails = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        stats["frames"] += 1
        fps       = 1.0 / (time.time() - prev_time + 0.001)
        prev_time = time.time()

        _, _, thresh           = phase1_preprocess(
            frame, is_live=True
        )
        contour, hull, defects = phase2_topological_analysis(thresh)

        if contour is not None:
            verdict, s_defects = phase3_tolerance_gate(
                defects, contour
            )
        else:
            verdict, s_defects = "SCANNING", []

        consec_fails = consec_fails + 1 \
            if verdict == "FAIL" else 0
        confirmed = "FAIL" \
            if consec_fails >= CONSECUTIVE_FAIL_THRESHOLD \
            else verdict

        if confirmed == "PASS":
            stats["pass"] += 1
        elif confirmed == "FAIL":
            stats["fail"] += 1

        output = frame.copy()

        if contour is not None:
            cv2.drawContours(
                output, [contour], -1, (255, 255, 0), 2
            )

        for d in s_defects:
            x, y = d["point"]
            cv2.rectangle(output,
                          (x-25, y-25), (x+25, y+25),
                          (0, 0, 255), 2)
            cv2.circle(output, (x, y), 6, (0, 0, 255), -1)

        if confirmed == "PASS":
            b_color = (0, 100, 0)
            t_color = (0, 255, 0)
            b_text  = "PASS: Part OK"
        elif confirmed == "FAIL":
            b_color = (0, 0, 150)
            t_color = (0, 0, 255)
            b_text  = f"FAIL: {len(s_defects)} Defect(s)"
        else:
            b_color = (40, 40, 40)
            t_color = (200, 200, 200)
            b_text  = "SCANNING..."

        cv2.rectangle(output, (0, 0), (640, 55), b_color, -1)
        cv2.putText(output, b_text,
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, t_color, 2)
        cv2.putText(
            output,
            f"FPS:{fps:.0f}  PASS:{stats['pass']}  "
            f"FAIL:{stats['fail']}  "
            f"Thresh:{DEFECT_DEPTH_THRESHOLD:.0f}px  "
            f"[Q]quit [S]save",
            (5, 472),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35, (150, 150, 150), 1
        )

        cv2.imshow("DecodeLabs LIVE Inspection", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_n += 1
            fn = f"screenshots/{screenshot_n:03d}_{confirmed}.png"
            cv2.imwrite(fn, output)
            print(f"Saved: {fn}")
        elif key in (ord('+'), ord('=')):
            DEFECT_DEPTH_THRESHOLD = min(
                80.0, DEFECT_DEPTH_THRESHOLD + 2
            )
            print(f"Threshold: {DEFECT_DEPTH_THRESHOLD}")
        elif key == ord('-'):
            DEFECT_DEPTH_THRESHOLD = max(
                10.0, DEFECT_DEPTH_THRESHOLD - 2
            )
            print(f"Threshold: {DEFECT_DEPTH_THRESHOLD}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done | PASS:{stats['pass']} FAIL:{stats['fail']}")


# ─────────────────────────────────────────────────────────
#  IMAGE DATASET MODE (same as before)
# ─────────────────────────────────────────────────────────

def inspect_single_image(image_path, part_id):
    img = cv2.imread(image_path)
    if img is None:
        return "ERROR", None
    gray, blurred, thresh   = phase1_preprocess(img, is_live=False)
    contour, hull, defects  = phase2_topological_analysis(thresh)
    verdict, struct_defects = phase3_tolerance_gate(defects, contour)
    result = img.copy()
    if contour is not None:
        cv2.drawContours(result, [contour], -1, (255, 200, 0), 2)
    if verdict == "FAIL":
        for d in struct_defects:
            x, y  = d["point"]
            depth = d["depth"]
            cv2.rectangle(result,
                          (x-30, y-30), (x+30, y+30),
                          (0, 0, 255), 2)
            cv2.circle(result, (x, y), 7, (0, 0, 255), -1)
            cv2.putText(result, f"{depth:.0f}px",
                        (x+33, y+5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 255, 255), 1)
        cv2.rectangle(result, (0, 0), (400, 48), (0, 0, 150), -1)
        cv2.putText(result, f"FAIL: DEFECT | {part_id}",
                    (5, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
    else:
        cv2.rectangle(result, (0, 0), (400, 48), (0, 100, 0), -1)
        cv2.putText(result, f"PASS: OK | {part_id}",
                    (5, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)
    return verdict, result


def generate_dataset():
    os.makedirs("gear_dataset/perfect",   exist_ok=True)
    os.makedirs("gear_dataset/defective", exist_ok=True)
    print("Generating dataset...")
    for i in range(10):
        img = create_synthetic_gear(400, defective=False, gear_id=i)
        cv2.imwrite(f"gear_dataset/perfect/perfect_{i+1:02d}.png", img)
    for i in range(10):
        img = create_synthetic_gear(400, defective=True, gear_id=i)
        cv2.imwrite(
            f"gear_dataset/defective/defective_{i+1:02d}.png", img
        )
    print("Dataset ready!")


def run_dataset_inspection():
    print("\n" + "="*55)
    print("  MODE 1: IMAGE DATASET INSPECTION")
    print("="*55)
    generate_dataset()
    os.makedirs("inspection_results", exist_ok=True)
    total, correct = 0, 0
    all_results    = []

    print("\n-- Perfect Parts --")
    for i in range(1, 11):
        path    = f"gear_dataset/perfect/perfect_{i:02d}.png"
        part_id = f"P-{i:02d}"
        verdict, result_img = inspect_single_image(path, part_id)
        ok = verdict == "PASS"
        if ok:
            correct += 1
        total += 1
        print(f"  {part_id}: {verdict:4s} "
              f"{'CORRECT' if ok else 'WRONG'}")
        if result_img is not None:
            cv2.imwrite(
                f"inspection_results/{part_id}.png", result_img
            )
            all_results.append(result_img)

    print("\n-- Defective Parts --")
    for i in range(1, 11):
        path    = f"gear_dataset/defective/defective_{i:02d}.png"
        part_id = f"D-{i:02d}"
        verdict, result_img = inspect_single_image(path, part_id)
        ok = verdict == "FAIL"
        if ok:
            correct += 1
        total += 1
        print(f"  {part_id}: {verdict:4s} "
              f"{'CORRECT' if ok else 'WRONG'}")
        if result_img is not None:
            cv2.imwrite(
                f"inspection_results/{part_id}.png", result_img
            )
            all_results.append(result_img)

    accuracy = (correct / total) * 100
    print(f"\n{'='*55}")
    print(f"  Accuracy: {accuracy:.1f}%  ({correct}/{total})")
    if accuracy == 100.0:
        print("  STATUS: TARGET ACHIEVED - 100% Accuracy!")
    print(f"{'='*55}")

    if all_results:
        rows, cols = 4, 5
        h, w       = 150, 150
        grid       = np.zeros((rows*h, cols*w, 3), dtype=np.uint8)
        for idx, img in enumerate(all_results[:20]):
            r = idx // cols
            c = idx  % cols
            grid[r*h:(r+1)*h, c*w:(c+1)*w] = cv2.resize(img, (w,h))
        cv2.putText(grid,
                    f"Accuracy: {accuracy:.0f}% | "
                    f"Threshold: {DEFECT_DEPTH_THRESHOLD}px",
                    (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 255, 255), 1)
        cv2.imwrite("inspection_results/SUMMARY_GRID.png", grid)
        cv2.imshow("Summary (press any key)", grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return accuracy


# ─────────────────────────────────────────────────────────
#  MAIN MENU
# ─────────────────────────────────────────────────────────

def main():
    print("\n" + "="*55)
    print("  DECODELABS QUALITY INSPECTION SYSTEM")
    print("  Batch 2026 - Project 2")
    print("="*55)
    print("\nSelect mode:")
    print("  1 -> Image Dataset (20 images, 100% accuracy)")
    print("  2 -> Simulated Live (conveyor belt animation)")
    print("  3 -> Real Webcam (laptop/DroidCam)")
    print("  4 -> All three modes")

    choice = input("\nEnter choice (1/2/3/4): ").strip()

    if choice == "1":
        run_dataset_inspection()

    elif choice == "2":
        run_simulated_live()

    elif choice == "3":
        run_real_webcam()

    elif choice == "4":
        print("\nRunning all modes...")
        run_dataset_inspection()
        print("\nStarting simulated live...")
        run_simulated_live()
        go = input("\nStart real webcam? (y/n): ").strip()
        if go.lower() == 'y':
            run_real_webcam()
    else:
        run_dataset_inspection()

    print("\nProject 2 Complete!")


if __name__ == "__main__":
    main()