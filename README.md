# Project 2: Automated Quality Inspection (Computer Vision)
## DecodeLabs Internship — Robotics & Automation — Batch 2026
### Author: Fakhir Ali Khan (2023-MC-29)
### Institution: UET Lahore — Mechatronics & Control Engineering

---

## Project Overview
Automated optical inspection system that detects defective
gear parts on a conveyor belt using Computer Vision and OpenCV.
The system replicates a real industrial quality control pipeline
that never blinks and never experiences cognitive fatigue.

---

## Tech Stack
- Python 3.10
- OpenCV (cv2)
- NumPy

---

## IPO Architecture

| Stage | Action | OpenCV Function |
|---|---|---|
| Input | Flatten to grayscale | cv2.cvtColor |
| Pre-process | Smooth and binarize | cv2.GaussianBlur, cv2.threshold |
| Topology | Trace gear boundary | cv2.findContours |
| Measurement | Measure defect gaps | cv2.convexHull, cv2.convexityDefects |
| Output | PASS or FAIL verdict | cv2.rectangle, cv2.putText |

---

## 3 Inspection Modes

Mode 1 - Image Dataset
Processes 20 synthetic gear images (10 perfect + 10 defective)
Achieves 100% sorting accuracy

Mode 2 - Simulated Live Conveyor Belt
Animates each gear being inspected one by one
Simulates real factory conveyor belt pipeline
No physical camera required

Mode 3 - Real Webcam
Live real-time inspection using laptop or phone camera
Show any circular object to get instant PASS or FAIL

---

## How It Works

Normal tooth valley depth  =  ~37px  <  45px threshold  =  PASS
Missing tooth notch depth  = ~105px  >  45px threshold  =  FAIL

The convexity defect depth is the key measurement.
OpenCV returns raw distance scaled by 256.
Must divide by 256.0 to get actual pixel depth.

---

## Results

- Total Parts Inspected : 20
- Correct Verdicts      : 20
- Sorting Accuracy      : 100%
- Status                : TARGET ACHIEVED

---

## How to Run

Step 1 - Install requirements
pip install opencv-python numpy

Step 2 - Run the program
python main_inspection.py

Step 3 - Select mode
1 = Image Dataset
2 = Simulated Live
3 = Real Webcam
4 = All three

---

## Project Structure

decodelabs-project-2/
├── main_inspection.py
├── gear_dataset/
│   ├── perfect/        (10 perfect gear images)
│   └── defective/      (10 defective gear images)
├── inspection_results/
│   ├── P-01.png to P-10.png
│   ├── D-01.png to D-10.png
│   └── SUMMARY_GRID.png
└── live_screenshots/   (simulated conveyor results)

---

## Contact
Name     : Fakhir Ali Khan
Reg No   : 2023-MC-29
Program  : Mechatronics and Control Engineering
Institute: UET Lahore
Internship: DecodeLabs — Robotics and Automation Track
Batch    : 2026
