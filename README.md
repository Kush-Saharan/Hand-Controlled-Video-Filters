#  Gesture-Controlled Real-Time Filter System (OpenCV + MediaPipe)

A real-time computer vision project that lets you **control visual filters using hand gestures** and apply **AR effects (like sunglasses)** on your face.

Built using **Python, OpenCV, and MediaPipe**, this project demonstrates gesture interaction, image processing, and augmented reality basics.

---

##  Features

*  **Two Interaction Modes**

  *  **Pointing Mode** → select filters by pointing at UI boxes
  *  **Pinch Mode** → cycle filters by pinching anywhere

*  **Multiple Filters**

  * Sepia
  * Emboss
  * Brightness
  * Duo Tone (improved)
  * TV effect
  * None (original feed)

*  **AR Sunglasses Filter**

  * Face + eye detection using MediaPipe
  * Dynamic scaling based on face size

*  **Real-Time Performance**

  * Live webcam processing
  * Smooth interaction

---

##  Project Structure

```bash
project/
│
├── gesture_control.py          #  Point-based filter selection
├── gesture_control_pinch.py    #  Pinch-based filter switching
│
├── hand_tracking.py
├── hand_tracking_pinch.py
├── utils.py
│
├── filters/
│   ├── brightness.py
│   ├── sepia.py
│   ├── emboss.py
│   ├── duo_tone.py
│   ├── tv_60.py
│   ├── sunglasses.py
│
├── assets/
│   └── sunglasses.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

##  Installation

### 1. Clone the repository

```bash
git clone https://github.com/Kush-Saharan/Hand-Controlled-Video-Filters.git
cd Hand-Controlled-Video-Filters
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

##  Running the Project

###  1. Pointing Mode (UI-based selection)

```bash
python3 gesture_control.py
```

**How it works:**

* Point your index finger at filter boxes on screen
* Selected box gets activated
* Works like a virtual touch interface

---

###  2. Pinch Mode (Gesture-based switching)

```bash
python3 gesture_control_pinch.py
```

**How it works:**

* Pinch (thumb + index finger) anywhere
* Each pinch → switches to next filter
* Minimal UI (only filter name shown)

---

##  Controls

*  **Pinch** → change filter (pinch mode)
*  **Point** → select filter (point mode)
*  **ESC** → exit

---

##  How It Works

###  Hand Tracking

* Uses MediaPipe Hands
* Detects landmarks for thumb & index finger
* Computes distance → used for pinch detection

---

###  Gesture Logic

**Point Mode:**

* Index finger position → mapped to UI boxes
* Selection based on bounding box collision

**Pinch Mode:**

* Distance between thumb & index finger
* If below threshold → trigger filter switch

---

###  Filter Pipeline

Each frame:

1. Capture webcam feed
2. Detect hand gesture
3. Update current filter
4. Apply filter
5. Display result

---

###  AR Sunglasses Filter

* Uses MediaPipe FaceMesh
* Detects eye landmarks
* Computes:

  * eye distance → scaling
  * angle → rotation
* Overlays PNG using alpha blending

---

##  Key Concepts

* Computer Vision (OpenCV)
* Hand Landmark Detection
* Face Landmark Tracking
* Alpha Blending (AR overlays)
* Gesture-based UI systems