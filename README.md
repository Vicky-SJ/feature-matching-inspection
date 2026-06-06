# ORB Inspection System

A machine vision inspection system developed with Python and OpenCV for industrial inspection applications.

The system compares live camera images against reference images using ORB (Oriented FAST and Rotated BRIEF) feature matching. Inspection is performed according to the selected product model and validated using stable matching logic within a PLC-controlled inspection cycle.

---

## Project Structure

This project consists of two main programs:

### 1. setup_ref.py

Reference image setup utility.

Functions:

* Capture reference images from inspection cameras
* Select Region of Interest (ROI) interactively
* Save reference images to the database folder
* Automatically update ROI configuration values in the inspection program

Workflow:

1. Position the product in front of the camera
2. Capture reference image
3. Select ROI using mouse
4. Save reference image
5. Update ROI configuration automatically

---

### 2. Inspection Program

Main inspection application.

Functions:

* Load model-specific reference images
* Receive inspection trigger from PLC or manual input
* Perform ORB feature matching
* Validate stable matching across multiple frames
* Generate PASS / NG decisions
* Log inspection results
* Display real-time inspection dashboard

---

## Features

* ORB feature extraction and matching
* ROI-based inspection
* Model-based reference image selection
* Lowe's ratio test for match filtering
* Stable frame validation
* Independent left/right inspection stations
* Manual PASS override
* Session logging
* OPC-UA communication support
* PLC trigger-based inspection workflow
* Real-time dashboard UI
* JSON export for analytics and monitoring systems

---

## Technologies

* Python
* OpenCV
* NumPy
* Socket Communication
* OPC-UA
* JSON Data Logging

---

## Inspection Logic

1. Receive the product model from manual input, PLC, or model server.
2. Load the corresponding reference images.
3. Wait for inspection trigger.
4. Capture live images from inspection cameras.
5. Crop image according to configured ROI.
6. Extract ORB features from reference and live images.
7. Perform feature matching using Lowe's ratio test.
8. Count valid feature matches.
9. Validate matching stability across consecutive frames.
10. Determine PASS or NG independently for each inspection station.
11. Store inspection results in JSON format.

---

## System Architecture

```text
Model Input (Manual / PLC / Server)
                │
                ▼
      Reference Image Loader
                │
                ▼
           ROI Cropping
                │
                ▼
       ORB Feature Matching
                │
                ▼
      Stable Frame Validation
                │
                ▼
      PASS / NG Decision Logic
                │
                ▼
      JSON Export & Dashboard
```

---

## Output Example

```json
{
  "model": "00001",
  "L": {
    "status": "PASS",
    "best_match": 8,
    "manual": false
  },
  "R": {
    "status": "PASS",
    "best_match": 10,
    "manual": false
  }
}
```

---

## ROI Guidelines

For best inspection accuracy:

* Select ROI as tightly as possible around the target object.
* Avoid including unnecessary background areas.
* Background features may generate additional ORB keypoints and affect matching consistency.
* Use consistent lighting conditions during reference image capture and inspection.

---

## Limitations

ORB feature matching is effective for textured objects but has several limitations:

### Small Objects

Very small objects may contain too few visual features, resulting in unstable ORB keypoint detection and lower matching accuracy.

### Transparent or Hollow Objects

Objects that allow the background to be visible through them are not ideal for ORB matching.

Examples:

* Transparent parts
* Wire-frame structures
* Objects with large holes or openings

Background features may dominate the matching process and produce inconsistent results.

### Low-Texture Surfaces

Objects with smooth, uniform surfaces provide limited visual features for ORB detection.

Examples:

* Plain plastic surfaces
* Uniform painted parts
* Featureless metal components

Such objects may require alternative inspection methods.

---

## Data Integration

Inspection results are automatically exported in JSON format.

Generated logs can be integrated with:

* Node-RED
* Power BI
* MES Systems
* Production Monitoring Dashboards
* Custom Analytics Applications

---

## Future Improvements

* Real PLC deployment
* Database integration
* Multi-object inspection support
* Feature match visualization
* Web-based monitoring dashboard
* Deep learning-based inspection models

---

## Author

Developed as a machine vision and industrial automation project using OpenCV and ORB feature matching techniques.

This project demonstrates practical applications of computer vision for industrial inspection, model verification, and production-line automation.
