# ORB Inspection System

A machine vision inspection system developed with Python and OpenCV for industrial inspection applications.

The system compares live camera images against reference images using ORB feature matching. Inspection is performed according to the selected product model and validated using stable matching logic within a PLC-controlled inspection cycle.

## Features

* ORB feature extraction and matching
* Model-based reference image selection
* Lowe's ratio test for match filtering
* Stable frame validation
* Independent left/right inspection stations
* Manual PASS override
* Session logging
* OPC-UA communication support
* PLC trigger-based inspection workflow
* Real-time dashboard UI
* JSON export for external analytics and monitoring systems

## Technologies

* Python
* OpenCV
* NumPy
* Socket Communication
* OPC-UA
* JSON Data Logging

## Inspection Logic

1. Receive the product model from manual input, model server, or PLC.
2. Load the corresponding reference images for the selected model.
3. Wait for an inspection trigger signal.
4. Capture live images from the inspection cameras.
5. Extract ORB features from both reference and live images.
6. Perform feature matching using Lowe's ratio test.
7. Count valid matches during the inspection cycle.
8. Validate matching stability across multiple consecutive frames.
9. Determine PASS or NG status independently for each inspection station.
10. Store inspection results in JSON format for traceability and external dashboard integration.

## System Architecture

```text
Model Input (Manual / PLC / Server)
                │
                ▼
      Reference Image Loader
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

## Output Example

```json
{
  "model": "00001",
  "L": {
    "status": "PASS",
    "best_match": 8,
    "manual": true,
    "sessions": []
  },
  "R": {
    "status": "PASS",
    "best_match": 10,
    "manual": true,
    "sessions": []
  }
}
```

## Data Integration

Inspection results are automatically exported in JSON format.

The generated log files can be integrated with:

* Node-RED
* Power BI
* MES Systems
* Production Monitoring Dashboards
* Custom Analytics Applications

## Future Improvements

* Real PLC deployment
* Database integration
* Multi-object inspection support
* Deep learning-based inspection models
* Web-based monitoring dashboard

## Author

Developed as a machine vision and industrial automation prototype using OpenCV and ORB feature matching techniques.
