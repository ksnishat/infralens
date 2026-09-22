# InfraLens API Documentation

> OpenAPI 3.0 Specification for InfraLens Computer Vision Backend API

## Base URL

```
http://localhost:8000
```

## Endpoints

### `GET /`

Root endpoint.

**Response:**
```json
{
  "service": "InfraLens API",
  "version": "1.0.0",
  "description": "Industrial Rust Detection System"
}
```

---

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2026-10-02T10:30:00.000Z"
}
```

---

### `POST /detect`

Detect rust and corrosion defects in an image using YOLOv8.

**Request (Multipart Form):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | Image file (JPG/PNG/TIFF, max 10MB) |
| `conf` | float | No | Confidence threshold (default: 0.25) |
| `iou` | float | No | IoU threshold for NMS (default: 0.45) |

**Response Schema:**

| Field | Type | Description |
|-------|------|-------------|
| `detections` | array[object] | List of detected defects |
| `num_detections` | integer | Total number of detections |
| `processing_time_ms` | float | Processing time in milliseconds |
| `annotated_image` | string | Base64 encoded image with detection boxes |

**Detection Object:**
| Field | Type | Description |
|-------|------|-------------|
| `class_name` | string | Defect class (e.g., "rust", "corrosion") |
| `class_id` | integer | Class ID |
| `confidence` | float | Confidence score (0.0 to 1.0) |
| `bbox` | array[float] | Bounding box [x1, y1, x2, y2] |
| `center` | array[float] | Center coordinates [x, y] |

---

### `POST /analyze`

Analyze image and generate AI-powered safety report.

**Request:** Same as `/detect`

**Response Schema:**

| Field | Type | Description |
|-------|------|-------------|
| `detections` | array | Same as `/detect` |
| `num_detections` | integer | Total detections |
| `severity` | string | Overall severity ("low", "medium", "high") |
| `risk_assessment` | string | Risk level assessment |
| `safety_recommendations` | array[string] | List of safety recommendations |
| `report` | string | Full technical safety report |
| `processing_time_ms` | float | Processing time in ms |

---

### `POST /report`

Generate a comprehensive technical report (ISO-compliant).

**Request:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | Image file |
| `language` | string | No | Report language ("en" or "de") |

**Response Schema:**

| Field | Type | Description |
|-------|------|-------------|
| `report` | string | Generated technical report |
| `language` | string | Report language |
| `risk_rating` | string | Risk rating |
| `recommendations` | array[string] | Recommendations |
| `processing_time_ms` | float | Processing time |

---

### `GET /docs` and `GET /openapi.json`

Interactive API docs and OpenAPI spec.

## Metrics

- `infralens_detection_seconds` - YOLO detection latency histogram
- `infralens_detections_total` - Total detections counter (with class labels)
- `infralens_report_generation_seconds` - Report generation latency
- `infralens_api_requests_total` - Total API request counter