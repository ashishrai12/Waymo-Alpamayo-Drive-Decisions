# Alpamayo R1 Autonomous Driving Demo

A minimal demo showcasing Alpamayo R1's video-language-action reasoning capabilities using Waymo Open Dataset camera data.

## Overview

This demo demonstrates high-level autonomous driving decision making through:
- **Perception**: Scene analysis from camera frames
- **Reasoning**: Language-based decision making
- **Action**: Discrete driving actions
- **Explanation**: Human-readable reasoning

## Features

- Modular Python implementation
- Mock Alpamayo R1 interface (easily swappable with real model)
- OpenCV-based visualization with frame-by-frame playback
- JSON-structured decision outputs
- Urban driving scenarios (intersections, pedestrians, traffic)

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install opencv-python numpy
   ```

## Usage

### Prepare Data

1. Download Waymo Open Dataset scenes
2. Extract front camera video clips to MP4 format
3. Place video files in the `data/` folder

### Run Demo

```bash
python main.py --video_path data/your_waymo_video.mp4 --fps 1
```

Example with a sample video:
```bash
python main.py --video_path data/sample_urban_driving.mp4 --fps 2
```

Options:
- `--video_path`: Path to Waymo video file (required)
- `--fps`: Frames per second to sample (default: 1)
- `--mock`: Use mock Alpamayo policy (default: True)

### Controls

- **Space**: Pause/Play
- **N**: Next frame (when paused)
- **P**: Previous frame (when paused)
- **Q** or **ESC**: Quit

## Architecture

### Files

- `main.py`: Entry point and pipeline orchestration
- `data_loader.py`: Video loading and frame sampling
- `alpamayo_policy.py`: Alpamayo R1 interface (mock implementation)
- `decision_schema.py`: JSON schema validation
- `visualize.py`: OpenCV-based UI and overlays

### Decision Format

Each timestep outputs JSON with:
```json
{
  "frame_id": 0,
  "scene_type": "intersection",
  "agents": [{"type": "pedestrian", "position": "crossing"}],
  "traffic_light": "red",
  "hazards": ["pedestrian crossing"],
  "decision": "stop",
  "confidence": 0.89,
  "reason": "Pedestrian detected at crosswalk"
}
```

### Actions

- `accelerate`: Increase speed
- `maintain_speed`: Keep current speed
- `slow_down`: Gradually reduce speed
- `brake`: Hard stop
- `stop`: Come to complete stop
- `yield`: Give way to others

## Extending

### Real Alpamayo Integration

Replace the mock in `alpamayo_policy.py` with actual model calls:

```python
def decide(self, frame, prompt):
    # Process frame with vision model
    # Send to Alpamayo with prompt
    # Return structured JSON
    pass
```

### Additional Data Sources

Modify `data_loader.py` to load from:
- Waymo TFRecords directly
- Other autonomous driving datasets
- Live camera feeds

### Enhanced UI

Replace OpenCV with Streamlit for web-based interface:
- Timeline scrubbing
- Decision history
- Interactive playback

## Dependencies

- opencv-python: Video processing and visualization
- numpy: Array operations

## License

[Add appropriate license]

## Contributing

[Add contribution guidelines]
