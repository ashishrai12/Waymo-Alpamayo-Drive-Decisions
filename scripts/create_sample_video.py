import cv2
import numpy as np
import os
import argparse

def create_synthetic_video(output_path, num_frames=30, fps=1):
    """
    Creates a simple synthetic video mimicking a front-facing dashcam with a 'pedestrian'.
    This is useful for testing without needing to download a real Waymo segment.
    """
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Creating synthetic video at {output_path} with {num_frames} frames at {fps} fps...")

    for i in range(num_frames):
        # Create a basic 'road' scene
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Sky: light blue
        frame[0:height//2, :] = (235, 206, 135)
        # Road: dark gray
        frame[height//2:height, :] = (80, 80, 80)
        
        # Draw some 'lane lines'
        cv2.line(frame, (width//2, height), (width//2, height//2), (255, 255, 255), 5)
        
        # Draw a moving 'pedestrian' (a red rectangle)
        ped_x = int(width * (i / num_frames))
        ped_y = height//2 + 50
        cv2.rectangle(frame, (ped_x, ped_y), (ped_x + 30, ped_y + 80), (0, 0, 255), -1)
        
        # Draw a traffic light (green then yellow then red)
        tl_x, tl_y = width - 100, 50
        cv2.rectangle(frame, (tl_x, tl_y), (tl_x + 30, tl_y + 90), (50, 50, 50), -1)
        
        if i < num_frames // 3:
            # Green
            cv2.circle(frame, (tl_x + 15, tl_y + 75), 10, (0, 255, 0), -1)
        elif i < 2 * (num_frames // 3):
            # Yellow
            cv2.circle(frame, (tl_x + 15, tl_y + 45), 10, (0, 255, 255), -1)
        else:
            # Red
            cv2.circle(frame, (tl_x + 15, tl_y + 15), 10, (0, 0, 255), -1)

        # Add text overlay
        cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)

    out.release()
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create synthetic sample video for Waymo-Alpamayo Demo")
    parser.add_argument("--output", type=str, default="data/sample_video.mp4", help="Output path")
    args = parser.parse_args()
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    create_synthetic_video(args.output)
