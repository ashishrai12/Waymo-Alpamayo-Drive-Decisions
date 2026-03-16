import streamlit as st
import cv2
import json
import os
import tempfile
import time

# Ensure the app can find the src module since it might be run from the root
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from alpamayo_demo.core.policy import AlpamayoPolicy

st.set_page_config(
    page_title="Alpamayo R1 Autonomous Driving",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚗 Alpamayo R1 Video-Language-Action Demo")
st.markdown("This application demonstrates high-level autonomous driving decision making using visual frames and simulated AI reasoning.")

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")
policy_type = st.sidebar.radio("Policy Type", ["Mock (Fast)", "Real Model (Requires API Setup)"])
is_mock = policy_type == "Mock (Fast)"

fps_input = st.sidebar.slider("Sampling FPS (Frames per second to analyze)", min_value=1, max_value=10, value=1)
playback_speed = st.sidebar.slider("UI Playback Delay (seconds between frames)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

# Initialize policy in session state to avoid reloading
if 'policy' not in st.session_state or st.session_state.is_mock != is_mock:
    st.session_state.policy = AlpamayoPolicy(mock=is_mock)
    st.session_state.is_mock = is_mock

# Default video path
DEFAULT_VIDEO_PATH = "data/sample_video.mp4"

# Let user choose a file or use the built-in one
video_source_option = st.sidebar.radio("Video Source", ["Use Default Sample Video", "Upload custom MP4"])

video_path_to_use = None
temp_file_obj = None

if video_source_option == "Use Default Sample Video":
    if os.path.exists(DEFAULT_VIDEO_PATH):
        video_path_to_use = DEFAULT_VIDEO_PATH
        st.sidebar.success(f"Found default video: `{DEFAULT_VIDEO_PATH}`")
    else:
        st.sidebar.warning("Default video not found. Run `python scripts/create_sample_video.py` first, or upload a custom one.")
else:
    uploaded_file = st.sidebar.file_uploader("Upload a Waymo dashboard clip (MP4)", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        # Save to temp file to read with OpenCV
        temp_file_obj = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file_obj.write(uploaded_file.read())
        video_path_to_use = temp_file_obj.name
        
if st.button("Start Analysis") and video_path_to_use:
    
    # Load Video
    cap = cv2.VideoCapture(video_path_to_use)
    if not cap.isOpened():
        st.error(f"Failed to open video at {video_path_to_use}")
    else:
        # Calculate frame skip
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps <= 0: orig_fps = 30 # fallback
        frame_interval = max(1, int(orig_fps / fps_input))
        
        st.success("Video loaded successfully. Beginning pipeline...")
        
        # Prepare layout boxes
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Front Camera View")
            video_placeholder = st.empty()
            
        with col2:
            st.subheader("Alpamayo R1 Output")
            decision_placeholder = st.empty()
            
            # Use smaller columns for metrics
            mcol1, mcol2 = st.columns(2)
            metric_decision = mcol1.empty()
            metric_confidence = mcol2.empty()
        
        goal_prompt = """
        You are an autonomous vehicle driving in an urban environment.
        Analyze the current scene from the front camera and decide the next action.
        Consider safety, traffic rules, and smooth driving.
        Output your decision in the specified JSON format.
        """

        frame_count = 0
        analyzed_count = 0
        
        # Progress bar
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Only process scheduled frames
            if frame_count % frame_interval != 0:
                continue
                
            analyzed_count += 1
            
            # Convert frame for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference
            try:
                 decision_json = st.session_state.policy.decide(frame, goal_prompt)
                 decision = json.loads(decision_json)
                 decision['frame_id'] = analyzed_count
            except Exception as e:
                 st.error(f"Inference error on frame {frame_count}: {e}")
                 decision = {"error": str(e)}

            # Update UI
            video_placeholder.image(frame_rgb, use_container_width=True, channels="RGB")
            
            # Format nicely
            if "error" not in decision:
                action_color = "green"
                if decision.get('decision') in ['stop', 'brake']: action_color = "red"
                elif decision.get('decision') in ['slow_down', 'yield']: action_color = "orange"
                
                metric_decision.metric("Action", decision.get('decision', 'N/A').upper())
                metric_confidence.metric("Confidence", f"{decision.get('confidence', 0.0):.1%}")

                decision_placeholder.markdown(f"""
                **Scene:** `{decision.get('scene_type', 'N/A')}`  
                **Traffic Light:** `{decision.get('traffic_light', 'N/A')}`  
                **Reasoning:** _{decision.get('reason', 'N/A')}_
                
                **Detected Hazards:**
                {', '.join(decision.get('hazards', [])) if decision.get('hazards') else 'None'}
                """)
                
                # Full JSON expander
                with decision_placeholder.expander("Show Raw JSON"):
                    st.json(decision)
            
            # Update Progress
            progress_val = min(1.0, float(frame_count) / total_frames) if total_frames > 0 else 0.0
            progress_bar.progress(progress_val)
            
            # Sleep for UI playback effect (unless it's analyzing a very long real video)
            time.sleep(playback_speed)
            
        cap.release()
        st.balloons()
        st.success("Analysis Complete!")

# Cleanup temp files
if temp_file_obj:
    try:
        os.unlink(temp_file_obj.name)
    except:
        pass
