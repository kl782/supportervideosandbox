import streamlit as st
import os
import cv2
import json
import base64
import requests
import subprocess
import textwrap
import re
import tempfile
from openai import OpenAI
from anthropic import Anthropic
from PIL import Image
import numpy as np
import time
from pathlib import Path

def create_text_overlay(input_path, output_path, text, segment_id):
    """ Create a text overlay with font size proportional to video dimensions """
    try:
        # First, get video dimensions
        probe_cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=,:p=0 "{input_path}"'
        dimensions = subprocess.check_output(probe_cmd, shell=True).decode('utf-8').strip().split(',')
        
        if len(dimensions) >= 2:
            width, height = int(dimensions[0]), int(dimensions[1])
            st.info(f"Video dimensions for segment {segment_id}: {width}x{height}")
        else:
            # Default dimensions if we can't detect
            width, height = 720, 1280
            st.warning(f"Could not detect video dimensions for segment {segment_id}, using defaults: {width}x{height}")
        
        # Calculate font size as a percentage of video height (3.5% of height)
        font_size = max(24, int(height * 0.035))
        # Calculate box border width based on font size
        box_border = max(5, int(font_size * 0.15))
        
        # Calculate y position (center of video)
        y_position = f"h/2"
        
        # Clean and prepare text
        # Remove apostrophes and normalize quotes
        cleaned_text = re.sub(r"['']", "", text)
        cleaned_text = re.sub(r'["""]', '"', cleaned_text)
        
        # Calculate optimal line width based on video width (approx 40% of width in characters)
        line_width = max(20, int(width * 0.4 / font_size))
        wrapped_lines = textwrap.wrap(cleaned_text, width=line_width)
        joined_text = '\n'.join(wrapped_lines)
        
        # Escape special characters for ffmpeg
        safe_text = (joined_text
            .replace(":", "\\:")
            .replace("%", "\\%")
            .replace(",", "\\,")
            .replace("[", "\\[")
            .replace("]", "\\]")
            .replace("(", "\\(")
            .replace(")", "\\)")
            .replace("{", "\\{")
            .replace("}", "\\}")
            .replace("\"", "\\\""))
        
        # Create text overlay command
        # Check if Coolvetica.otf exists, otherwise use a system font
        font_file = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Default system font on many Linux distros
        coolvetica_path = os.path.join(os.getcwd(), "Coolvetica.otf")
        if os.path.exists(coolvetica_path):
            font_file = coolvetica_path
        
        text_cmd = (
            f'ffmpeg -y -i "{input_path}" -vf '
            f'"drawtext=text=\'{safe_text}\':'
            f'fontfile=\'{font_file}\':'
            f'fontcolor=white:'
            f'fontsize={font_size}:'
            f'box=1:'
            f'boxcolor=0x8B0000@0.8:'
            f'boxborderw={box_border}:'
            f'x=(w-text_w)/2:'
            f'y={y_position}:'
            f'line_spacing=8:'
            f'shadowcolor=black@0.7:'
            f'shadowx=2:'
            f'shadowy=2" '
            f'-c:a copy "{output_path}"'
        )
        
        st.info(f"Creating text overlay for segment {segment_id} with font size {font_size}")
        subprocess.call(text_cmd, shell=True)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        else:
            st.error(f"Failed to create text overlay for segment {segment_id}")
            return input_path
    except Exception as e:
        st.error(f"Error creating text overlay: {str(e)}")
        return input_path

def generate_trailer_plan(videos_data):
    """Generate a trailer plan using Claude via direct HTTP request"""
    try:
        # Get petition text from session state
        petition_text = st.session_state.get("petition_text", "")
        
        # Prepare prompt for Claude
        prompt = f"""
        Create a compelling trailer for the given Change.org petition.
        
        PETITION TEXT:
        {petition_text}
        
        I have the following videos to work with:
        
        {json.dumps(videos_data, indent=2)}
        Each video transcript contains timestamps in the format [H:MM:SS] at the beginning of each line. These timestamps are crucial for trimming the video clips.
        
        Based on these videos and the petition text, I need you to:
        
        1. Analyze the content and determine the core message of the petition
        2. Plan a 30-60 second trailer that effectively communicates this message
        3. Decide which video clips to use and in what order
        4. IMPORTANT: For each clip, specify:
           a. The video ID
           b. The exact start timestamp (in H:MM:SS format) from the transcript
           c. The exact end timestamp or duration
           d. The text overlay content
        
        
        Your response should have two parts:
        
        PART 1: Your reasoning process and analysis of the videos. Explain your thinking about how to create the most effective trailer that supports the petition's goals.
        
PART 2: A structured JSON trailer plan with segments array like this:
        
        {{
          "segments": [
            {{
              "type": "video",
              "video_id": 2,
              "start_time": "0:01:24",
              "duration": 8,
              "text": "Our community needs clean water"
            }},
            {{
              "type": "transition",
              "text": "Join our cause",
              "duration": 3
            }},
            ...
          ]
        }}
        
        Make sure all video segments specify a valid video_id, start_time (in H:MM:SS format from transcript), duration (in seconds), and text overlay.
        """
        
        
        # Direct API call using requests
        import requests
        
        headers = {
            "x-api-key": st.secrets["anthropic"]["api_key"],
            "content-type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": "claude-3-7-sonnet-20250219",
            "max_tokens": 4000,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            st.error(f"API call failed with status code: {response.status_code}")
            st.error(f"Response: {response.text}")
            return "Failed to generate trailer plan", f"Error: API call failed with status code {response.status_code}"
        
        response_json = response.json()
        content = response_json.get('content', [{}])[0].get('text', str(response_json))
        
        # Split the response into reasoning and plan parts
        parts = content.split("PART 2:")
        if len(parts) >= 2:
            reasoning = parts[0].replace("PART 1:", "").strip()
            plan = parts[1].strip()
        else:
            # If the response isn't properly structured, use the whole response as the plan
            reasoning = "Claude's reasoning could not be separated from the plan."
            plan = content
        
        return plan, reasoning
    
    except Exception as e:
        st.error(f"Error generating trailer plan: {str(e)}")
        return "Failed to generate trailer plan", f"Error: {str(e)}"


def create_trailer_segment(video_info, clip_info, output_dir, segment_id):
    """Create a segment for the trailer"""
    try:
        # Get the input video path
        input_path = video_info["path"]
        segment_start = clip_info.get("start_time", "0:00:00")
        segment_duration = clip_info.get("duration", 10)
        
        # Convert time format to seconds if needed
        if isinstance(segment_start, str) and ":" in segment_start:
            h, m, s = segment_start.split(":")
            start_seconds = int(h) * 3600 + int(m) * 60 + int(s)
        else:
            start_seconds = float(segment_start)
        
        # Create output path for the segment
        temp_output_path = os.path.join(output_dir, f"segment_{segment_id}_temp.mp4")
        final_output_path = os.path.join(output_dir, f"segment_{segment_id}.mp4")
        
        # Extract the segment
        extract_cmd = f'ffmpeg -y -ss {start_seconds} -i "{input_path}" -t {segment_duration} -c copy "{temp_output_path}"'
        subprocess.call(extract_cmd, shell=True)
        
        # Add text overlay if specified
        overlay_text = clip_info.get("text", "")
        if overlay_text and os.path.exists(temp_output_path) and os.path.getsize(temp_output_path) > 0:
            return create_text_overlay(temp_output_path, final_output_path, overlay_text, segment_id)
        elif os.path.exists(temp_output_path) and os.path.getsize(temp_output_path) > 0:
            # If no text overlay needed, just move the file
            os.rename(temp_output_path, final_output_path)
            return final_output_path
        else:
            st.error(f"Failed to create segment {segment_id}")
            return None
    
    except Exception as e:
        st.error(f"Error creating trailer segment: {str(e)}")
        return None

def create_transition_slide(text, duration, output_path, slide_id):
    """Create a transition slide with text using PIL instead of ffmpeg"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        import moviepy.editor as mp
        
        # Create a dark background image
        width, height = 1280, 720  # Standard HD resolution
        background_color = (40, 40, 40)  # Dark gray
        
        img = Image.new('RGB', (width, height), background_color)
        draw = ImageDraw.Draw(img)
        
        # Try to use a nice font, fall back to default
        try:
            # Try to use a nice font if available
            font = ImageFont.truetype("Arial", 40)
        except:
            # Fall back to default font
            font = ImageFont.load_default()
        
        # Wrap text to fit in the image
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            # Estimate width with default font - approximate
            if len(test_line) * 15 < width - 100:  # rough estimate for width
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                
        if current_line:
            lines.append(' '.join(current_line))
        
        # Draw text centered
        text_color = (255, 255, 255)  # White
        y_position = height // 2 - (len(lines) * 50) // 2
        
        for line in lines:
            # Estimate the position to center the text
            line_width = len(line) * 15  # rough estimate
            x_position = (width - line_width) // 2
            
            draw.text((x_position, y_position), line, fill=text_color, font=font)
            y_position += 50
        
        # Save temporary image
        temp_img_path = os.path.join(os.path.dirname(output_path), f"slide_{slide_id}.png")
        img.save(temp_img_path)
        
        # Create video from still image using moviepy
        clip = mp.ImageClip(temp_img_path, duration=duration)
        clip.write_videofile(output_path, fps=24, codec='libx264', audio=False)
        
        # Clean up temporary image
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        else:
            st.error(f"Failed to create transition slide {slide_id}")
            return None
    
    except Exception as e:
        st.error(f"Error creating transition slide: {str(e)}")
        return None

def create_trailer_from_plan(videos, plan, output_dir):
    """Create the trailer from the plan"""
    try:
        # Parse the plan (assuming it's in a format we can work with)
        # This will depend on how Claude structures the plan
        segments = []
        
        # For now, using a simple parsing approach
        # This would need to be adapted based on the actual format Claude produces
        if isinstance(plan, str):
            # Try to parse as JSON first
            try:
                plan_data = json.loads(plan)
                if isinstance(plan_data, list):
                    segments = plan_data
                elif isinstance(plan_data, dict) and "segments" in plan_data:
                    segments = plan_data["segments"]
                else:
                    # Handle other structured formats
                    segments = [{"text": plan, "type": "transition", "duration": 10}]
            except:
                # If not JSON, parse as structured text
                # Simple parsing based on lines and indentation
                lines = plan.strip().split("\n")
                current_segment = None
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check if line starts a new segment
                    if line.startswith("Segment") or line.startswith("Clip") or line.startswith("1.") or line.startswith("- "):
                        if current_segment:
                            segments.append(current_segment)
                        
                        # Start a new segment
                        current_segment = {
                            "type": "video" if "Video" in line else "transition",
                            "text": "",
                            "duration": 5
                        }
                        
                        # Try to extract video ID if present
                        video_id_match = re.search(r"Video (\d+)", line)
                        if video_id_match:
                            current_segment["video_id"] = int(video_id_match.group(1)) - 1
                        
                        # Try to extract duration if present
                        duration_match = re.search(r"(\d+)(?:\.\d+)? seconds", line)
                        if duration_match:
                            current_segment["duration"] = float(duration_match.group(1))
                    
                    # Check if line contains text overlay information
                    elif "Text:" in line or "Overlay:" in line:
                        text_parts = line.split(":", 1)
                        if len(text_parts) > 1:
                            current_segment["text"] = text_parts[1].strip().strip('"')
                    
                    # If it's a continuation of the current segment
                    elif current_segment:
                        # Add to the description or details
                        if "description" not in current_segment:
                            current_segment["description"] = line
                
                # Add the last segment
                if current_segment:
                    segments.append(current_segment)
        
        # Create segments
        segment_paths = []
        
        for i, segment in enumerate(segments):
            segment_type = segment.get("type", "video")
            
            if segment_type == "video":
                video_id = segment.get("video_id", 0)
                # Ensure video_id is within range
                video_id = min(video_id, len(videos) - 1) if videos else 0
                
                if videos:
                    video_info = videos[video_id]
                    clip_info = {
                        "start_time": segment.get("start_time", "0:00:00"),
                        "duration": segment.get("duration", 5),
                        "text": segment.get("text", "")
                    }
                    
                    segment_path = create_trailer_segment(video_info, clip_info, output_dir, i)
                    if segment_path:
                        segment_paths.append(segment_path)
            
            elif segment_type == "transition":
                text = segment.get("text", "")
                duration = segment.get("duration", 3)
                
                transition_path = os.path.join(output_dir, f"transition_{i}.mp4")
                slide_path = create_transition_slide(text, duration, transition_path, i)
                
                if slide_path:
                    segment_paths.append(slide_path)
        
        # Combine all segments
        if segment_paths:
            # Create a file list for ffmpeg
            file_list_path = os.path.join(output_dir, "file_list.txt")
            with open(file_list_path, "w") as f:
                for path in segment_paths:
                    f.write(f"file '{os.path.abspath(path)}'\n")
            
            # Combine the segments (using concat demuxer for handling potentially different codecs/framerates)
            output_path = os.path.join(output_dir, "final_trailer.mp4")
            concat_cmd = f'ffmpeg -y -f concat -safe 0 -i "{file_list_path}" -c copy "{output_path}"'
            
            subprocess.call(concat_cmd, shell=True)
            
            # Check if the output file was created
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                st.success("Final trailer created successfully!")
                return output_path
            else:
                st.error("Failed to create final trailer")
                return None
        else:
            st.error("No segments were created")
            return None
    
    except Exception as e:
        st.error(f"Error creating trailer: {str(e)}")
        return None

def parse_claude_plan(plan_text):
    """Parse Claude's trailer plan into a structured format"""
    try:
        # First, try to parse as JSON
        try:
            plan_data = json.loads(plan_text)
            if isinstance(plan_data, dict) and "segments" in plan_data:
                return plan_data["segments"]
            elif isinstance(plan_data, list):
                return plan_data
            else:
                st.warning("JSON format received but missing expected 'segments' array. Using as-is.")
                return [plan_data]
        except json.JSONDecodeError:
            # Not valid JSON, try to extract JSON block
            json_pattern = r"```json\s*([\s\S]*?)\s*```"
            matches = re.findall(json_pattern, plan_text)
            
            if matches:
                try:
                    json_data = json.loads(matches[0])
                    if isinstance(json_data, dict) and "segments" in json_data:
                        return json_data["segments"]
                    elif isinstance(json_data, list):
                        return json_data
                    else:
                        st.warning("Found JSON in code block but missing expected structure. Using as-is.")
                        return [json_data]
                except:
                    pass
            
            # If we still don't have valid JSON, try to parse as structured text
            st.warning("Could not parse response as JSON. Attempting to parse as text.")
            # Text parsing logic here (previous implementation)
            
            # Return a simple transition slide if all parsing fails
            return [{
                "type": "transition",
                "text": "Failed to parse plan. Please check the Claude reasoning tab for details.",
                "duration": 10
            }]
    
    except Exception as e:
        st.error(f"Error parsing Claude's plan: {str(e)}")
        return [{
            "type": "transition",
            "text": "Error parsing plan. Please try again.",
            "duration": 10
        }]





def get_video_duration(video_path):
    """Get the duration of a video file"""
    try:
        cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{video_path}"'
        output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
        return float(output)
    except Exception as e:
        st.error(f"Error getting video duration: {str(e)}")
        return 0.0

def extract_segment_from_transcript(transcript, start_time_str, duration=10):
    """Extract a segment of text from a transcript based on timestamp"""
    try:
        # Parse transcript lines
        lines = transcript.split('\n')
        extracted_text = []
        extracted_timestamps = []
        
        # Parse start time
        if ':' in start_time_str:
            h, m, s = start_time_str.split(':')
            start_seconds = int(h) * 3600 + int(m) * 60 + float(s)
        else:
            start_seconds = float(start_time_str)
        
        end_seconds = start_seconds + duration
        
        # Find lines within the time range
        for line in lines:
            if line.startswith('['):
                # Extract timestamp
                timestamp_end = line.find(']')
                if timestamp_end > 0:
                    timestamp_str = line[1:timestamp_end]
                    if ':' in timestamp_str:
                        h, m, s = timestamp_str.split(':')
                        line_seconds = int(h) * 3600 + int(m) * 60 + float(s)
                        
                        if start_seconds <= line_seconds <= end_seconds:
                            # Extract text after timestamp
                            text = line[timestamp_end + 1:].strip()
                            extracted_text.append(text)
                            extracted_timestamps.append(line_seconds)
        
        # Get the actual start and end time of the extracted text
        actual_start = min(extracted_timestamps) if extracted_timestamps else start_seconds
        actual_end = max(extracted_timestamps) if extracted_timestamps else end_seconds
        
        return {
            "text": ' '.join(extracted_text),
            "actual_start": actual_start,
            "actual_end": actual_end
        }
    except Exception as e:
        st.error(f"Error extracting segment from transcript: {str(e)}")
        return {
            "text": "",
            "actual_start": start_seconds,
            "actual_end": end_seconds
        }

def download_font():
    """Download Coolvetica font if not available"""
    font_path = os.path.join(os.getcwd(), "Coolvetica.otf")
    
    if not os.path.exists(font_path):
        try:
            font_url = "https://www.dafont.com/coolvetica.font"
            st.info(f"Downloading Coolvetica font from {font_url}")
            
            # For demonstration purposes - would need a direct download link
            # This is a placeholder and would need to be replaced with actual font download
            st.warning("Font download not implemented - using system fonts")
            return False
        except Exception as e:
            st.error(f"Error downloading font: {str(e)}")
            return False
    
    return True

def create_empty_video(duration, width, height, output_path):
    """Create an empty video with a solid background"""
    try:
        cmd = (
            f'ffmpeg -y -f lavfi -i color=c=black:s={width}x{height}:d={duration} '
            f'-c:v libx264 -tune stillimage -pix_fmt yuv420p "{output_path}"'
        )
        subprocess.call(cmd, shell=True)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        else:
            return None
    except Exception as e:
        st.error(f"Error creating empty video: {str(e)}")
        return None

def create_trailer_from_plan(videos, plan, output_dir):
    """Create the trailer from the plan"""
    try:
        # Parse the plan if it's a string
        if isinstance(plan, str):
            segments = parse_claude_plan(plan)
        else:
            segments = plan  # Assume it's already parsed
        
        # Create segments
        segment_paths = []
        
        for i, segment in enumerate(segments):
            segment_type = segment.get("type", "video")
            
            if segment_type == "video":
                video_id = segment.get("video_id", 0)
                # Convert video_id to 0-based index if needed
                if isinstance(video_id, int) and video_id > 0:
                    video_id = video_id - 1
                    
                # Ensure video_id is within range
                video_id = min(video_id, len(videos) - 1) if videos else 0
                
                if videos:
                    video_info = videos[video_id]
                    clip_info = {
                        "start_time": segment.get("start_time", "0:00:00"),
                        "end_time": segment.get("end_time", None),
                        "duration": segment.get("duration", 5),
                        "text": segment.get("text", "")
                    }
                    
                    segment_path = create_trailer_segment(video_info, clip_info, output_dir, i)
                    if segment_path:
                        segment_paths.append(segment_path)
                        
            elif segment_type == "transition":
                text = segment.get("text", "")
                duration = segment.get("duration", 3)
                
                transition_path = os.path.join(output_dir, f"transition_{i}.mp4")
                slide_path = create_transition_slide(text, duration, transition_path, i)
                
                if slide_path:
                    segment_paths.append(slide_path)
        
        # Combine all segments
        if segment_paths:
            # Use moviepy to concatenate videos
            try:
                import moviepy.editor as mp
                
                clips = []
                for path in segment_paths:
                    clip = mp.VideoFileClip(path)
                    clips.append(clip)
                
                final_clip = mp.concatenate_videoclips(clips)
                output_path = os.path.join(output_dir, "final_trailer.mp4")
                final_clip.write_videofile(output_path, codec='libx264')
                
                # Close clips to free resources
                for clip in clips:
                    clip.close()
                final_clip.close()
                
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    st.success("Final trailer created successfully!")
                    return output_path
                else:
                    st.error("Failed to create final trailer with moviepy")
                    return None
                
            except Exception as e:
                st.error(f"Error concatenating video clips with moviepy: {str(e)}")
                
                # Fallback: Try generating a text report of the trailer plan
                st.warning("Creating a trailer plan report instead of video")
                report = "# Trailer Plan Report\n\n"
                
                for i, segment in enumerate(segments):
                    segment_type = segment.get("type", "video")
                    if segment_type == "video":
                        video_id = segment.get("video_id", 0)
                        if isinstance(video_id, int) and video_id > 0:
                            video_id = video_id - 1
                        video_id = min(video_id, len(videos) - 1) if videos else 0
                        
                        if videos:
                            video_info = videos[video_id]
                            video_name = video_info.get("name", f"Video {video_id+1}")
                            
                            report += f"## Segment {i+1}: Video Clip\n"
                            report += f"- **Source:** {video_name}\n"
                            report += f"- **Start Time:** {segment.get('start_time', '0:00:00')}\n"
                            report += f"- **Duration:** {segment.get('duration', 5)} seconds\n"
                            report += f"- **Text Overlay:** \"{segment.get('text', '')}\"\n\n"
                            
                            # Include a transcript excerpt if available
                            start_time = segment.get("start_time", "0:00:00")
                            duration = segment.get("duration", 5)
                            transcript_segment = extract_segment_from_transcript(
                                video_info.get("transcript", ""), 
                                start_time, 
                                duration
                            )
                            
                            if transcript_segment.get("text"):
                                report += f"- **Transcript excerpt:**\n  \"{transcript_segment.get('text')}\"\n\n"
                    
                    elif segment_type == "transition":
                        report += f"## Segment {i+1}: Transition Slide\n"
                        report += f"- **Text:** \"{segment.get('text', '')}\"\n"
                        report += f"- **Duration:** {segment.get('duration', 3)} seconds\n\n"
                
                # Save the report
                report_path = os.path.join(output_dir, "trailer_plan_report.md")
                with open(report_path, "w") as f:
                    f.write(report)
                
                st.info("Created a trailer plan report instead of video. You can view the details in the app.")
                st.markdown(report)
                
                return report_path
        else:
            st.error("No segments were created")
            return None
    
    except Exception as e:
        st.error(f"Error creating trailer: {str(e)}")
        return None


# Add a function to standardize videos before concatenation
def standardize_video(input_path, output_path):
    """Standardize video format to ensure compatibility during concatenation"""
    try:
        cmd = (
            f'ffmpeg -y -i "{input_path}" -c:v libx264 -preset medium '
            f'-crf 23 -pix_fmt yuv420p -r 30 -c:a aac -b:a 128k "{output_path}"'
        )
        subprocess.call(cmd, shell=True)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        else:
            st.error(f"Failed to standardize video: {input_path}")
            return input_path
    except Exception as e:
        st.error(f"Error standardizing video: {str(e)}")
        return input_path

# Optional: Add this function to create an animated transition
def create_fade_transition(duration, output_path):
    """Create a fade transition video"""
    try:
        # Create a black video that fades in and out
        cmd = (
            f'ffmpeg -y -f lavfi -i color=black:s=1280x720:d={duration} '
            f'-vf "fade=t=in:st=0:d={duration/2},fade=t=out:st={duration/2}:d={duration/2}" '
            f'-c:v libx264 -preset medium -pix_fmt yuv420p "{output_path}"'
        )
        subprocess.call(cmd, shell=True)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        else:
            return None
    except Exception as e:
        st.error(f"Error creating fade transition: {str(e)}")
        return None
        
def extract_audio_from_video(video_path, output_dir):
    """Extract audio from video file using Python libraries instead of ffmpeg"""
    try:
        import moviepy.editor as mp
        
        filename = os.path.basename(video_path)
        basename = os.path.splitext(filename)[0]
        audio_path = os.path.join(output_dir, f"{basename}.mp3")
        
        # Using moviepy to extract audio
        video = mp.VideoFileClip(video_path)
        audio = video.audio
        
        if audio is not None:
            audio.write_audiofile(audio_path)
            video.close()
            
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                return audio_path
            else:
                st.error(f"Failed to extract audio from {filename}")
                return None
        else:
            st.error(f"No audio stream found in {filename}")
            return None
    except Exception as e:
        st.error(f"Error extracting audio from {filename}: {str(e)}")
        return None

def transcribe_audio(client, file_path):
    """Transcribe audio using OpenAI's transcription service"""
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="srt"
        )
    # Pass the transcription directly for processing
    return process_transcription(transcription)

def process_transcription(transcription):
    """Process the raw transcription into the desired format"""
    blocks = transcription.split('\n\n')
    processed_lines = []
    for block in blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            time_range = lines[1]
            text = lines[2]
            start_time = time_range.split(' --> ')[0]
            # Convert the time format from "00:00:00,000" to "0:00:00"
            formatted_start_time = format_time(start_time)
            processed_line = f"[{formatted_start_time}]{text}"
            processed_lines.append(processed_line)
    return '\n'.join(processed_lines)

def format_time(time_str):
    """Format time from "00:00:00,000" to "0:00:00" """
    parts = time_str.split(',')[0].split(':')
    return f"{int(parts[0])}:{parts[1]}:{parts[2]}"

def save_transcription(transcript, output_dir, basename):
    """Save transcription to a file"""
    transcript_path = os.path.join(output_dir, f"{basename}.srt")
    with open(transcript_path, 'w') as f:
        f.write(transcript)
    return transcript_path

def capture_first_frame(video_path, output_dir):
    """Capture first frame of video as screenshot"""
    filename = os.path.basename(video_path)
    basename = os.path.splitext(filename)[0]
    screenshot_path = os.path.join(output_dir, f"{basename}.png")
    try:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(screenshot_path, frame)
            cap.release()
            return screenshot_path
        else:
            st.error(f"Failed to capture first frame from {filename}")
            cap.release()
            return None
    except Exception as e:
        st.error(f"Error capturing first frame from {filename}: {str(e)}")
        return None

def analyze_video_content(video_path, screenshot_path, transcript):
    """Analyze video content using OpenAI GPT-4o vision via direct HTTP request"""
    try:
        # Encode the screenshot
        with open(screenshot_path, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
            
        prompt = f"""
        Analyze this video based on its first frame and transcript.
        
        Transcript:
        {transcript}
        
        Please provide the following information:
        1. Who is speaking (if anyone)?
        2. What is the main message or topic?
        3. How does this relate to a petition or advocacy campaign?
        4. What demographic information can you observe about any people in the frame?
        5. Is this a news report, personal testimony, or something else?
        
        Format your response as JSON:
        {{
          "speaker_type": "supporter", "news_reporter", "other", or "none",
          "main_message": "Summary of the main point being made",
          "petition_relevance": "How this relates to advocacy",
          "demographic_notes": "Demographics of people shown (if any)",
          "content_type": "testimony", "news", "interview", or other appropriate category
        }}
        """
        
        # Direct API call using requests
        import requests
        
        headers = {
            "Authorization": f"Bearer {st.secrets['openai']['api_key']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1000,
            "response_format": {"type": "json_object"}
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            st.error(f"OpenAI API call failed with status code: {response.status_code}")
            st.error(f"Response: {response.text}")
            return {
                "speaker_type": "unknown",
                "main_message": "Error analyzing content",
                "petition_relevance": "unknown",
                "demographic_notes": "unknown",
                "content_type": "unknown"
            }
        
        response_json = response.json()
        
        if "choices" in response_json and len(response_json["choices"]) > 0:
            content = response_json["choices"][0]["message"]["content"]
            try:
                return json.loads(content)
            except json.JSONDecodeError as json_err:
                st.error(f"Invalid JSON in API response: {str(json_err)}")
        else:
            st.error("Empty or invalid response from OpenAI API")
        
        # Return default object if any validation fails
        return {
            "speaker_type": "unknown",
            "main_message": "Video analysis unavailable",
            "petition_relevance": "unknown",
            "demographic_notes": "unknown",
            "content_type": "unknown"
        }
    
    except Exception as e:
        st.error(f"Error analyzing video content: {str(e)}")
        return {
            "speaker_type": "unknown",
            "main_message": "Error analyzing content",
            "petition_relevance": "unknown",
            "demographic_notes": "unknown",
            "content_type": "unknown"
        }


# Set page configuration
st.set_page_config(
    page_title="Trailer: Supporter Videos",
    page_icon="🎬",
    layout="wide"
)

# Initialize session state variables
if "processed_videos" not in st.session_state:
    st.session_state.processed_videos = []
if "scene_timestamps" not in st.session_state:
    st.session_state.scene_timestamps = {}
if "trailer_plan" not in st.session_state:
    st.session_state.trailer_plan = None
if "reasoning" not in st.session_state:
    st.session_state.reasoning = None

# Setup tabs
tab1, tab2, tab3 = st.tabs(["Upload Videos", "Generate Trailer", "Stylize (Coming Soon)"])

# API key management - using Streamlit secrets instead of UI inputs
with tab1:
    st.title("Upload Videos for Petition Trailer")
    
    # Add petition text input
    st.subheader("Petition Details")
    petition_text = st.text_area(
        "Paste the petition text here",
        height=200,
    )
    
    # Store petition text in session state
    if petition_text:
        st.session_state.petition_text = petition_text
        
    try:
    # Initialize OpenAI client without proxies argument
        openai_client = OpenAI(api_key=st.secrets["openai"]["api_key"])
        st.success("OpenAI client initialized")
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {str(e)}")
        openai_client = None
    
    try:
        # Initialize Anthropic client without proxies argument
        anthropic_client = Anthropic(api_key=st.secrets["anthropic"]["api_key"])
        st.success("Anthropic client initialized")
    except Exception as e:
        st.error(f"Error initializing Anthropic client: {str(e)}")
        anthropic_client = None

    # Create directories
    temp_dir = os.path.join(os.getcwd(), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    audio_dir = os.path.join(temp_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    
    transcript_dir = os.path.join(temp_dir, "transcripts")
    os.makedirs(transcript_dir, exist_ok=True)
    
    screenshot_dir = os.path.join(temp_dir, "screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)


    # Video upload functionality
    uploaded_files = st.file_uploader("Upload Video Files", type=["mp4", "mov", "avi"], accept_multiple_files=True)
    
    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} videos")
        
        # Display thumbnails of uploaded videos in a grid
        cols = st.columns(min(3, len(uploaded_files)))
        for i, uploaded_file in enumerate(uploaded_files):
            # Save uploaded file to disk
            video_path = os.path.join(temp_dir, uploaded_file.name)
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Display video thumbnail in grid
            with cols[i % 3]:
                st.video(video_path, start_time=0)
                st.caption(uploaded_file.name)
        
        # Process all videos button
        if st.button("Process All Videos"):
            if not openai_client:
                st.error("Please check your OpenAI API key in the Streamlit secrets")
            else:
                # Clear any existing processed videos
                st.session_state.processed_videos = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    video_path = os.path.join(temp_dir, uploaded_file.name)
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        # Step 1: Extract audio
                        audio_path = extract_audio_from_video(video_path, audio_dir)
                        
                        if not audio_path:
                            st.error(f"Failed to extract audio from {uploaded_file.name}")
                            continue
                        
                        # Step 2: Transcribe audio
                        transcript = transcribe_audio(openai_client, audio_path)
                        
                        if not transcript:
                            st.error(f"Failed to transcribe audio from {uploaded_file.name}")
                            continue
                        
                        # Save transcription
                        basename = os.path.splitext(uploaded_file.name)[0]
                        transcript_path = save_transcription(transcript, transcript_dir, basename)
                        
                        # Step 3: Capture first frame
                        screenshot_path = capture_first_frame(video_path, screenshot_dir)
                        
                        if not screenshot_path:
                            st.error(f"Failed to capture screenshot from {uploaded_file.name}")
                            continue
                        
                        # Step 4: Analyze video content
                        video_analysis = analyze_video_content(video_path, screenshot_path, transcript)
                        
                        # Add processed video to session state
                        st.session_state.processed_videos.append({
                            "name": uploaded_file.name,
                            "path": video_path,
                            "transcript": transcript,
                            "screenshot": screenshot_path,
                            "analysis": video_analysis
                        })
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                progress_bar.progress(100)
                status_text.text(f"Processed {len(uploaded_files)} videos successfully!")
                
                # Show summary of processed videos
                st.success(f"Processed {len(st.session_state.processed_videos)} videos. Switch to the 'Generate Trailer' tab to create your trailer.")
    
    # Trailer generation tab
    with tab2:
        st.title("Generate Petition Trailer")
        
        if not st.session_state.processed_videos:
            st.warning("Please upload and process videos in the Upload tab first")
        else:
            # Display processed videos summary
            st.subheader(f"Processed Videos: {len(st.session_state.processed_videos)}")
            
            # Create an expandable section for each video
            for i, video in enumerate(st.session_state.processed_videos):
                with st.expander(f"Video {i+1}: {video['name']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(video["screenshot"], caption=f"Screenshot from {video['name']}")
                    with col2:
                        st.json(video["analysis"])
                    st.text_area("Transcript", video["transcript"], height=150)
        
        # Generate trailer button
        if st.button("Generate Trailer Plan"):
            if "anthropic" not in st.secrets or "api_key" not in st.secrets["anthropic"]:
                st.error("Please add your Anthropic API key to the Streamlit secrets")
            else:
                with st.spinner("Generating trailer plan with Claude..."):
                    # Prepare data for Claude
                    videos_data = []
                    for i, video in enumerate(st.session_state.processed_videos):
                        videos_data.append({
                            "id": i+1,
                            "name": video["name"],
                            "transcript": video["transcript"],
                            "analysis": video["analysis"]
                        })
                    
                    # Call Claude to generate a trailer plan
                    trailer_plan, reasoning = generate_trailer_plan(videos_data)
                    
                    # Store in session state
                    st.session_state.trailer_plan = trailer_plan
                    st.session_state.reasoning = reasoning
        
        # Display the trailer plan if available
        if st.session_state.trailer_plan:
            st.subheader("Claude's Reasoning")
            st.write(st.session_state.reasoning)
            
            st.subheader("Trailer Plan")
            st.write(st.session_state.trailer_plan)
            
            # Execute trailer generation
            if st.button("Create Trailer"):
                with st.spinner("Creating trailer..."):
                    result = create_trailer_from_plan(
                        st.session_state.processed_videos,
                        st.session_state.trailer_plan,
                        output_dir
                    )
                    
                    if result:
                        st.success("Trailer created successfully!")
                        st.video(result)
                    else:
                        st.error("Failed to create trailer")

# Stylization tab (placeholder for future development)
with tab3:
    st.title("Stylize Your Trailer (Coming Soon)")
    st.info("This feature will allow you to stylize your trailer with Canva templates and background music.")
    st.warning("This feature is under development and will be available soon.")


# Main app execution starts here
if __name__ == "__main__":
    # Download font at startup (optional)
    download_font()
