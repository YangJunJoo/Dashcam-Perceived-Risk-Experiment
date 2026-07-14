# %%
from datasets import load_dataset
import pandas as pd
import os
import numpy as np
import moviepy.config

# %%
# get current working directory
import os
os.getcwd()

# %%
import pandas as pd

try:
    metadata = pd.read_csv(r'metadata_ideal.csv', encoding='latin-1')
    print("Successfully loaded with latin-1 encoding.")
except UnicodeDecodeError:
    print("latin-1 encoding failed. Trying cp1252...")
    try:
        metadata = pd.read_csv(r'metadata_ideal.csv', encoding='cp1252')
        print("Successfully loaded with cp1252 encoding.")
    except UnicodeDecodeError:
        print("cp1252 encoding failed. Trying utf-16...")
        try:
            metadata = pd.read_csv(r'metadata_ideal.csv', encoding='utf-16')
            print("Successfully loaded with utf-16 encoding.")
        except UnicodeDecodeError:
            print("All common encodings failed. You might need to determine the correct encoding.")
            # If none of the above work, you might need to use a tool to detect the encoding
            # or inspect the file's properties.

# %%
metadata

# %% [markdown]
# Trimming videos

# %%
import os
import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip, CompositeVideoClip, VideoClip
from PIL import Image, ImageDraw, ImageFont
import math

# === USER SETTINGS ===
base_dir      = r"D:\01_Data\07_Dashcam\Nexar"         # adjust to your root folder
trimmed_dir   = os.path.join(base_dir, "survey_pilot_v2")
os.makedirs(trimmed_dir, exist_ok=True)

font_path     = r"C:\Windows\Fonts\arial.ttf"          # must point to a .ttf on your system
font_size     = 50

bar_font_size = 15                          # e.g. half your timer font
bar_font      = ImageFont.truetype(font_path, bar_font_size)

offsets       = [0.5, 1.0, 1.5, 2.0, 2.5]    # seconds before event to end trim
trim_duration = 5.0                  # length of each clip in seconds
# ======================

# (1) Your metadata_sample DataFrame must already be defined,
#     with columns 'video_path' and 'time_of_event'.
# metadata_sample = pd.read_csv("your_metadata.csv")

metadata_trimmed_list = []

for idx, row in metadata.iloc[:].iterrows():
    rel_path  = row['video_path']
    full_path = os.path.join(base_dir, rel_path)

    if not os.path.isfile(full_path):
        print(f"⚠️ Missing file: {full_path}")
        continue

    event_time = row['time_of_event']

    try:
        with VideoFileClip(full_path) as clip:
            video_dur = clip.duration
            w, h      = clip.size
            font      = ImageFont.truetype(font_path, font_size)

            for offset in offsets:
                end_ref    = event_time - offset
                start_time = max(0, end_ref - trim_duration)
                end_time   = min(video_dur, start_time + trim_duration)
                actual_dur = end_time - start_time

                if actual_dur <= 0:
                    print(f"➖ Skipping {rel_path} @ offset {offset}s (dur<=0)")
                    continue

                # --- build the RGBA generator ---
                def make_rgba(t):
                    remaining = max(0, actual_dur - t)
                    remaining_half = math.ceil(remaining * 10) / 10


                    # transparent canvas
                    img  = Image.new("RGBA", (w, h), (0, 0, 0, 0))
                    draw = ImageDraw.Draw(img)

                    # # bar dimensions & spacing
                    # bar_width   = 4      # px
                    bar_height  = 4     # px
                    # bar_spacing = 1     # px

                    # total_bars   = w // (bar_width + bar_spacing)
                    # bars_to_draw = int((remaining / actual_dur) * total_bars)

                    # # draw bars from right edge inward
                    y0 = h - bar_height - 1
                    # for i in range(bars_to_draw):
                    #     # i=0   → rightmost bar
                    #     # i=1   → one bar to its left, etc.
                    #     x0 = w - (i+1) * (bar_width + bar_spacing)
                    #     draw.rectangle(
                    #         [x0, y0, x0+bar_width, y0+bar_height],
                    #         fill=(255, 255, 255, 255)
                    #     )

                    # draw remaining-time text just above the bar
                    text  = f"{remaining_half:.1f} sec"
                    tbbox = draw.textbbox((0, 0), text, font=font)
                    tw, th = tbbox[2] - tbbox[0], tbbox[3] - tbbox[1]
                    x_txt   = w - tw - 10
                    y_txt   = y0 - th - 120
                    draw.text((x_txt, y_txt), text, font=font, fill=(255, 255, 255, 255))

                    return np.array(img)

                # RGB frames for color
                def make_rgb(t):
                    return make_rgba(t)[:, :, :3]

                # mask frames from alpha channel (0–1 floats)
                def make_mask(t):
                    return make_rgba(t)[:, :, 3].astype(float) / 255.0

                # Create the PIL-based text clip + mask
                txt_clip = VideoClip(make_rgb,  duration=actual_dur)
                mask_clip= VideoClip(make_mask, ismask=True, duration=actual_dur)
                txt_clip = txt_clip.set_mask(mask_clip)

                # Trim the source clip
                trimmed = clip.subclip(start_time, end_time)

                # Composite and write
                out_name = (
                    f"{os.path.splitext(os.path.basename(rel_path))[0]}"
                    f"_off{int(offset*100)}_new.mp4"
                )
                out_path = os.path.join(trimmed_dir, out_name)

                print(f"✂️ Trimming {rel_path}: {start_time:.2f}s → {end_time:.2f}s (off={offset}s)")

                # Skip if file already exists
                if os.path.exists(out_path):
                    # print(f"⚠️ File already exists: {out_path}")
                    continue

                final = CompositeVideoClip([trimmed, txt_clip])
                final.write_videofile(
                    out_path,
                    codec="libx264",
                    audio_codec="aac" if final.audio else None,
                    fps=clip.fps,
                    logger="bar"
                )

                # Cleanup
                trimmed.close()
                txt_clip.close()
                mask_clip.close()
                final.close()

                # Record metadata
                rec = row.to_dict()
                rec.update({
                    'trimmed_video_path': out_path,
                    'trim_start_time':    start_time,
                    'trim_end_time':      end_time,
                    'trim_offset':        offset,
                    'trim_duration':      actual_dur
                })
                metadata_trimmed_list.append(rec)

    except Exception as e:
        print(f"❌ Error processing {rel_path}: {e}")

# Build summary DataFrame
if metadata_trimmed_list:
    metadata_trimmed = pd.DataFrame(metadata_trimmed_list)
    print("\n✅ Trimming complete. Sample metadata:")
    print(metadata_trimmed.head())
else:
    print("\n⚠️ No clips were successfully trimmed.")

# %%
# Rename files in folder_path
folder_path = r"D:\01_Data\07_Dashcam\Nexar\survey_pilot_v3"

# Replace the file name with the new file name
for file in os.listdir(folder_path):
    # Replace the file_name_new with the new_file_name
    new_file_name = file.replace("_new", "")
    os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_file_name))
    os.rename(os.path.join(folder_path, new_file_name), os.path.join(folder_path, 'new_'+ new_file_name))

    

