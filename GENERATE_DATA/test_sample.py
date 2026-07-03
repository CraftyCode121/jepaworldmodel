import numpy as np
import imageio.v2 as imageio
from pathlib import Path 

Path("MP4_Vids").mkdir(exist_ok=True)

data = np.load("synthetic_dataset/elastic_wall_bounce/elastic_wall_bounce_000000.npz")

frames = data["frames"]      

with imageio.get_writer(
    "MP4_Vids/output.mp4",
    fps=15,
    codec="libx264",
    quality=8
) as writer:
    for frame in frames:
        writer.append_data(frame)

print("Done!")