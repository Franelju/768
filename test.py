#%%  Imports
import imageio as iio
import numpy as np

#%%
# read the video (it fits into memory)
# Note: this will open the image twice. Check the docs (advanced usage) if
# this is an issue for your use-case
metadata = iio.immeta("imageio:cockatoo.mp4", exclude_applied=False)
frames = iio.imread("imageio:cockatoo.mp4", index=None)

# manually convert the video
gray_frames = np.dot(frames, [0.2989, 0.5870, 0.1140])
gray_frames = np.round(gray_frames).astype(np.uint8)
gray_frames_as_rgb = np.stack([gray_frames] * 3, axis=-1)

# write the video
iio.imwrite("cockatoo_gray.mp4", gray_frames_as_rgb, fps=metadata["fps"])
# %%
