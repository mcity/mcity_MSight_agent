# upstream/

Code that belongs in MSight_Vision itself rather than in the agent. It lives here
until the merge, and is launched from this path in the meantime.

- `msight_vision_nodes/annotated_frame_publisher.py`: an `msight_core` node that
  draws detection boxes and republishes frames for `image_to_video_aggregator`.
  Used by the record/archive chain.
