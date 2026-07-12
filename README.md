# Video pipelines

Recommended pipeline for real life footage:

- script:
  - upscale - cheaper than after interpolation, but may interfere with 3d camera reconstruction, if so do after, but before depth
  - interpolate to 240fps - may interfere with 3d camera reconstruction, if so do after, but before depth
  - extract 3d camera (effects compatible)
  - get depth, doesnt matter if its before or after 3d camera.
- blender:
  - make and animate 3d characters and environments, export as models
- after effects:
  - import 3d models, anchor using 3d camera
  - composite character and footage, block out character with depth map
  - velocity
  - import 3d camera.
  - apply effects on 3d character / matte layer / 3d camera movement

Unused:

- restore / sharpen - maybe redundant with upscaling,
- stabilize - optional if using gimball

Alternatives:

- import 3d camera to blender and composite with character
  - cons:
    - character velocity needs to match footage, else movement will drift
    - slower to iterate (e.g., modify character position)
  - pros:
    - may have more control over lightning etc
    - may be more performant vs working on a exported video layer.
- 3d camera track can be swapper with after effects built in tracker.

## Current findings

- upscale to 4k
  - seedvr2 unstable, worked some time but crash usually
  - esrgan stable, a bit slow.
- depth
  - depth anything v2/3:
    - crash 4k
  - depth crafter:
    - crash 4k
    - 150-400s/it in 1080p. way too slow.
    - works well with new settings, a couple minutes?
  - the anime scripter: poor quality
  - depth anything v3 streaming: cuda device not ready.
- normal
