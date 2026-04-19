# Video pipelines

recommended pipeline for real life footage:

what is better to in after effects vs blender:
blender:
- import and animate models
- build environments

after effects:
- fog / depth effects

ideal:
1. stabilize
2. upscale
3. interpolate to 240fps
4. reconstruct 3d camera.
5.


realistic:

decisions
- fine to have same velocity on clip as animation (same as sfm)
- i dont need to stabilize if i use a gyro probably.


questions:
- is stabilization deterministic layer / operation that can be applied independently on dfferent videos? of so, we can use it separately on footage and 3d model to avoid having to pre bake everything and losing depth maps etc.


pipeline

1. record in 4k 30 fps without stabilization.
2. reconstruct 3d camera.
3. add 3d models.
4. render video.
5. stabilize.
6. interpolate



nier example:

funnel:
- velocity
- upscale
- interpolate
- restore / sharpen
- apply effects on 3d character matte layer,
- add environmentral effects, anchor using 3d camera, use depth map for occlusion and effects.

- script: extract 3d camera
- blender: make and animate 3d character, export only character with nothing else
- script: get depth map from footage
- after effecs: composite character and footage, block out character with depth map
