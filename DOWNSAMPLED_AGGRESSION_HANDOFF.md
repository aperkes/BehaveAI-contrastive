# Downsampled aggression-classification handoff

## BehaveAI tracking changes

`simple_track.py` now writes the following metadata into every exported crop
manifest:

- `source_fps`
- `source_width`
- `source_height`
- `frame_step` (`frame_skip + 1`)
- `time_seconds`, calculated as `(frame - 1) / source_fps`

`frame` remains the one-based frame number in the original source video. For
example, `frame_skip = 3` produces `frame_step = 4` and normally processes
frames 1, 5, 9, and so on.

The manifest also contains `id`, `x`, `y`, `original_id`, bounding boxes, and
oriented bounding boxes. The contrastive pipeline preserves the added metadata
columns in `identity_predictions.csv`. In the final matched predictions:

- `id` and `predicted_identity` are the corrected global numeric identity.
- `original_id` is the original BehaveAI motion-track ID.
- `identity_mapping.csv` maps the global numeric identities to biological fish
  IDs such as `B14.1.0`.

This means an aggression classifier can operate from the matched prediction CSV
without opening the source video. FPS, elapsed time, and frame dimensions are
all available in the CSV.

The metadata change was tested on a 10-second, 30-fps, 1080x1080 video. Frame 1
was recorded at 0 seconds and frame 300 at 9.96666667 seconds. The resulting
manifest was also successfully loaded by `contrastive_identity.py`.

Existing manifests do not acquire the new columns automatically. They must be
regenerated with a new output directory or `--force-recompute`, or updated with
a one-time metadata backfill while the source video is available.

## Existing contrastive support for frame skipping

The contrastive code already reads `frame_step` and uses it when determining
whether detections belong to a contiguous track fragment. Regularly sampled
frames such as 1, 5, 9 are therefore treated as contiguous when
`frame_step = 4`. Unexpected larger gaps are still treated as missing tracking.

## Why the current aggression inference is incompatible

The current code in
`/data/tracking/raine.fights/behaviorMapper/infer_aggression.py` assumes dense
30-fps trajectories:

- `sparse_windows()` rejects a window if any original frame number is absent.
  A regularly downsampled trajectory therefore produces almost no windows.
- Velocity is calculated with operations such as `np.gradient(xy) * fps`, which
  assumes adjacent observations are exactly one source frame apart.
- One-second windows use `round(fps)` rows rather than elapsed time.

Simply treating downsampled rows as consecutive frames would inflate or distort
velocity and make the temporal windows too long.

## Proposed aggression-code changes

1. Read `source_fps`, `frame_step`, `time_seconds`, `source_width`, and
   `source_height` from `identity_predictions.csv`. Do not require the source
   video for metadata.
2. Calculate motion using actual elapsed time. For two observations, use
   `dt = (frame_2 - frame_1) / source_fps` (or the difference in
   `time_seconds`) and divide displacement by `dt`.
3. Define classification windows and strides in seconds, not numbers of CSV
   rows. A one-second window at `frame_step = 4` contains approximately eight
   observations at 30 fps, not 30 observations.
4. Treat a gap equal to `frame_step` as normal continuity. Treat larger gaps as
   missed tracking. Interpolate only short missing gaps if the training
   procedure uses the same policy.
5. Train and infer using the same sampling interval, or explicitly resample all
   trajectories onto a declared time grid before feature calculation.
6. Preserve original source-frame numbers in aggression outputs so predictions
   remain aligned with video overlays and other BehaveAI results.
7. Ignore identity `-1` and tolerate changing numbers of visible/present fish.
8. Use `source_width` and `source_height` for side/quadrant logic and spatial
   normalization instead of querying the video.

For testing different `frame_skip` values, use distinct pipeline output
directories or pass `--force-recompute`; the identity pipeline's resume logic
otherwise intentionally reuses completed tracking manifests.
