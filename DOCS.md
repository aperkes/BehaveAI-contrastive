# BehaveAI fork: practical guide

This document covers the functionality added by this fork. The original BehaveAI
README and wiki remain the primary reference for installation, annotation, model
training, and the rest of the GUI.

Run these commands from the repository directory and, for GPU work, from the
`behaveAI` environment:

```bash
conda activate behaveAI
cd /path/to/BehaveAI
```

Every command supports `-h` or `--help`. Paths below are examples and should be
replaced with paths for the current project.

## Changes visible in the BehaveAI GUI

### Motion history and temporal gap

The Motion settings tab has two independent controls:

- **Motion history frames** selects one, two, or three previous frames. Three
  preserves the traditional multicolored motion history. One produces a signed
  difference image: pixels that become brighter are red and pixels that become
  darker are blue.
- **Motion frame gap** is the distance between frames used to construct the
  motion image. Increasing it captures motion over a longer interval without
  dropping detections from the tracking output.

`Frame skip` still controls which frames are processed and written by tracking.
It is not the same as the motion frame gap.

Motion construction is shared by annotation, dataset inspection, annotation
regeneration, batch tracking, and live inference through `motion_processor.py`.
This avoids those programs generating subtly different motion images.

### Sliding-median background subtraction

Select **sliding_median** in the Motion strategy menu to compare each frame with
the median of a recent frame window.

- **Sliding median window** controls the history length and defaults to 50.
- **Signed difference (positive red)** displays positive changes in red and
  negative changes in blue. With it disabled, the absolute difference is shown
  in grayscale.
- Settings that do not apply to sliding median, such as exponential decay, are
  disabled automatically.

Moving forward by one frame in the annotation GUI retains the median history in
RAM and adds only the new frame. Seeking elsewhere rebuilds the window and shows
progress in the launcher console.

Changing motion settings changes the images presented to a motion model. Existing
motion annotations may therefore need to be regenerated and motion models may
need to be retrained.

### Oriented detections

Head/tail disambiguation behaves as in BehaveAI but is substantially faster. A
single tight oriented crop is extracted and rotated into four hypotheses, which
are classified as one batch. Previously, the complete video frame was warped four
times per detection.

### Projects with a disabled model stream

Batch tracking no longer requires a motion annotation YAML when the primary
motion model is disabled, or a static YAML when the primary static model is
disabled.

## Common command-line workflows

### Track one video and create a crop manifest

```bash
python simple_track.py \
  --project-dir projects/simple_fish \
  --in_file /data/video.mp4 \
  --out_folder projects/simple_fish/output/video_tracking \
  --export-crop-manifest
```

This writes the usual tracking CSV plus a compact crop manifest. The manifest
stores source frame numbers, timing, track IDs, and axis-aligned or oriented
bounding boxes. It does not duplicate every crop as an image; downstream tools
reconstruct only the crops they need from the source video.

Add `--make_vid` only when a BehaveAI tracking-overlay video is wanted.

### Run the full identity pipeline for one video

```bash
python identity_pipeline.py \
  --project-dir projects/simple_fish \
  --in_file /data/video.mp4 \
  --num-individuals 8 \
  --input-color rgb \
  --device cuda
```

This runs `simple_track.py`, `contrastive_identity.py`, and
`identity_overlay.py`. It creates an identity checkpoint, full prediction CSV,
and an overlay video beneath the project output directory.

### Run the stages separately

```bash
python contrastive_identity.py \
  --mode train-predict \
  --manifest output/video_crop_manifest.csv \
  --output-dir output/video_contrastive \
  --num-individuals 8 \
  --input-color grayscale \
  --device cuda

python identity_overlay.py \
  --video /data/video.mp4 \
  --predictions output/video_contrastive/identity_predictions.csv \
  --output output/video_identity_overlay.mp4 \
  -ss 300 -t 600
```

### Match identities across two or three videos

Use the wrapper when the videos have not yet been tracked:

```bash
python match_video_identity_pipeline.py \
  --project-dir projects/fightClub \
  --videos /data/day1.mp4 /data/day2.mp4 /data/day3.mp4 \
  --assignments SurpriseWinnerEffectVideoAssignments.csv \
  --output-dir projects/fightClub/output/three_day_match \
  --device cuda \
  --visualize
```

The wrapper resumes completed tracking, training, and embedding stages. Use
`--force-recompute` to ignore reusable results. The assignment CSV must describe
the known fish and permitted locations for each video interval. Videos containing
a fish replacement must be split at the replacement time before matching.

If crop manifests already exist, call `match_video_identities.py` directly.

### Long videos

Contrastive training and identity embedding use representative crops by default:

- `--max-training-crops 250000` caps the shared training pool.
- `--max-embedding-crops 250000` caps the crops embedded per video for identity
  assignment.
- Set either value to `0` to reproduce uncapped behavior.

Sampling is balanced across conservative fragments. The final
`identity_predictions.csv` still contains every detection; identities inferred
from sampled crops are propagated across their fragments. A capped `latents.csv`
contains only the embedded sample, not every detection. Changing a cap causes the
matching pipeline to rerun the affected identity stage but does not require
tracking the videos again.

## Script reference

### `simple_track.py`

Command-line BehaveAI inference and crop-manifest export. It uses the same model,
motion processor, oriented-box logic, and `kalman_tracker.py` as GUI inference.

Required:

- `--in_file`, `-i`: input MP4.

Project and model selection:

- `--project-dir`: project containing `BehaveAI_settings.ini` and the default
  `model_primary_static/train/weights/best.pt`.
- `--config`: explicit INI; overrides the config inferred from `--project-dir`.
- `--model`: explicit primary model; overrides the model inferred from
  `--project-dir`.

Output:

- `--out_folder`, `-o`: output directory; defaults to `output`.
- `--export-crop-manifest`: write the downstream crop manifest.
- `--make_vid`, `-v`: also encode the BehaveAI overlay video; off by default.

### `contrastive_identity.py`

Trains an appearance embedding from BehaveAI tracks and assigns a known number
of identities. The default is an idtracker.ai-style grayscale ResNet-18 trained
on conservative contiguous fragments.

Core inputs and modes:

- `--manifest PATH`: required; repeat it to train on multiple manifests.
- `--video PATH`: override the source path stored in a manifest; give once per
  manifest.
- `--frame-step N`: continuity interval for an old manifest lacking
  `frame_step`; give once for all manifests or once per manifest.
- `--mode`: `train-predict` (default), `train`, `predict`, `reassign`, `embed`,
  `sample-crops`, or `sample-pairs`.
- `--output-dir`: output directory; default `./contrastive_output`.
- `--num-individuals`: required by `train-predict`, `predict`, and `reassign`.
- `--checkpoint`: trained model required by `predict`, `reassign`, and `embed`.
- `--init-checkpoint`: initialize a new training run from an existing model.
- `--reuse-latents`: existing `latents.csv` required by `reassign`; avoids crop
  decoding and neural-network inference.

Outputs and inspection:

- `--export-latents CSV`: write sampled detection embeddings and track means.
- `--latent-projection {umap,pca,none}`: projection included in latent output;
  default `umap`.
- `--visualization-dimensions N`: projected dimensions; default 2.
- `--projection-fit-samples N`: maximum points used to fit a projection;
  default 50,000.
- `--umap-neighbors N`, `--umap-min-dist X`: UMAP controls; defaults 15 and 0.1.
- `--export-crop-samples [N]`: export representative input crops; default 100
  when the flag has no number.
- `--export-pair-samples [N]`: export N positive and N negative pairs; default
  300 of each when the flag has no number.

Training and architecture:

- `--architecture {idtracker-resnet18,small-cnn}`: default ResNet-18; the
  small CNN is retained for old checkpoints and comparisons.
- `--input-color {grayscale,rgb}`: default grayscale.
- `--input-size N`, `--embedding-dim N`, `--projection-dim N`: network sizes.
  Defaults depend on the architecture; the ResNet normally uses 96-pixel crops
  and an 8-dimensional embedding.
- `--epochs N`: maximum epochs; default 30. Early stopping can finish sooner.
- `--learning-rate X`: default 0.001.
- `--margin X`: defaults to 10 for ResNet training and 0.5 for the legacy model.
- `--pairs-per-batch N`: negative pairs per optimizer step; the same number of
  positive pairs is added. Default 400.
- `--forward-pair-chunk-size N`: pairs evaluated at once inside an optimizer
  batch; default 128 and useful for controlling VRAM.
- `--validation-interval N`: optimizer steps between silhouette checks; automatic
  when omitted.
- `--validation-warmup N`: default 5.
- `--silhouette-target X`: early-success target; default 0.91.
- `--validation-patience N`: checks without improvement before stopping;
  default 30.
- `--validation-images-per-individual N`: default 1,000.
- `--silhouette-samples N`: default 2,000.
- `--seed N`: default 42.
- `--device {auto,cpu,cuda,cuda:0,mps}`: default `auto`.

Conservative fragments and assignment:

- `--training-fragments {conservative,tracks}`: default `conservative`; whole
  tracks are available for comparison.
- `--min-fragment-length N`: default 4 detections.
- `--fragment-close-distance X`: close-encounter threshold in mean box lengths;
  default 0.75.
- `--fragment-assignment-margin X`: direct-versus-swapped ambiguity threshold;
  default 0.35.
- `--fragment-motion-jump X`: motion prediction-error threshold in box lengths;
  default 1.5.
- `--fragment-uncertainty-buffer N`: observations removed on each side of an
  ambiguous event; default 5.
- `--fragment-interpolation-gap N`: gaps interpolated only for uncertainty
  detection; default 10.
- `--identity-assignment {idtracker,kmeans}`: default `idtracker`, which assigns
  fragments with co-occurrence constraints; `kmeans` is an unconstrained
  comparison.
- `--global-fragment-min-frames N`: minimum length for every member of an
  anchoring coexisting group; default 30.

Crop storage and scale:

- `--max-training-crops N`: default 250,000; `0` disables the cap.
- `--max-embedding-crops N`: default 250,000; `0` disables the cap.
- `--crop-storage {auto,ram,disk,on-demand}`: default `auto`, choosing RAM when
  practical and a disk-backed array otherwise.
- `--preload-crops`: compatibility alias for `--crop-storage ram`.
- `--cache-crops N`: on-demand RAM cache size; default 2,048, and `0` disables it.
- `--inference-batch-size N`: default 128.
- `--batch-size`, `--tracks-per-batch`, `--samples-per-track`, and
  `--batches-per-epoch`: legacy small-CNN batching controls. `--batch-size`
  defaults to 128 and `--samples-per-track` to 4.

Optional foreground masking:

- `--background-subtraction [median|none]`: specifying the flag without a value
  selects a video-wide median background.
- `--background-samples N`: frames sampled across the video for the median;
  default 1,000.
- `--background-threshold N`: minimum per-channel difference retained;
  default 25.
- `--background-morph-kernel N`: mask cleanup kernel; default 3, while 0 or 1
  disables cleanup.

Important outputs include `contrastive_identity.pt`,
`identity_predictions.csv`, `identity_assignment_quality.json`, and, when
requested, `latents.csv` and `latents_tracks.csv`. In the prediction CSV, `id`
is the contrastive identity and `original_id` is the BehaveAI tracker ID.

### `identity_pipeline.py`

Three-stage wrapper for one video: tracking, contrastive training/assignment,
and overlay creation.

Required:

- `--project-dir`, `--in_file` (also `--in-file`), and `--num-individuals`.

Optional:

- `--output-dir`: defaults to
  `PROJECT/output/VIDEO_identity_pipeline`.
- `--input-color {grayscale,rgb}` and `--device DEVICE`.
- `--init-checkpoint PATH`: initialize training from another model.
- `--background-subtraction`: enable median foreground masking.
- `-ss`, `--overlay-start`: overlay start in seconds; default 0.
- `-t`, `--overlay-duration`: overlay duration; omitted means to the end.
- `--show-track-ids`: display source track IDs as well as identities.

### `identity_overlay.py`

Renders boxes and assigned identities on a video.

- `--video`, `--predictions`, and `--output` are required.
- `-ss` is the start time in seconds; default 0.
- `-t` is the duration; omitted means to the end.
- `--codec` is the four-character OpenCV codec; default `mp4v`.
- `--show-track-ids` adds the original BehaveAI track ID.

### `match_video_identities.py`

Trains one shared encoder and resolves identities across two or three existing
crop manifests using appearance plus known location constraints.

Required:

- `--manifests M1 M2 [M3]`.
- `--assignments CSV`.
- `--output-dir DIR`.

Optional:

- `--input-color {grayscale,rgb}`, `--device`, `--init-checkpoint`, `--epochs`,
  and `--background-subtraction` control shared training.
- `--min-location-fraction X`: fraction of a local cluster that must be
  compatible with a fish's allowed locations; default 0.5.
- `--max-training-crops N`, `--max-embedding-crops N`: defaults 250,000; zero
  disables a cap.
- `--center X Y`: supply once for all videos or repeat once per manifest. When
  omitted, the center is detected separately for every video.
- `--exclude-edges`: discard detections outside the centered inclusion region.
- `--inclusion-size WIDTH HEIGHT`: custom region and implies `--exclude-edges`;
  the default excluded-edge region is 880 by 1,030 pixels.
- `--force-recompute`: ignore reusable checkpoints and embeddings.

### `match_video_identity_pipeline.py`

Wrapper that runs `simple_track.py` for two or three MP4s and then calls
`match_video_identities.py`.

Required:

- `--project-dir`, `--videos V1 V2 [V3]`, and `--assignments CSV`.

Optional arguments mirror `match_video_identities.py`: `--output-dir`,
`--input-color`, `--device`, `--init-checkpoint`, `--epochs`,
`--background-subtraction`, `--min-location-fraction`, both crop caps,
`--center`, `--exclude-edges`, `--inclusion-size`, and `--force-recompute`.
`--visualize` additionally renders the first ten minutes of every matched video.

### `transfer_identity.py`

Applies a source video's identity model to one target video. It first attempts
direct inference. If assignment coverage, confidence, or collision quality is
poor, it initializes from the source checkpoint and fine-tunes on the target.
Numeric identity labels are not guaranteed to retain the same semantic order;
use external constraints or the secondary classifier when exact names matter.

Required:

- `--project-dir`, `--in_file` (also `--in-file`), `--source-video`, and
  `--num-individuals`.

Optional:

- `--checkpoint`, `--model`: explicit source checkpoint; overrides discovery.
- `--target-manifest`: reuse target tracking.
- `--source-manifest`: reuse source tracking if a source model must be trained.
- `--output-dir`, `--source-output-dir`: override standard pipeline directories.
- `--input-color`, `--device`, `--epochs`, and `--background-subtraction`.
- `--min-assigned-fraction`: direct-inference coverage required; default 0.90.
- `--min-confidence`: mean confidence required; default 0.80.
- `--max-collision-fraction`: maximum collision-frame fraction; default 0.01.

### `classify_identity_samples.py`

Samples crops from already assigned contrastive identities, runs a BehaveAI
secondary-static classifier, and adds `secondary_class` to a new CSV. It does
not overwrite `track_id` or the contrastive identity.

```bash
python classify_identity_samples.py \
  --video /data/video.mp4 \
  --predictions output/identity_predictions.csv \
  --model projects/simple_fish/model_secondary_static \
  --samples-per-identity 100
```

- `--video`, `--predictions`, and `--model` are required. The model can be a
  `best.pt` file or a model directory.
- `--output` defaults to `identity_predictions_secondary.csv` beside the input.
- `--samples-per-identity` defaults to 100 evenly spaced crops.
- `--batch-size` defaults to 128.
- `--device` accepts an Ultralytics device such as `0` or `cpu`; default `auto`.

### `findCenter.py`

Finds the arena center without cropping the video. The middle frame is used by
default, avoiding black startup footage in long recordings.

```bash
python findCenter.py /data/pi14.2026.04.20.06.00.mp4 --show-score
```

- Positional `video` is required.
- `--template` overrides automatic white/black template selection.
- `--frame N` uses a specific zero-based frame instead of the middle frame.
- `--show-score` prints the normalized template-match score.

The printed center is in original, unpadded video coordinates and can be passed
to `--center X Y` in the cross-video matcher.

### `cropByCenter.py`

Legacy/debug utility copied from FindYourCenter. It finds the same arena center
and crops a video. Cross-video identity matching uses `findCenter.py` and does
not need to crop the source.

- `--in_file`, `-i`: required image or video.
- `--out_file`, `-o`: cropped output path.
- `--pad_log`, `-l`: padded-file log; default `./pad_log.txt`.
- `--template`, `-t`: explicit match template.
- `--frame`, `-f`: frame to inspect.
- `--visualize`, `-v`: show diagnostic plots.
- `--dump`, `-d`: print debugging information.

### `build_fish_assignment_table.py`

Project-specific utility that expands `SurpriseWinnerEffectEvents.csv` into the
long-form video/identity/location intervals consumed by cross-video matching.

```bash
python build_fish_assignment_table.py \
  --events SurpriseWinnerEffectEvents.csv \
  --output SurpriseWinnerEffectVideoAssignments.csv
```

- `--events`: source event CSV; default `SurpriseWinnerEffectEvents.csv`.
- `--output`: destination assignment CSV; default
  `SurpriseWinnerEffectVideoAssignments.csv`.

The biological event vocabulary, Pi range, dates, role order, and standard tank
locations are encoded in this script. Review it before applying it to another
experiment.

### `motion_processor.py`

Shared library, not a standalone command. `MotionProcessor` incrementally builds
motion images for sequential, exponential, and sliding-median strategies.
`build_motion_image()` provides the same operation for a finite frame sequence,
and `motion_warmup_samples()` reports the required history length.

## Output and compatibility notes

- Crop manifests retain BehaveAI `id`, `x`, and `y` fields and add source-video,
  timing, box, orientation, and `frame_step` metadata.
- Contrastive prediction files set `id` to the corrected identity and retain the
  source tracker ID in `original_id`. This keeps the output usable by programs
  written for ordinary BehaveAI track CSVs.
- An identity of `-1` means that a fragment could not be assigned confidently
  without violating co-occurrence constraints.
- Zero identity collisions means that no identity was assigned twice in a
  multi-detection frame. It does not by itself prove continuity across every
  track break.
- `input-color` is checkpoint metadata. Prediction restores the color mode and
  architecture stored in the checkpoint.
- A horizontal and vertical mirror are applied together during augmentation,
  producing a 180-degree equivalent. A horizontal-only mirror is not used.
