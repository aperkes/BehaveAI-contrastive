<img width="400" height="69" alt="BehaveAI_400" src="https://github.com/user-attachments/assets/6fb5cd16-d266-4e8b-9513-1734a45813bf" />

This is a modified fork of the original BehaveAI (https://github.com/troscianko/BehaveAI)

You should check out the original, and this repo is unlikely to be completely stable, however it is designed to be compatible and preserve all functions of BehaveAI, while adding a few additional ones, most noteable contrastive learning for identity recognition. I've added a quick summary of the changes I've made. A good portion of this code was vibe coded using chatGPT, so proceed with all appropriate caution. 

CLI: 
simple_track.py began as a useful wrapper on BehaveAI for batch processing but now has several additional functions, including exporting a crop manifest for downstream contrastive learning.

Contrastive Learning: 
contrastive_identity.py uses the crop manifest and the number of individuals to learn identities. There are few additional help scripts including idenity_pipeline.py and identity_overlay.py. Proper docs should be forthcoming. 

Changes to functionality of BehaveAI
I wanted to play around with some alternative motion algorithms, including simple background subtraction. It made sense to restructure BehaveAI slightly so that the motion generation was a single function call, and have these functions contained in motion_processor.py, so that I could change that function depending on the method used. 
While i was at it, I allowed for the motion to be based on a different stride, making it possible to intergrate longer timescales of motion without skipping frames in the tracking itself. 

Some minor bug-fixes and optimizations:
When assigned tracks, it was possible for two detections to receive the same track ID. ([9643cd7](https://github.com/troscianko/BehaveAI/commit/9643cd7fa28c5fddcff6f199433e2f9c886724e2))
Oriented-box crops are extracted a single time and then rotated instead of rotating the entire image four times. ([f826da1](https://github.com/troscianko/BehaveAI/commit/f826da14fbb8f8bcda9181eaf4e80f05cb31280a))
Tracking expected a YAML file to exist even if that model type (e.g., motion) was disabled. ([644bffb](https://github.com/troscianko/BehaveAI/commit/644bffb2ea0fde1f43261ec54af8de3e3178343f))

I'm working on proper documentation of all the new code. For now I had chatGPT spit one out, see DOCS.md.

Everything below is the original README:
# A framework for detecting, classifying and tracking moving objects

BehaveAI is a user-friendly tool that identifies and classifies animals and other objects in video footage from their movement as well as their static appearance. The framework converts motion information into false colours that allow both the human annotator and convolutional neural network (CNN) to easily identify patterns of movement. The framework integrates both motion streams and static streams in a similar fashion to the mammalian visual system, separating the tasks of detection and classification.

The framework also supports hierarchical models (e.g. detect something from it's movement, then work out what exactly it is from conventional static appearance, or vice-versa); and semi-supervised annotation, allowing the annotator to rapidly correct errors made by initial models, making for a more efficient and effective training process.

![BehaveAI_flies](https://github.com/user-attachments/assets/99ae83fb-c001-4d5a-8338-0607a914d0c4)


#### Key features:
- Identifies objects and their behaviour based on motion and/or static appearance
- Fast user-friendly annotation - in under an hour you can create a powerful tracking model
- Identifies small (2 px!), fast moving, and motion-blurred targets
- Can track any type(s) of animal/object
- Tracks multiple individuals, classifying the behaviour of each independently
- Semi-supervised annotation allows you to learn where the models are weak and focus on improving performance
- Tools for inspecting and editing the annotation library
- Live (edge) support - record videos, train, then run live video processing using the user interface
- Built around the versatile [YOLO](https://github.com/ultralytics/ultralytics) (You Only Look Once) architecture
- Computationally efficient - runs fine on low-end devices without GPUs
- Intuitive user interface with installers for Windows and Linux (including Raspberry Pi) and full user-interface. Also works on MacOS but currently no installer.
- Free & open source ([GNU Afferro General Public License](https://github.com/troscianko/BehaveAI/blob/main/LICENSE))
- NEW (September 2026) supports oriented bounding boxes and can track the orientation of objects/animals.

## User guide & installation instructions

See the project wiki [here](https://github.com/troscianko/BehaveAI/wiki) for detailed user guide and installation instructions.

## Paper & Citation:
[PLOS Biology publication](https://doi.org/10.1371/journal.pbio.3003632)

If you use BehaveAI please cite:

* Troscianko, Jolyon, Thomas A. O’Shea-Wheller, James A. M. Galloway, and Kevin J. Gaston. (2026) 'BehaveAI Enables Rapid Detection and Classification of Objects and Behavior from Motion’. _PLOS Biology_ 24, no. 2 (2026): e3003632. https://doi.org/10.1371/journal.pbio.3003632.


## Video Guide (v1.2):
[<img width="350" alt="Screenshot from 2025-11-04 17-38-49" src="https://github.com/user-attachments/assets/5d76855e-d24f-4107-a6b9-c13aa98e6f79" />](https://www.youtube.com/watch?v=atEL14nxz9s)


See the [project Wiki](https://github.com/troscianko/BehaveAI/wiki) for detailed instructions.
