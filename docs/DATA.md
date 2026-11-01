# Data and reference protocol

Obtain RWD from its original providers and respect the dataset/image licenses. Dataset images are not bundled.

```text
assets/rwd/
├── JPEGImages/<scene>/*.jpg
├── Annotations/<scene>/*.xml
├── ImageSets/<scene>/...
├── DATASET_MANIFEST.json
└── ACTIVE_ANNOTATION_REVISION.json
```

`--data /absolute/path/to/rwd` can point at an external dataset root containing JPEGImages and Annotations. Scene names are Aquatic, Aerial, Game, Medical and Surgical. Keep original basenames and EXIF handling.

The deployment protocol is stored in `protocols/<scene>/<task>.json`. References are exclusively the legal fit objects recorded in the released model's reference manifest. Calibration and test images do not supply extra references. Reference-bank extraction uses the original 2×2 region plus 1×1 context descriptor from frozen DINO; it does not reinterpret the descriptor as semantic parts.

Game corrections are recorded in `assets/rwd/ACTIVE_ANNOTATION_REVISION.json` and `prepared/annotation_revisions/`. Apply the reviewed labels when reconstructing the reference inputs. Historical model-training provenance remains unchanged; a corrected bank alone does not retrain a detector.

Task 1 merges unintroduced foreground categories into unknown. Task 2 adds the currently known categories to the known set. Keep the saved class order: the last predicted class is unknown, rather than an extra dataset class-name entry.

Images and annotation files are never silently downloaded by the inference interface. XML is only read during the separate evaluation phase after predictions are written.
