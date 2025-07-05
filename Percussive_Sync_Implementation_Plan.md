# Implementation Plan: Sync Video Cuts to High Hat or Kick Events

## 1. Enhance BeatSyncEngine for Percussive Event Detection
- **a.** In `beat_sync/beat_sync_engine.py`, add a method to perform onset detection with frequency analysis.
- **b.** For each detected onset, analyze the frequency content (e.g., using spectral centroid, bandwidth, or a simple filter bank).
- **c.** Classify each onset as likely "kick", "snare", "high hat", or "other" based on its frequency profile:
  - *Kick*: Low frequency (20–150 Hz)
  - *Snare*: Mid frequency (150–800 Hz)
  - *High Hat*: High frequency (3 kHz+)
- **d.** Store these classified onsets in the `BeatSyncResult` object (e.g., as `kick_times`, `hihat_times`, etc.).

## 2. Expose Percussive Event Timestamps
- **a.** Add methods to `BeatSyncEngine` to retrieve onset times for a specific class (e.g., `get_onset_times('hihat')`).
- **b.** Update any relevant interfaces to allow requesting a specific percussive event type.

## 3. Update MusicManager to Support Percussive Sync
- **a.** In `content/music_manager.py`, add support for requesting and storing high hat or kick times from the beat sync engine.
- **b.** Allow the user (or pipeline) to specify which percussive event to use for sync (default: "beat", options: "kick", "hihat", etc.).

## 4. Pipeline Integration
- **a.** In the video generation pipeline (e.g., `core/real_content_generator.py` and `content/content_selector.py`), add a parameter to select the percussive event type for sync.
- **b.** When selecting sync points, use the chosen percussive event times instead of generic beats.

## 5. Testing and Validation
- **a.** Write unit tests for onset classification (using synthetic or real drum audio).
- **b.** Write integration tests to ensure video cuts align with the chosen percussive event.
- **c.** Validate visually and aurally on several music tracks.

## 6. Documentation
- **a.** Update the README and code comments to explain the new percussive sync feature and how to use it.

---

### Optional: Advanced/Alternative Approaches
- Use a pre-trained drum transcription model (e.g., [madmom](https://github.com/CPJKU/madmom), [onsets-and-frames](https://github.com/magenta/onsets-and-frames)) for more accurate drum event detection.
- Allow manual override or fine-tuning of sync points for creative control.

---

**Ready for Claude Code to execute.** 