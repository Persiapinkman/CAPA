# External Artifacts

Large or regenerated files are stored outside the Git worktree. Existing paths such as `outputs/`, `logs/`, `.hf-cache`, and local virtual environments remain as compatibility symlinks.

The canonical physical root is recorded in `location.json`. Experiment records must store an artifact URI or absolute path plus a content hash when practical; they must not assume that a checkpoint is present in Git.
