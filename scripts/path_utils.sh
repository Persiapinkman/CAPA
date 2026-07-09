#!/usr/bin/env bash

# Shared path helpers for CAPA shell entrypoints.
# Relative model paths are resolved against the migrated /raid/zkq layout.

CAPA_STORAGE_ROOT="${CAPA_STORAGE_ROOT:-/raid/zkq}"
CAPA_MODELS_ROOT="${CAPA_MODELS_ROOT:-$CAPA_STORAGE_ROOT/models}"

resolve_model_dir() {
  local raw_path="$1"

  case "$raw_path" in
    /*)
      printf '%s\n' "$raw_path"
      ;;
    models/*)
      printf '%s\n' "$CAPA_STORAGE_ROOT/$raw_path"
      ;;
    ./*|../*)
      realpath -m "$ROOT_DIR/$raw_path"
      ;;
    *)
      if [[ -e "$ROOT_DIR/$raw_path" ]]; then
        realpath -m "$ROOT_DIR/$raw_path"
      else
        printf '%s\n' "$CAPA_MODELS_ROOT/$raw_path"
      fi
      ;;
  esac
}
