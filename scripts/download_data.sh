#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

download() {
    local slug=$1 dest=$2
    if [ -d "$dest" ]; then
        echo "$dest уже есть, пропускаю"
        return
    fi
    # качаем во временную папку, чтобы прерванная загрузка не оставила неполный датасет в data/raw
    local tmp
    tmp=$(mktemp -d)
    trap "rm -rf '$tmp'" EXIT
    kaggle datasets download -d "$slug" -p "$tmp" --unzip
    mv "$tmp" "$dest"
    trap - EXIT
}

download xhlulu/140k-real-and-fake-faces data/raw/140k
download manjilkarki/deepfake-and-real-images data/raw/openforensics

model=models/face_landmarker.task
if [ -f "$model" ]; then
    echo "$model уже есть, пропускаю"
else
    mkdir -p models
    curl -fsSL -o "$model.part" https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
    mv "$model.part" "$model"
fi
