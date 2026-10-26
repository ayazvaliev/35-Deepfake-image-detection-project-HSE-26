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

download_model() {
    local url=$1 model=$2
    if [ -f "$model" ]; then
        echo "$model уже есть, пропускаю"
        return
    fi
    mkdir -p models
    curl -fsSL -o "$model.part" "$url"
    mv "$model.part" "$model"
}

download_model https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task models/face_landmarker.task
download_model https://github.com/opencv/opencv_zoo/raw/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/face_detection_yunet_2023mar.onnx models/face_detection_yunet_2023mar.onnx
