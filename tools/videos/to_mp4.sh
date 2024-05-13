#!/bin/bash

# Check if exactly one argument is provided
if [ $# -ne 1 ]; then
  echo "Usage: $0 filename"
  exit 1
fi

# Extract the filename without extension
filename="${1%.*}"

# Echo the filename with the new extension
newname="${filename}.mp4"

# Convert to mp4
ffmpeg -i $1 -pix_fmt yuv420p -c:v h264 -crf 10 $newname
