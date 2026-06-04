#!/bin/bash

set -e

VENV_NAME="venv"

echo "------------------------------------"
echo "Setting up framework environment."

if [ ! -d "$VENV_NAME" ]; then
  python -m venv $VENV_NAME
  echo "Created new environment in $VENV_NAME"
fi

source "$VENV_NAME/bin/activate"

pip install --upgrade pip

echo "Installing framework."
pip install -e .

framework=(
  "vid_to_frames.py"
  "augmentations.py"
  "finetune.py"
  "quantification.py"
)

echo "------------------------------------"
echo "Running framework."

for script in "${framework[@]}"; do
  echo "------------------------------------"
  echo "Running the $script script."

  if [[ -f "$script" ]]; then
    python "$script"
  else
    echo "Error: $script was not found."
    exit 1
  fi

  echo "$script executed successfully."
done

echo "------------------------------------"
echo "Framework execution has been completed."

exit 0