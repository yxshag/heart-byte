#!/bin/bash
# Downloads the PTB-XL dataset from PhysioNet.
set -e

DATA_DIR="$(dirname "$0")/../data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "Downloading PTB-XL v1.0.3 into $DATA_DIR ..."
echo "This will take a while depending on your connection (~1.8 GB)."

wget -r -N -c -np \
  https://physionet.org/files/ptb-xl/1.0.3/

echo ""
echo "Done. Files are under: $DATA_DIR/physionet.org/files/ptb-xl/1.0.3/"
echo ""
echo "Key files to know about:"
echo "  ptbxl_database.csv   - main metadata table (one row per ECG record)"
echo "  scp_statements.csv   - maps diagnostic codes to human-readable labels"
echo "  records100/          - waveforms downsampled to 100Hz"
echo "  records500/          - original waveforms at 500Hz"
echo ""
echo "Next step: run src/explore_data.py to do an initial sanity check."
