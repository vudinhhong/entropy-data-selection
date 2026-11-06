#!/bin/bash
# Run the groups in order. The GPU steps are independent and can be run
# separately.
#
#   ./run_all.sh check      verify the paper's numbers and the contrast  (no GPU)
#   ./run_all.sh figures    rebuild Fig 1 from the released table        (no GPU)
#   ./run_all.sh build | train | analyse | calibrate | tables            (GPU)
#   ./run_all.sh gpu        every GPU group, in order
set -eu
cd "$(dirname "$0")"
PY=${PY:-python3}

check()     { $PY 17_reproduce/verify_paper_numbers.py
              $PY 16_contrast/two_ended_contrast.py
              $PY 18_direction/direction_pilot.py
              $PY tests/test_pilot_config.py; }
figures()   { (cd 07_figures        && $PY make_figures.py \
                                    && $PY make_pilot_figures.py); }
build()     { (cd 01_build_store    && $PY build_store.py --stage all); }
train()     { (cd 02_train_students && $PY run_experiments.py); }
analyse()   { (cd 03_store_analysis && $PY analyse_store.py); }
calibrate() { (cd 04_calibration    && $PY measure_calibration.py); }
tables()    { (cd 05_results        && $PY make_tables.py); }

case "${1:-}" in
  check)     check ;;
  figures)   figures ;;
  build)     build ;;
  train)     train ;;
  analyse)   analyse ;;
  calibrate) calibrate ;;
  tables)    tables ;;
  gpu)       build; train; analyse; calibrate; tables ;;
  *)         sed -n '2,8p' "$0"
             echo "steps: check figures build train analyse calibrate tables gpu" ;;
esac
