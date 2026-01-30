#!/usr/bin/env bash

datadir="data/"
file_name="${1:-${datadir}out_kapt_2_0_D_0_1_H_1_7_em5_1024x1024.h5}"
# datadir="data_scan/"
# file_name="${1:-${datadir}out_kapt_1_0_D_0_1_H_6_5_em6.h5}"
# datadir="data_2d3c/"
# file_name="${1:-${datadir}out_2d3c_kapt_2_0_D_0_025_kz_0_1_1024x1024.h5}"
# check h5clear exists
if ! command -v h5clear >/dev/null 2>&1; then
  echo "h5clear not found in PATH" >&2
  exit 2
fi

# check file exists
if [ ! -f "$file_name" ]; then
  echo "File not found: $file_name" >&2
  exit 3
fi

# run h5clear
h5clear -s "$file_name"