#!/usr/bin/env bash

datadir="data/"
file_name="${1:-${datadir}out_kapt_1_2_chi_0_1_H_1_0_em3.h5}"

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