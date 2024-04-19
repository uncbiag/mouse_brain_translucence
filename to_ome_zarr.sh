#!/usr/bin/env bash

# Copy this script to data directory containing the *.tif files.

for ff in *.tif; do
  echo "Converting $ff"
  ngff-zarr -i $ff -o $(basename $ff .tif).ome.zarr --input-backend itk -m dask_image_gaussian
done
