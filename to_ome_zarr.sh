#!/usr/bin/env bash

for ff in *.tif; do
  echo "Converting $ff"
  # For 12x downsample, 1.8x1.8x5.5 um x y z spacing
  ngff-zarr -i $ff -o $(basename $ff .tif).ome.zarr \
    --input-backend itk \
    -m dask_image_gaussian \
    -s z 66.0 y 21.6 x 21.6 \
    -u z micrometer y micrometer x micrometer
done
