#!/bin/bash

if test $# -lt 1 
then
echo "Fits file argument is required"
echo "Example: $0 image1.fits"
echo "the input image must have the '.fits' suffix "
echo " "; exit                                          
fi
help()
{
cat << HELP
USAGE: $0   YOUR_FITS_IMAGE   NORMALIZER (equal to image width in pixels)
must have a suffix '.fits'
OPTIONS: -h help text
HELP
exit 0
}

[ -z "$1" ] && help
[ "$1" = "-h" ] && help

if [ -z "$2" ]; then
  echo -e "In order to calculate the normalization factor for MAPS_im2uv as N*N"
  echo -e "where NxN are the grid dimensions, please enter N:"
  read  ngrid
else
  let ngrid=$2
  echo "you entered: $ngrid, the normalizer is $normalizer"
fi
let normalizer=ngrid*ngrid
echo "The normalizer is $normalizer"

set -x   # display every command before executing it

MAPS_im2uv -i $1 -o  $(basename $1 .fits)_Visibility.dat -n $normalizer #-p 974
# -t
# -p 960


echo -e "now visgen is doing a NOISY simulation......\n"
visgen -n $(basename $1 .fits) -s ALMA50 -A $SIM/array/sgra_array.txt -G  $(basename $1 .fits)_Visibility.dat  -V sgra_obs_spec -v -p $(basename $1 .fits)n_uvdata.txt
echo -e "converting to uv-fits format.\n"

maps2uvfits $(basename $1 .fits).vis $(basename $1 .fits)n.uvfits -23.019278  -67.753167 5000  $SIM/array/sgra_array.txt

echo "cp $(basename $1 .fits)_uvdata.txt $(basename $1 .fits)_Visibility.dat $(basename $1 .fits).fits ../test"
cp $(basename $1 .fits)n_uvdata.txt $(basename $1 .fits)_Visibility.dat $(basename $1 .fits).fits ../lib2xrgaus

echo -e " "
echo -e "Done!\n"

exit
