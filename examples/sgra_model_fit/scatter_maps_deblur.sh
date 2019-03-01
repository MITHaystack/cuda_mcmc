#!/bin/bash

#
# scatter_maps_deblur.sh
#

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

bfname=$(basename $1 .fits)

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


#
# Scatter the fits image in $1 and save it in the almost namesake file
# but with 's' suffix
#
scatter-better -i $1 -o ${bfname}s.fits

#
# Run MAPS simulation toolchain with noise (i.e without the visgen -N option)
#
MAPS_im2uv -i ${bfname}s.fits -o  ${bfname}s_Visibility.dat -n $normalizer

echo -e "now visgen is doing a NOISY simulation......\n"

visgen -n ${bfname}s -s ALMA50 -A $SIM/array/sgra_array.txt \
       -G ${bfname}s_Visibility.dat  -V sgra_obs_spec -v \
       -p ${bfname}sn_uvdata.txt

#
# Currently the .uvfits files are not needed
#
# echo -e "converting to uv-fits format.\n"

# maps2uvfits ${bfname}s.vis ${bfname}sn.uvfits -23.019278  -67.753167 5000  \
#             $SIM/array/sgra_array.txt

#echo "cp ${bfname}_uvdata.txt ${bfname}_Visibility.dat ${bfname}.fits ../test"
#cp ${bfname}n_uvdata.txt ${bfname}_Visibility.dat ${bfname}.fits ../lib2xrgaus

#
# Descatter the UV-data obtained from the MAPS simulated observation of the
# scattered image in the fits file with 's' suffix
#
deblur_uv  -i ${bfname}sn_uvdata.txt -o ${bfname}snd_uvdata.txt

set +x   # disable display every command before executing it


echo -e " "
echo -e "Done!\n"






