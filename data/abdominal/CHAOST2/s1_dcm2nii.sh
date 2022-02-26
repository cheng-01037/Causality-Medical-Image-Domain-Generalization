#!bin/bash
# Convert dicom-like images to nii files in 3D
# This is the first step for image pre-processing

# Feed path to the downloaded data here
DATAPATH=./MR # please put chaos dataset training fold here which contains ground truth

# Feed path to the output folder here
OUTPATH=./niis/T2SPIR

if [ ! -d  $OUTPATH ]
then
    mkdir -p $OUTPATH
fi

for sid in $(ls "$DATAPATH")
do
	dcm2nii -o "$DATAPATH/$sid/T2SPIR" "$DATAPATH/$sid/T2SPIR/DICOM_anon";
	find "$DATAPATH/$sid/T2SPIR" -name "*.nii.gz" -exec mv {} "$OUTPATH/image_$sid.nii.gz" \;
done;


