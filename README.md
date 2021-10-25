# alphafold-singularity
alphafold setup with Singularity

To bulild Singularity image
singularity build alphfold.sif alphfold.def

Modify run-alphafold.bash to use proper alphfold.sif and alphafold data folder
Modify run-alphafold.py to use correct path for run-alphafold.bash
