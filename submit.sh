#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J RADS_PROCESSING[1991-1999]
### -- ask for number of cores (default: 1) -- 
#BSUB -n 8 
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=4GB]"
### -- set walltime limit: hh:mm -- 
#BSUB -W 72:00 
### -- set the email address -- 
### BSUB -u MAIL@gmail.com
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_%J_%I.out
#BSUB -e Output_%J_%I.err 

### Execute script
python3 grid.py >  Out_$LSB_JOBINDEX.out
