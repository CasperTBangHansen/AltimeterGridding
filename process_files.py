from pathlib import Path
from src.fileHandler.concatenateFiles import process_netcdfs, concatenate_date_netcdf_files

def main():
    DATAPATH = Path("radsCycles")
    POLARPATH = Path("radsPolar")
    PROCESSEDPATH = Path("Processed_v4")
    FILEMATCHING = '*.nc'
    CROSSOVER_BASEPATH = Path("radsXover")
    process_netcdfs(DATAPATH, POLARPATH, CROSSOVER_BASEPATH, PROCESSEDPATH, FILEMATCHING)

    OUTFOLDER = PROCESSEDPATH / Path('all')
    INPUTFILES = PROCESSEDPATH.glob('[!all]*/*.nc')
    concatenate_date_netcdf_files(INPUTFILES, OUTFOLDER)

if __name__ == '__main__':
    main()