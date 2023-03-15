# Requirements

## Installion requirements
[RADS](https://github.com/remkos/rads) has to be installed on the system for downloading the satellite data. Install [GMT](https://www.generic-mapping-tools.org/) for getting the correct ocean masks. If [GMT](https://www.generic-mapping-tools.org/) is installed it would download the masks automatically. Make sure the python packages in the [requirements.txt](requirements.txt) file is installed.

## Environment variables
The following environment variables have to be set for step 4 in the pipeline. The environment variables are needed to make a connection to a PostgreSQL database.

| Environment variable                                  | Explanation                                                 |
|--------------------------------------|-------------------------------------------------------------|
| ALTIMETRY_USERNAME                   | Username for the database                                   |
| ALTIMETRY_PASSWORD                   | Password for the database                                   |
| ALTIMETRY_HOST                       | Hostname of the database                                    |
| ALTIMETRY_DATABASE_PORT              | Port of the database                                        |
| ALTIMETRY_DATABASE                   | Database name                                               |
| ALTIMETRY_DATABASE_CONNECTION_ENGINE | Engine to use for the connection (likely 'psycopg2')          |
| ALTIMETRY_DATABASE_TYPE              | Type of database (likely 'postgresql')                        |
| ALTIMETRY_CREATE_TABLES              | If the connection should create the tables ('true'/'false') |


# Suggestion

## Pythran
Before running the scripts in the pipeline use ```pythran``` on  [grid_funcs.py](src/grid_funcs.py). This will make a ~100 times speedup on the block mean process.
```bash
pythran src/grid_funcs.py
```

## Databases
If the tables in the database is not setup before running step 4 in the pipeline it is strongly recommended to set the environment variable ```ALTIMETRY_CREATE_TABLES='true'```.

# Pipeline
1. Generate the cycles for each satellite
```bash
./getAllRadsData.sh
```
Feel free to change the amount of cycles to download. Especially for the later satellites as they are still producing data.
2. Execute the following python script to process the data and put them into the correct formats
```python
python process_files.py
```
3. Execute the following python script for gridding the data.
```python
python grid.py
```
4. Upload grids to a database
```python
python upload_data.py
```
