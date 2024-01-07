# routes
Dashboard showing running route statistics from FIT files.
A route is a time series of geospatial data with additional running metadata such as
pace, distance, and optionally data from sensors (e.g. heart rate or cadence.)

## Installation
* Set up Virtualenv, do `pip install -r requirements.txt`
* Place your `.fit`-files in the directory pointed to by `utils.py`, or change the dir
```
$ python -i app/importfiles.py
```

```
> sync()
```

Exit Python and then run

```
$ python app/main.py
```
