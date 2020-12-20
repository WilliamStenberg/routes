# routes
Dashboard showing running route statistics from FIT files.
A route is a time series of geospatial data with additional running metadata such as
pace, distance, and optionally data from sensors (e.g. heart rate or cadence.)

## Installation
TOTO: write
* Virtualenv
* MongoDB (local installation example)

## Roadmap
* Parse FIT-files in `data/fitfiles` into dataframes
* Aggregate route properties (mean pace, total distance...)
* Save into MongoDB database
* Fetch satellite map for location, store the image in `data/maps`
* Plotly Dashboard for single route
* Plotly Dashboard routes overview (aggregations)
