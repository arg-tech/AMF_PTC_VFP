#!/bin/sh
exec gunicorn -b :5003 --workers 1 --access-logfile - --error-logfile - app:application --timeout 300
