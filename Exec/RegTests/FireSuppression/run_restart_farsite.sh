#!/bin/sh
# run_restart.sh on the FARSITE path (the CTest registration cannot set METHOD)
METHOD=farsite exec sh "$(dirname "$0")/run_restart.sh" "$@"
