#!/bin/sh
# The watched action file: start the run with an empty file, append the line
# once the run has read the empty file, and check that the re-read (and its
# broadcast to every rank) stopped the head.
#
#   [MPIRUN="mpirun -np 2"] [METHOD=farsite] sh run_poll.sh /path/to/erf_exec [extra erf args...]
#
# The run polls the file every 5 fire steps; the front would reach the line
# at 31 s (step 62) of the 40 s run, so the line must be applied by then.

set -u
EXE=${1:?usage: run_poll.sh /path/to/erf_exec [extra args]}
shift || true
PYTHON=${PYTHON:-python3}
m=${METHOD:-levelset}
here=$(cd "$(dirname "$0")" && pwd)
[ -f erf_plotfile.py ] || cp "$here/../../CanonicalTests/Canonical_RANS/erf_plotfile.py" .

plt="plt_poll_${m}_"
log="suppression_poll_${m}.csv"
actions="actions_poll_${m}.txt"
runlog="run_poll_${m}.log"
rm -rf "${plt}"* "$log" "$runlog"
printf '# written empty by run_poll.sh; the line is appended while the run polls\n' > "$actions"

${MPIRUN:-} "$EXE" inputs_poll erf.fire.propagation_method="$m" erf.fire_plot_file="$plt" \
    erf.fire.suppression.log="$log" erf.fire.suppression.file="$actions" "$@" > "$runlog" 2>&1 &
pid=$!

# Wait for the start-up read of the empty file, then append the line.
n=0
while ! grep -q "Suppression: read 0 action" "$runlog" 2>/dev/null; do
    sleep 0.05
    n=$((n + 1))
    if ! kill -0 $pid 2>/dev/null; then echo "  poll/$m: the run ended before reading the empty file: FAIL"; cat "$runlog" | tail -20; exit 1; fi
    if [ $n -gt 2400 ]; then echo "  poll/$m: timed out waiting for the start-up read: FAIL"; kill $pid; exit 1; fi
done
printf 'L1  line  0  241 0 241 400  rate=100  -1  -\n' >> "$actions"
echo "  poll/$m: line appended to $actions after the start-up read"

wait $pid || { echo "  poll/$m: run failed (see $runlog)"; tail -20 "$runlog"; exit 1; }
$PYTHON check_suppression.py poll "${plt}00080" "$log"
