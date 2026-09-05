#!/bin/bash
# Run the flux-partition variants and check them against each other.
#
#   [MPIRUN="mpirun -np 4"] ./run_partition.sh /path/to/erf_exec [extra erf args...]
#   SKIP_RUN=1 ./run_partition.sh x     # only rebuild the table from existing logs
#
# Every step with fire_debug prints the sensible flux handed to the
# atmosphere and, when injected, the latent flux. The table shows the values
# at the last coupling step, the partition factor the code reports and the
# burned fire cells. The checked decks run one-way (fire_atm_feedback = 0),
# so the fire evolves identically and the checks are exact: the default deck
# equals the legacy deck line for line; the cfbm sensible flux is 1/(1+M_f)
# times the legacy one at every step; the latent flux is identical under
# both partitions; the factor is applied with the latent flux off; and the
# wet deck reports 1/1.30. The two *_2way decks inject the fluxes and are
# listed for comparison only: their fire fronts drift apart by a step or
# two as the weaker heating changes the wind. The three smoke_* decks turn
# the smoke tracer on: the emission must be identical under both partitions
# (the factor is divided back out before the fuel burnt is formed), and
# smoke_heat_from_fuel must scale it by smoke_heat_of_comb over the fuel
# model's heat content.

set -u

EXE=${1:?usage: run_partition.sh /path/to/erf_exec [extra args]}
shift || true

VARIANTS="default legacy cfbm cfbm_wet cfbm_nolatent legacy_2way cfbm_2way smoke_legacy smoke_cfbm smoke_cfbm_fuel"

for v in $VARIANTS; do
    log="run_${v}.log"
    if [ "${SKIP_RUN:-0}" = "1" ] && [ -f "$log" ]; then continue; fi
    ${MPIRUN:-} "$EXE" "inputs_${v}" "$@" > "$log" 2>&1 || { echo "run $v failed (see $log)"; exit 1; }
done

sens() { grep 'Sensible flux to the atmosphere' "run_$1.log" | sed 's/.*max \([^ ]*\) W.*/\1/'; }
lat()  { grep 'Max latent flux' "run_$1.log" | sed 's/.*Max latent flux: \([^ ]*\) W.*/\1/'; }
fac()  { grep 'Sensible flux to the atmosphere' "run_$1.log" | tail -1 | sed 's/.*f_dry_fuel=\([^,]*\),.*/\1/'; }
mf()   { grep 'Sensible flux to the atmosphere' "run_$1.log" | tail -1 | sed 's/.*fuel moisture=\([^)]*\)).*/\1/'; }
cells(){ grep 'active fire cells' "run_$1.log" | tail -1 | awk '{print $NF}'; }
smk()  { grep 'Phase 4 smoke' "run_$1.log" | sed 's/.*smoke_src_max=\([^ ]*\) kg.*/\1/'; }
hpk()  { grep 'Phase 4 smoke' "run_$1.log" | tail -1 | sed 's/.*heat_per_kg=\([^ ]*\) J.*/\1/'; }

printf "%-16s %6s %8s %8s %12s %12s %7s %15s %11s\n" variant M_f f_dry steps sens_max_W latent_max_W cells smoke_src_max heat_per_kg
printf "%-16s %6s %8s %8s %12s %12s %7s %15s %11s\n" ---------------- ------ -------- -------- ------------ ------------ ------- --------------- -----------
for v in $VARIANTS; do
    printf "%-16s %6s %8s %8s %12s %12s %7s %15s %11s\n" "$v" "$(mf $v)" "$(fac $v)" "$(sens $v | wc -l | tr -d ' ')" \
        "$(sens $v | tail -1 | cut -c1-12)" "$(lat $v | tail -1 | cut -c1-12)" "$(cells $v)" "$(smk $v | tail -1 | cut -c1-15)" "$(hpk $v)"
done
echo

status=0
check() { if [ "$2" = "1" ]; then echo "  $1: PASS"; else echo "  $1: FAIL"; status=1; fi; }

# 1. default == legacy, every flux line identical
same=$(diff <(sens default; lat default) <(sens legacy; lat legacy) > /dev/null && echo 1 || echo 0)
check "default deck reproduces the legacy deck ($(sens default | wc -l | tr -d ' ') sensible lines)" "$same"

# 2. cfbm sensible = f_dry * legacy sensible at every step (relative 1e-6), and the factor is 1/(1+M_f)
r=$(paste <(sens legacy) <(sens cfbm) | awk -v f="$(fac cfbm)" -v mf="$(mf cfbm)" '
    BEGIN { ok = 1; worst = 0 }
    { if ($1 > 0) { d = ($2 - f * $1) / $1; if (d < 0) d = -d; if (d > worst) worst = d; if (d > 1e-6) ok = 0 } }
    END { fe = 1.0 / (1.0 + mf); e = f - fe; if (e < 0) e = -e; if (e > 1e-9) ok = 0; printf "%d %.2e %.6f %.6f", ok, worst, f, fe }')
set -- $r
check "cfbm sensible flux = f_dry x legacy at every step (worst rel. diff $2; f_dry $3, 1/(1+M_f) $4)" "$1"

# 3. latent flux identical between legacy and cfbm
same=$(diff <(lat legacy) <(lat cfbm) > /dev/null && echo 1 || echo 0)
check "latent flux unchanged by the partition ($(lat legacy | wc -l | tr -d ' ') lines)" "$same"

# 4. factor applied with the latent flux off: same sensible flux as cfbm, no latent lines
r=$(diff <(sens cfbm) <(sens cfbm_nolatent) > /dev/null && [ "$(lat cfbm_nolatent | wc -l | tr -d ' ')" = "0" ] && echo 1 || echo 0)
check "factor applied with inject_latent = false (no latent lines, same sensible flux)" "$r"

# 5. wet deck factor
r=$(awk -v f="$(fac cfbm_wet)" 'BEGIN { e = f - 1.0/1.30; if (e < 0) e = -e; print (e < 1e-9) ? 1 : 0 }')
check "wet deck reports f_dry = 1/1.30 ($(fac cfbm_wet))" "$r"

# 6. smoke emission identical under both partitions (the factor is divided back out)
same=$(diff <(smk smoke_legacy) <(smk smoke_cfbm) > /dev/null && echo 1 || echo 0)
check "smoke source identical under legacy and cfbm partitions ($(smk smoke_legacy | wc -l | tr -d ' ') lines; heat_per_kg $(hpk smoke_legacy) vs $(hpk smoke_cfbm) with the factor folded in)" "$same"

# 7. smoke_heat_from_fuel scales the emission by smoke_heat_of_comb / h_fuel at every line
r=$(paste <(smk smoke_cfbm) <(smk smoke_cfbm_fuel) | awk -v h0="$(hpk smoke_cfbm)" -v h1="$(hpk smoke_cfbm_fuel)" '
    BEGIN { ok = 1; worst = 0; e = h0 / h1 }
    { if ($1 > 0) { d = ($2 / $1 - e) / e; if (d < 0) d = -d; if (d > 1e-6) ok = 0; if (d > worst) worst = d } }
    END { printf "%d %.2e %.6f", ok, worst, e }')
set -- $r
check "smoke_heat_from_fuel scales the source by smoke_heat_of_comb / h_fuel = $3 at every line (worst rel. diff $2)" "$1"

# Two-way pair, informational: steps at which the sensible flux ratio departs from f_dry.
n=$(paste <(sens legacy_2way) <(sens cfbm_2way) | awk -v f="$(fac cfbm_2way)" '{ if ($1 > 0) { d = ($2 - f * $1) / $1; if (d < 0) d = -d; if (d > 1e-6) n++ } } END { print n + 0 }')
echo "  two-way pair: $n of $(sens legacy_2way | wc -l | tr -d ' ') steps depart from the factor (front timing drift), cells $(cells legacy_2way) vs $(cells cfbm_2way)"

[ $status -eq 0 ] && echo "ALL PASS" || echo "SOME CHECKS FAILED"
exit $status
