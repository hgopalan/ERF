#!/usr/bin/env python3
"""Negative test for Tests/check_fire_dust_inputs.py.

    python3 Tests/test_check_fire_dust_inputs.py [repo_root]

check_fire_dust_inputs.py is the only thing that catches an erf.fire.* or
erf.dust.* key nothing reads, because ParmParse accepts an unread key in silence.
A checker that never fails would pass the build just as well as a correct one, so
this test feeds it decks whose verdict is known:

  * decks that set a dead key -- a misspelt name, a misspelt unit suffix, a
    property under the wrong prefix -- must fail, and the message must name the
    dead key;
  * a deck that sets only live keys, including the two families whose names the
    code builds at run time (erf.fire.custom_fuel.<code>.* and
    erf.fire.firebreak.<n>.*), must pass, so that the families are recognised by
    being read rather than by being skipped;
  * with the source mutated so that a property is read under a new name, the
    checker must fail on the decks that still use the old one.  That case is what
    keeps the custom_fuel property list derived from ERF_CustomFuel.cpp instead of
    hard-coded here, and its sibling -- a property moved from pp.query to
    pp.contains -- must still be seen as read;
  * misuse of the command line (a deck path that is not there, --extra-deck with
    no path, an unknown option) must exit non-zero, because a deck silently
    dropped is a pass that scanned one deck fewer than it was asked to.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHECKER = os.path.join(ROOT, "Tests", "check_fire_dust_inputs.py")
CUSTOM_FUEL_SRC = os.path.join(ROOT, "Source", "Fire", "ERF_CustomFuel.cpp")

# A block of keys the code really does read, used as the control deck and as the
# body every "one dead key" deck is built from.
LIVE_DECK = """\
erf.fire.fuel_model_id        = 1007
erf.fire.custom_fuel.codes    = 1007
erf.fire.custom_fuel.1007.name              = shadow_fuel
erf.fire.custom_fuel.1007.w_1h_kg_m2        = 0.6
erf.fire.custom_fuel.1007.sav_1h_1_m        = 5000.0
erf.fire.custom_fuel.1007.depth_m           = 0.6
erf.fire.custom_fuel.1007.moisture_ext      = 0.25
erf.fire.custom_fuel.1007.heat_content_J_kg = 1.86e7
erf.fire.firebreak.0.type     = rectangle
erf.fire.firebreak.0.x_lo     = 100.0
"""

# key set in the deck -> the text the checker must print for it
DEAD_KEYS = {
    # a plain misspelling under erf.fire.
    "erf.fire.no_such_fire_key": "erf.fire.no_such_fire_key",
    # a per-fuel property with a misspelt unit suffix: the name is one letter from
    # a live one, and only a checker that knows the real property list can tell
    "erf.fire.custom_fuel.1007.w_1h_kg_m3": "erf.fire.custom_fuel.N.w_1h_kg_m3",
    # a per-fuel property spelt without the fuel code, so it is not a key at all
    "erf.fire.custom_fuel.w_1h_kg_m2": "erf.fire.custom_fuel.w_1h_kg_m2",
    # "codes" misspelt: the one custom_fuel key that is a literal in the source
    "erf.fire.custom_fuel.code": "erf.fire.custom_fuel.code",
    # the other run-time-built family
    "erf.fire.firebreak.0.radius_m": "erf.fire.firebreak.N.radius_m",
    # and the dust side
    "erf.dust.no_such_dust_key": "erf.dust.no_such_dust_key",
}

def run_argv(args):
    p = subprocess.run([sys.executable, CHECKER] + args, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True)
    return p.returncode, p.stdout

def run(root, extra_deck=None):
    args = [root]
    if extra_deck:
        args += ["--extra-deck", extra_deck]
    return run_argv(args)

def write_deck(tmp, name, text):
    path = os.path.join(tmp, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path

def shadow_root(tmp, mutate):
    """A copy of ROOT in which Source/Fire/ERF_CustomFuel.cpp is the only real
    file -- everything the checker reads is symlinked -- so the source can be
    mutated without touching the tree."""
    root = tempfile.mkdtemp(prefix="shadow", dir=tmp)
    os.makedirs(os.path.join(root, "Source", "Fire"))
    for entry in os.listdir(ROOT):
        if entry != "Source":
            os.symlink(os.path.join(ROOT, entry), os.path.join(root, entry))
    for entry in os.listdir(os.path.join(ROOT, "Source")):
        if entry != "Fire":
            os.symlink(os.path.join(ROOT, "Source", entry),
                       os.path.join(root, "Source", entry))
    for entry in os.listdir(os.path.join(ROOT, "Source", "Fire")):
        src = os.path.join(ROOT, "Source", "Fire", entry)
        dst = os.path.join(root, "Source", "Fire", entry)
        if entry == os.path.basename(CUSTOM_FUEL_SRC):
            with open(src, encoding="utf-8") as f:
                text = f.read()
            with open(dst, "w", encoding="utf-8") as f:
                f.write(mutate(text))
        else:
            os.symlink(src, dst)
    return root

def main():
    failures = []
    mutation_ran = False

    rc, out = run(ROOT)
    if rc != 0:
        print(out)
        print("FAIL  the tree itself does not pass check_fire_dust_inputs.py, so this "
              "test cannot tell a working checker from a broken one; fix "
              "FireDustInputsDocs first")
        return 1

    tmp = tempfile.mkdtemp(prefix="fire_dust_inputs_neg_")
    try:
        # 1. a deck of live keys only, including both run-time-built families
        path = write_deck(tmp, "inputs_live_only", LIVE_DECK)
        rc, out = run(ROOT, path)
        if rc != 0:
            bad = [l for l in out.splitlines() if "inputs_live_only" in l]
            failures.append("a deck of keys the code reads was rejected: "
                            + ("; ".join(bad) if bad else out.strip()))

        # 2. one dead key at a time, on top of the same live block
        for i, (dead, expect) in enumerate(sorted(DEAD_KEYS.items())):
            name = "inputs_dead_%d" % i
            path = write_deck(tmp, name, LIVE_DECK + dead + " = 1.0\n")
            rc, out = run(ROOT, path)
            want = "sets a key nothing reads: " + expect
            if rc == 0:
                failures.append("the checker passed a deck setting the dead key " + dead)
            elif want not in out:
                got = [l for l in out.splitlines() if name in l] or ["(nothing about " + name + ")"]
                failures.append("setting %s should report %r, got: %s"
                                % (dead, want, " | ".join(got)))
            for line in out.splitlines():
                # it must not blame any of the live keys in the same deck
                if name in line and expect not in line:
                    failures.append("setting %s also reported a live key: %s" % (dead, line.strip()))
                # and this deck is outside the tree, so it must be named as it was
                # given.  A walk-up relative path is not only unreadable here, it is
                # something os.path.relpath cannot build at all across two Windows
                # drives, where it raises ValueError instead.
                if name in line and os.pardir + os.sep in line:
                    failures.append("an out-of-tree deck was reported by a walk-up path: "
                                    + line.strip())

        # 3. misuse of the command line must be loud.  deck_files() drops anything
        #    that is not a file, so a mistyped --extra-deck would otherwise leave the
        #    run passing having scanned one deck fewer than it was asked to.
        for argv, what in (
                ([ROOT, "--extra-deck", os.path.join(tmp, "inputs_not_there")],
                 "a deck that is not there"),
                ([ROOT, "--extra-deck"], "--extra-deck with no path"),
                ([ROOT, "--no-such-option"], "an unknown option"),
                ([ROOT, ROOT], "two repo roots")):
            rc, out = run_argv(argv)
            if rc == 0:
                failures.append("the checker accepted %s and reported PASS" % what)

        # 4. the custom_fuel property names must come from the source, not from a
        #    list inside the checker: rename one in a shadow copy of the source and
        #    the decks that still set the old name must be reported
        def rename(text):
            new, n = re.subn(r'\bneed\s*\(\s*"w_1h_kg_m2"', 'need("w_1h_kg_m2_renamed"', text)
            if n != 1:
                raise SystemExit("test_check_fire_dust_inputs.py: cannot find the "
                                 "w_1h_kg_m2 read in " + CUSTOM_FUEL_SRC)
            return new

        # pp.contains is a read too -- it decides whether the deck set the key -- and
        # the checker's literal scan already counts it.  The per-property scan has to
        # agree, or moving one property to contains would make it read as dead.
        def to_contains(text):
            new, n = re.subn(r'pp\.query\(\(pre \+ "burnout_time_s"\)\.c_str\(\), burn_s\);',
                             'pp.contains((pre + "burnout_time_s").c_str());', text)
            if n != 1:
                raise SystemExit("test_check_fire_dust_inputs.py: cannot find the "
                                 "burnout_time_s read in " + CUSTOM_FUEL_SRC)
            return new
        try:
            # the shadow tree must be faithful, or a failure below would prove nothing
            rc, out = run(shadow_root(tmp, lambda text: text))
            if rc != 0:
                failures.append("the unmutated shadow tree does not pass, so the mutation "
                                "below proves nothing: " + " | ".join(
                                    l for l in out.splitlines() if l.startswith("FAIL"))[:400])
            rc, out = run(shadow_root(tmp, rename))
            rc_c, out_c = run(shadow_root(tmp, to_contains))
        except OSError as e:
            # the shadow tree is symlinked, which an unprivileged Windows account
            # cannot do; the rest of this test carries on without it
            print("SKIP  the mutated-source cases need symlinks: %s" % e)
            mutation_ran = False
        else:
            mutation_ran = True
            if rc == 0:
                failures.append("the checker still passed after w_1h_kg_m2 was renamed in "
                                "ERF_CustomFuel.cpp, so it is not reading the property names "
                                "out of the source")
            elif "sets a key nothing reads: erf.fire.custom_fuel.N.w_1h_kg_m2" not in out:
                failures.append("after renaming w_1h_kg_m2 the checker failed for another "
                                "reason: " + " | ".join(l for l in out.splitlines()
                                                        if l.startswith("FAIL"))[:400])
            if rc_c != 0:
                failures.append("a property read through pp.contains instead of pp.query "
                                "read as dead: " + " | ".join(l for l in out_c.splitlines()
                                                              if l.startswith("FAIL"))[:400])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    for f in failures:
        print("FAIL  " + f)
    if failures:
        print("%d problem(s)" % len(failures))
        return 1
    print("PASS  check_fire_dust_inputs.py accepts %d live keys and reports each of the %d "
          "dead keys by name%s"
          % (len(LIVE_DECK.strip().splitlines()), len(DEAD_KEYS),
             ", follows a renamed property in the source and reads one through pp.contains"
             if mutation_ran else ""))
    return 0

if __name__ == "__main__":
    sys.exit(main())
