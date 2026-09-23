#!/usr/bin/env python3
"""Cross-check the fire and dust input keys three ways: what the code reads,
what the documentation lists, and what the decks set.

    python3 Tests/check_fire_dust_inputs.py [repo_root] [--extra-deck PATH]...

Fails (exit 1) when
  * a key documented anywhere under Docs/sphinx_doc is not read by the code
    (a renamed or misspelt key in the docs),
  * a key the code reads is missing from the Inputs.rst tables,
  * a fire, dust or hazard deck sets a key the code does not read (ParmParse
    never warns, so such a key is a silent no-op),
  * the fire master reference deck lacks a key the code reads, or
  * the dust inputs generator and the generated Inputs.rst table disagree with
    the dust parser.
ParmParse reads are collected from the five files that parse these keys; add a
file here if a new one starts reading erf.fire.* or erf.dust.* keys.

Two key families are not string literals in the source: the code builds their
names at run time from an index (erf.fire.firebreak.<n>.* and
erf.fire.custom_fuel.<code>.*). Both are indexed here as <family>.N.<property>,
the same spelling Inputs.rst uses. The custom_fuel property names are read out
of ERF_CustomFuel.cpp, so renaming one there fails this test; the firebreak ones
are listed in code_keys() and have to be kept in step by hand.

--extra-deck adds a deck to the ones scanned by rule 3; it exists so that
Tests/test_check_fire_dust_inputs.py can feed this checker decks that must
fail, and is not used by the build.
"""
import glob
import os
import re
import sys

def _parse_argv(argv):
    root, extra = None, []
    it = iter(argv)
    for a in it:
        if a == "--extra-deck":
            extra.append(next(it, None))
        elif a.startswith("--"):
            sys.exit("usage: check_fire_dust_inputs.py [repo_root] [--extra-deck PATH]...")
        elif root is None:
            root = a
        else:
            sys.exit("check_fire_dust_inputs.py: only one repo root may be given")
    if any(e is None for e in extra):
        sys.exit("check_fire_dust_inputs.py: --extra-deck needs a path")
    return root, extra

_root, EXTRA_DECKS = _parse_argv(sys.argv[1:])

ROOT = os.path.abspath(_root) if _root else \
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def read(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()

FAMILIES = ("erf.fire.", "erf.dust.", "erf.fire_dust_", "erf.fire_plot_",
            "erf.mrf_fire_", "erf.pbl_mrf_fire_", "erf.dust_mrf_Sc_t")

def in_family(k):
    return any(k.startswith(f) for f in FAMILIES)

def norm(k):
    """firebreak.<n>.x -> firebreak.N.x ; custom_fuel.<code>.x -> custom_fuel.N.x ;
    drop a _lev<N> suffix ; strip trailing dot"""
    k = re.sub(r"firebreak\.\d+\.", "firebreak.N.", k)
    k = re.sub(r"custom_fuel\.\d+\.", "custom_fuel.N.", k)
    k = re.sub(r"_lev\d+$", "", k)
    return k.rstrip(".")

# ---------------------------------------------------------------- code reads
def custom_fuel_keys():
    """erf.fire.custom_fuel.*, read by Source/Fire/ERF_CustomFuel.cpp.

    Only custom_fuel.codes is a string literal there; the per-fuel properties are
    queried through a prefix built at run time from the fuel code, so grepping for
    whole key names finds nothing and every key a FireCustomFuel deck sets would
    look dead.  Take the prefix and the property names from the source instead of
    listing them here, so that renaming a property in the source (without the deck
    or Inputs.rst following) still fails this test.  Both spots are asserted: a
    rewrite that stops matching them stops the test rather than passing it.
    """
    src = os.path.join(ROOT, "Source/Fire/ERF_CustomFuel.cpp")
    s = read(src)
    keys = set()
    for m in re.finditer(r'\bpp\.(?:query|queryarr|contains)\s*\(\s*"([^"]+)"', s):
        keys.add("erf.fire." + m.group(1))

    # const std::string pre = "custom_fuel." + std::to_string(code) + ".";
    m = re.search(r'\bconst\s+std::string\s+pre\s*=\s*"([^"]+)"\s*\+\s*std::to_string', s)
    if m is None:
        sys.exit("check_fire_dust_inputs.py: ERF_CustomFuel.cpp no longer builds its "
                 "per-fuel keys from a \"pre\" prefix; update custom_fuel_keys()")
    pre = "erf.fire." + m.group(1) + "N."
    if norm("erf.fire." + m.group(1) + "1000.probe") != pre + "probe":
        sys.exit("check_fire_dust_inputs.py: norm() does not fold the index out of "
                 "erf.fire." + m.group(1) + "<code>.*; update both together")

    # pp.query((pre + "name").c_str(), ...) and the need() helper, which queries
    # (pre + name) and aborts when the deck leaves it out. contains is a read as
    # well -- it asks whether the deck set the key -- and is accepted by the
    # literal scan above, so the two stay on the same list of call names.
    props = set(re.findall(r'\bpp\.(?:query|queryarr|contains)\s*\(\s*\(\s*pre\s*\+\s*"([^"]+)"', s))
    props |= set(re.findall(r'\bneed\s*\(\s*"([^"]+)"\s*,', s))
    if not props:
        sys.exit("check_fire_dust_inputs.py: found no " + pre + "* property reads in "
                 + os.path.relpath(src, ROOT) + "; update custom_fuel_keys()")
    for name in props:
        keys.add(pre + name)
    return keys

def code_keys():
    keys = set()
    s = read(os.path.join(ROOT, "Source/Fire/ERF_FireParams.H"))
    for m in re.finditer(r'\bpp\.(?:query|queryarr|contains)\s*\(\s*"([^"]+)"', s):
        keys.add("erf.fire." + m.group(1))
    for m in re.finditer(r'\bpp_erf\.(?:query|queryarr)\s*\(\s*"([^"]+)"', s):
        keys.add("erf." + m.group(1))
    if "firebreak." in s:   # built at run time from "firebreak." + n + "." + name
        for name in ("type", "x_lo", "y_lo", "x_hi", "y_hi", "cx", "cy", "radius"):
            keys.add("erf.fire.firebreak.N." + name)
    s = read(os.path.join(ROOT, "Source/Dust/ERF_DustParams.H"))
    for m in re.finditer(r'\bpp\.(?:query|queryarr)\s*\(\s*"([^"]+)"', s):
        keys.add("erf.dust." + m.group(1))
    s = read(os.path.join(ROOT, "Source/ERF.cpp"))
    for m in re.finditer(r'\bpp\.query\s*\(\s*"(fire_[A-Za-z0-9_]+)"', s):
        keys.add("erf." + m.group(1))
    keys |= custom_fuel_keys()
    s = read(os.path.join(ROOT, "Source/DataStructs/ERF_TurbStruct.H"))
    for m in re.finditer(r'query_one_or_per_level\s*\(\s*pp,\s*"([^"]+)"', s):
        k = "erf." + m.group(1)
        if in_family(k):
            keys.add(k)
    return keys

# ---------------------------------------------------------------- docs
KEY_RE = re.compile(r"erf\.(?:fire|dust)[A-Za-z0-9_.]*|erf\.fire_(?:dust|plot)_[A-Za-z0-9_]*|"
                    r"erf\.(?:pbl_)?mrf_fire_[A-Za-z0-9_]*|erf\.dust_mrf_Sc_t")

def doc_keys():
    found = {}
    for path in glob.glob(os.path.join(ROOT, "Docs/sphinx_doc/**/*.rst"), recursive=True):
        s = read(path).replace("\\_", "_")
        for m in KEY_RE.finditer(s):
            k = norm(m.group(0))
            if in_family(k):
                found.setdefault(k, os.path.relpath(path, ROOT))
    return found

def inputs_rst_keys():
    """Keys named in Inputs.rst as a table row (**key**) or in prose (``key``).
    A row may fold sibling keys as key.a/b/c; expand them."""
    s = read(os.path.join(ROOT, "Docs/sphinx_doc/Inputs.rst")).replace("\\_", "_")
    keys = set()
    for m in re.finditer(r"\*\*(erf\.[A-Za-z0-9_./]+)\*\*|``(erf\.[A-Za-z0-9_.]+)``", s):
        k = m.group(1) or m.group(2)
        if "/" in k:
            head, _, tail = k.rpartition(".")
            for alt in tail.split("/"):
                keys.add(norm(head + "." + alt))
        else:
            keys.add(norm(k))
    return keys

# ---------------------------------------------------------------- decks
DECK_DIRS = ["Exec/CanonicalTests/Fire", "Exec/CanonicalTests/Dust", "Exec/CanonicalTests/Hazard"]

def deck_files():
    files = []
    for d in DECK_DIRS:
        files += glob.glob(os.path.join(ROOT, d, "**/inputs*"), recursive=True)
    files += glob.glob(os.path.join(ROOT, "Exec/RegTests/Fire*/inputs*"))
    # The glob results are filtered below, but an --extra-deck was named by hand:
    # dropping a mistyped one would leave the run reporting PASS having scanned one
    # deck fewer than it was asked to.
    for f in EXTRA_DECKS:
        if not os.path.isfile(f):
            sys.exit("check_fire_dust_inputs.py: --extra-deck " + f + " is not a file")
        files.append(os.path.abspath(f))
    return sorted(f for f in files if os.path.isfile(f))

def show(path):
    """A deck's path as the report names it: relative to the root when it is under
    the root, as given for an --extra-deck outside it. Tested for the prefix rather
    than handed to os.path.relpath, which raises ValueError -- not a walk-up path --
    for two paths on different Windows drives, as an --extra-deck in a temporary
    directory may well be."""
    head = os.path.join(ROOT, "")
    return path[len(head):] if path.startswith(head) else path

def deck_keys(path, commented=False):
    pat = r"^\s*#?\s*(erf\.[A-Za-z0-9_.]+)\s*=" if commented else r"^\s*(erf\.[A-Za-z0-9_.]+)\s*="
    return {norm(m.group(1)) for m in re.finditer(pat, read(path), re.M)}

# ---------------------------------------------------------------- dust generator
def generator_keys():
    s = read(os.path.join(ROOT, "Docs/sphinx_doc/tools/gen_dust_inputs.py"))
    keys = set()
    for m in re.finditer(r'^p\("([^"]+)"', s, re.M):
        k = m.group(1)
        keys.add(k if k.startswith("erf.") else "erf.dust." + k)
    return keys

def main():
    code = code_keys()
    docs = doc_keys()
    inputs_rst = inputs_rst_keys()
    problems = []

    # 1. documented but never read (a prefix of a read key is a heading, not a key)
    for k, where in sorted(docs.items()):
        if k in code:
            continue
        if any(c.startswith(k + ".") for c in code):
            continue
        problems.append(f"documented in {where} but read by nothing: {k}")

    # 2. read but not in the Inputs.rst tables
    for k in sorted(code):
        if k not in inputs_rst:
            problems.append(f"read by the code but not in Docs/sphinx_doc/Inputs.rst: {k}")

    # 3. deck keys nothing reads
    decks = deck_files()
    for path in decks:
        for k in sorted(deck_keys(path)):
            if in_family(k) and k not in code:
                problems.append(f"{show(path)} sets a key nothing reads: {k}")

    # 4. the fire master reference lists every fire-side key (active or commented)
    master = os.path.join(ROOT, "Exec/CanonicalTests/Fire/inputs_fire_master_reference")
    listed = deck_keys(master, commented=True)
    for k in sorted(code):
        if (k.startswith("erf.fire.") or k.startswith("erf.fire_")) and k not in listed:
            problems.append(f"fire master reference does not list: {k}")

    # 5. dust generator vs parser vs generated table
    gen = generator_keys()
    dust_code = {k for k in code if k.startswith(("erf.dust.", "erf.fire_dust_")) or k == "erf.dust_mrf_Sc_t"}
    for k in sorted(gen - code):
        problems.append(f"gen_dust_inputs.py lists a key nothing reads: {k}")
    for k in sorted(dust_code - gen):
        problems.append(f"gen_dust_inputs.py is missing a key the code reads: {k}")
    for k in sorted(gen - inputs_rst):
        problems.append(f"Inputs.rst dust table is stale (regenerate with gen_dust_inputs.py): missing {k}")

    print(f"code reads {len(code)} keys; docs mention {len(docs)}; Inputs.rst lists "
          f"{len([k for k in inputs_rst if in_family(k)])}; {len(decks)} decks scanned")
    for p in problems:
        print("FAIL  " + p)
    if problems:
        print(f"{len(problems)} problem(s)")
        return 1
    print("PASS  every documented key is read, every read key is documented, no deck sets an unread key")
    return 0

if __name__ == "__main__":
    sys.exit(main())
