#!/usr/bin/env python3
"""Cross-check the fire and dust input keys three ways: what the code reads,
what the documentation lists, and what the decks set.

    python3 Tests/check_fire_dust_inputs.py [repo_root]

Fails (exit 1) when
  * a key documented anywhere under Docs/sphinx_doc is not read by the code
    (a renamed or misspelt key in the docs),
  * a key the code reads is missing from the Inputs.rst tables,
  * a fire, dust or hazard deck sets a key the code does not read (ParmParse
    never warns, so such a key is a silent no-op),
  * the fire master reference deck lacks a key the code reads, or
  * the dust inputs generator and the generated Inputs.rst table disagree with
    the dust parser.
ParmParse reads are collected from the four files that parse these keys; add a
file here if a new one starts reading erf.fire.* or erf.dust.* keys.
"""
import glob
import os
import re
import sys

ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def read(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()

FAMILIES = ("erf.fire.", "erf.dust.", "erf.fire_dust_", "erf.fire_plot_",
            "erf.mrf_fire_", "erf.pbl_mrf_fire_", "erf.dust_mrf_Sc_t")

def in_family(k):
    return any(k.startswith(f) for f in FAMILIES)

def norm(k):
    """firebreak.<digit>.x -> firebreak.N.x ; drop a _lev<N> suffix ; strip trailing dot"""
    k = re.sub(r"firebreak\.\d+\.", "firebreak.N.", k)
    k = re.sub(r"_lev\d+$", "", k)
    return k.rstrip(".")

# ---------------------------------------------------------------- code reads
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
    return sorted(f for f in files if os.path.isfile(f))

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
                problems.append(f"{os.path.relpath(path, ROOT)} sets a key nothing reads: {k}")

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
