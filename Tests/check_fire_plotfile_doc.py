#!/usr/bin/env python3
"""The fire plotfile table in fire_output.rst against the plotfile catalog.

The docs table says its fields appear "in this fixed order", so a reader can
take a field's component index from it. This check reads the order the
catalog (fire_plotfile_var_names in ERF_FirePlotfileCatalog.H) builds, always
present fields first and then each optional block in its push_back order, and
requires the table to list exactly those fields in the same order.

    check_fire_plotfile_doc.py --catalog ERF_FirePlotfileCatalog.H --doc fire_output.rst
    check_fire_plotfile_doc.py --self-test
"""
import argparse
import re
import sys


def catalog_order(text):
    """Field names in the order fire_plotfile_var_names returns them."""
    start = text.index("amrex::Vector<std::string> names{")
    end = text.index("return names", start)
    return re.findall(r'"(fire_[A-Za-z0-9_]+)"', text[start:end])


def doc_order(text):
    """Field names in the order the plotfile table lists them."""
    head = text.index("Variables appear in this fixed")
    table = text.index(".. list-table::", head)
    end = text.index("\n\n", text.index("   * -", table))
    # a row starts with "   * - "; its first cell holds one or more ``names``
    names = []
    for row in re.findall(r"\n   \* - (.*)", text[table:end]):
        names += re.findall(r"``(fire_[A-Za-z0-9_]+)``", row)
    return names


def compare(cat, doc):
    """Messages for every difference; empty when the table matches."""
    msgs = []
    missing = [n for n in cat if n not in doc]
    extra = [n for n in doc if n not in cat]
    if missing:
        msgs.append(f"in the catalog but not in the table: {missing}")
    if extra:
        msgs.append(f"in the table but not in the catalog: {extra}")
    if not missing and not extra:
        for k, (c, d) in enumerate(zip(cat, doc)):
            if c != d:
                msgs.append(f"component {k}: the catalog has {c}, the table {d}")
                break
    return msgs


def self_test():
    cat = ["fire_phi", "fire_ros", "fire_arrival_time", "fire_heat_release"]
    ok = not compare(cat, list(cat))
    swapped = compare(cat, ["fire_phi", "fire_ros", "fire_heat_release", "fire_arrival_time"])
    dropped = compare(cat, cat[:-1])
    good = ok and bool(swapped) and "component 2" in swapped[0] and bool(dropped) and "not in the table" in dropped[0]
    print(f"self-test {'PASS' if good else 'FAIL'}: match passes {ok}, a swap fails {bool(swapped)},"
          f" a missing field fails {bool(dropped)}")
    return good


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog")
    ap.add_argument("--doc")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return 0 if self_test() else 1
    if not (a.catalog and a.doc):
        ap.error("--catalog and --doc are required without --self-test")
    cat = catalog_order(open(a.catalog).read())
    doc = doc_order(open(a.doc).read())
    msgs = compare(cat, doc)
    for m in msgs:
        print("FAIL:", m)
    print(f"{len(cat)} catalog fields, {len(doc)} table fields:", "PASS" if not msgs else "FAIL")
    return 0 if not msgs else 1


if __name__ == "__main__":
    sys.exit(main())
