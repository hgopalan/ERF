# SubDyn reference: a small lattice tower

Reference stiffness for ERF's frame model (`Source/MovingBodies/Towers/ERF_Frame.H`), computed by
OpenFAST's SubDyn. The unit tests `FrameSubDyn.*` read these SubDyn input files with ERF's reader and
compare ERF's condensed stiffness at the peak joint with SubDyn's.

## The tower

- 4 legs from the base corners (±3, ±3, 0) m, tapering to (±1, ±1, 21) m.
- Panel levels at z = 0, 7, 14 and 21 m, with horizontal struts at the three upper levels.
- X-bracing in every face of every panel.
- A peak joint at (0, 0, 25) m, joined to the 4 top corners.
- A cross-arm reaching (±7, 0, 21) m.
- Joint and member numbering is in the docstring of `make_cases.py`.

| Case | Beam theory | Elements per member (NDiv) | Sections | Base |
|---|---|---|---|---|
| `caseA` | Euler-Bernoulli (FEMMod 1) | 1 | arbitrary (MType 4), spin 0 | 4 legs fixed |
| `caseB` | Timoshenko (FEMMod 3) | 2 | circular legs (1c), rectangular struts and peak members (1r), arbitrary diagonals spun 30 deg, arbitrary cross-arm | 3 legs fixed; joint 1 free on a 6 x 6 spring (`towerB_SSI_joint1.dat`, with a Kxty coupling) |

Both cases have one interface joint, the peak (joint 17), locked to the transition piece. The
driver's reference point is at that joint.

## The oracle

`tower<case>_kbbt.txt` is the 6 x 6 matrix `KBBt` that SubDyn writes to `tower<case>.SD.sum.yaml`.
It is the stiffness of the whole tower condensed to the peak joint, with the base supports applied.

- DOF order: x, y, z translations, then rotations about x, y, z.
- Units: N/m, N/rad and N m/rad.
- Precision: SubDyn prints 7 significant digits (format ES15.6E2).
- Gravity does not enter `KBBt`: SubDyn's beams have no geometric stiffness.

## Regenerating

Run with OpenFAST 5.0.0 built in double precision:

```bash
python3 make_cases.py
cd caseA && subdyn_driver towerA.dvr && cd ..
cd caseB && subdyn_driver towerB.dvr && cd ..
```

`make_cases.py` rewrites the `.dat`, `.dvr` and SSI files byte for byte. Copy `KBBt` from each
`.SD.sum.yaml` into `tower<case>_kbbt.txt`; the `#` header lines of that file are comments.
