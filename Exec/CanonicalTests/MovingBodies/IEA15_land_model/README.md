# IEA 15 MW reference turbine, land-based, for the real-OpenFAST tests

The OpenFAST model of the IEA Wind 15 MW reference turbine that the real-OpenFAST
regression cases (`OpenFAST_ADM_Askervein`, `OpenFAST_ALM_Askervein`) run. The test
copies this whole directory next to each deck, and the decks name
`IEA15_land_model/IEA15-land.fst` or `IEA15_land_model/IEA15-land-bem.fst`.

## Source and licence

From the IEA Wind Task 37 repository, https://github.com/IEAWindTask37/IEA-15-240-RWT,
commit `e4993d63de10f165389534461dd544006750fe60`: the monopile model in
`OpenFAST/IEA-15-240-RWT-Monopile` and the shared blade, inflow and airfoil files in
`OpenFAST/IEA-15-240-RWT`. Apache License 2.0; `LICENSE` is the source's licence file,
unchanged. Every file changed from the source says so on its second (title) line.

## Changes from the source

Unchanged: the 100 airfoil files (`IEA-15-240-RWT/Airfoils`, 50 polars and their 50 coordinate
files) and the tower file (`IEA-15-240-RWT-Monopile_ElastoDyn_tower.dat`). The others, file by file:

- **`IEA15-land.fst`** (from `IEA-15-240-RWT-Monopile.fst`):
  - OpenFAST 5.0.0 keys added with their default values: `ModCoupling = 1`, `RhoInf`, `ConvTol`,
    `MaxConvIter`, `NRotors = 1`, `CompSoil = 0`, `MirrorRotor`, `SoilFile`.
  - Flow from ERF through OpenFAST's external-inflow interface: `CompInflow` 1 -> 2.
  - No controller: `CompServo` 1 -> 0, so no ServoDyn or controller files are needed.
  - On land: `CompSeaSt`, `CompHydro`, `CompSub` 1 -> 0.
  - `TMax` 300 -> 240 (ERF sets the stop time), `OutFileFmt` 2 -> 1 (text output).
  - The module file names point at the files of this directory. `BDBldFile(1..3)`, `ServoFile`,
    `SeaStFile`, `HydroFile` and `SubFile` still name files of the source that are not shipped: with
    `CompElast = 1` and those modules off, OpenFAST does not open them.
- **`IEA15-land-bem.fst`**: the same, with `AeroFile = IEA15-land_AeroDyn15_bem.dat`.
- **`IEA15-land_ElastoDyn.dat`** (from `IEA-15-240-RWT-Monopile_ElastoDyn.dat`):
  - Fixed rotor speed (7.55 rpm, pitch 0), fixed yaw and a fixed base: `GenDOF`, `YawDOF` and the
    six platform DOFs (`PtfmSgDOF` ... `PtfmYDOF`) True -> False. The blade flap and edge DOFs and the
    tower's fore-aft and side-to-side DOFs stay on, as in the source, so the tower top moves: the
    rotor speed ERF logs (the hub's angular velocity along the shaft) varies by up to 0.7 % about
    7.55 rpm in the tests, and the shaft tilt by up to 0.7 degree.
  - OpenFAST 5.0.0 keys added: `PitchDOF = False`, `PtfmRefxt`, `PtfmRefyt`, `PBrIner(1..3)`,
    `BlPIner(1..3)`.
  - The tower base stays 15 m above the ground (`TowerBsHt = PtfmRefzt = PtfmCMzt = 15`, the
    monopile's transition piece), so with `base_pos` z = 0 the tower starts 15 m above the
    terrain and the hub stands 150 m above it; AeroDyn's tower table also starts at 15 m.
  - File names point at this directory.
- **`IEA15-land_AeroDyn15_bem.dat`** (from `IEA-15-240-RWT-Monopile_AeroDyn15.dat`): `Wake_Mod = 1`
  (OpenFAST's BEM induction) as in the source; `Buoyancy` removed (not an AeroDyn 5.0.0 key); the
  tower table in the AeroDyn 5.0.0 layout, with the `TwrCp` and `TwrCa` columns added as zeros (used
  only for marine turbines) and the same stations, diameters and drag coefficients; airfoil and
  blade file names point at this directory. For `sampling = disk_corrected` or
  `upstream`, which hand OpenFAST the free stream.
- **`IEA15-land_AeroDyn15.dat`**: the same, with `Wake_Mod` 1 -> 0 (no induction: the actuator
  line and the plain disk sample a flow that already carries it) and `TwrPotent`, `TwrShadow`
  1 -> 0 (a forced tower puts its own wake into the flow).
- **`IEA15-land_InflowFile.dat`** (from `IEA-15-240-RWT_InflowFile.dat`): the steady wind of the
  standalone runs (`HWindSpeed` 10 -> 10.59, `PLexp` 0.12 -> 0). Named by the primary files but not read,
  since `CompInflow = 2`.
- **`IEA-15-240-RWT/IEA-15-240-RWT_AeroDyn15_blade.dat`** and
  **`IEA-15-240-RWT/IEA-15-240-RWT_ElastoDyn_blade.dat`**: rewritten in the OpenFAST 5.0.0 table
  layout with the same stations and properties (the AeroDyn table gains the `t_c`, `BlCpn`,
  `BlCpt`, `BlCan`, `BlCat`, `BlCam` columns as zeros; the ElastoDyn table drops `PitchAxis`).

Air density stays 1.225 kg/m^3 (`AirDens` in the primary files), which the decks' `prob.rho_0`
and `air_density` equal and the start-up audit compares.
