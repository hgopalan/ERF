# Deck for the bundled OpenFAST stub (Source/MovingBodies/OpenFAST/Stub), as in OpenFAST_ALM_Tower: an
# IEA-15-MW-sized rigid rotor turning at a fixed speed, with a uniform-Ct disk model for the loads, and a
# tower of 8 m diameter whose drag (Cd 1) the stub puts on its tower force nodes.
# It also names an AeroDyn file in OpenFAST's own format, which the stub skips (it reads key = value
# lines only) and ERF's start-up checks read: BEM induction and tower shadow on. With CompAero named, the
# start-up audit treats it as an OpenFAST model and notes that it gives no AirDens (harmless here).
2                        CompAero    - aerodynamic loads from AeroDyn
"aerodyn_bem_shadow.dat" AeroFile    - the AeroDyn file the checks read
dt = 0.01
num_blades = 3
num_blade_nodes = 10
num_tower_nodes = 6
rotor_radius = 120.0
hub_height = 150.0
rotor_speed_rpm = 7.55
ct = 0.8
cp = 0.45
air_density = 1.225
tower_diameter = 8.0
tower_cd = 1.0
