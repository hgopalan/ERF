def emit(csv_path, LAD_bulk):
    with open(csv_path, "w") as fh:
        fh.write("x_m,y_m,tree_id,H_tree_m,H_crown_base_m,"
                 "LAD_bulk,crown_area_frac,Cd_leaf,is_tree\n")
        tid = 1
        for i in range(16):        # 16 UCM cells in x
            x = 12.5 + i * 25.0    # cell-center
            if x >= 200.0:          # left half only
                continue
            for j in range(16):
                y = 12.5 + j * 25.0
                fh.write(f"{x},{y},{tid},15.0,3.0,{LAD_bulk},0.7,0.2,1\n")
                tid += 1

emit("tree_layout.csv",       0.5)
emit("tree_layout_dense.csv", 2.0)
