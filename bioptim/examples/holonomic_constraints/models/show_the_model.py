"""
This file is to display the human model into bioviz
"""
import os
import bioviz


biorbd_viz = bioviz.Viz(
    "three_bar.bioMod",
    show_gravity_vector=True,
    show_floor=False,
    show_local_ref_frame=True,
    show_global_ref_frame=True,
    show_markers=True,
    show_mass_center=True,
    show_global_center_of_mass=False,
    show_segments_center_of_mass=False,
    mesh_opacity=1,
    show_contacts=False,
    background_color=(1, 1, 1),
)

biorbd_viz.exec()
print("Done")
