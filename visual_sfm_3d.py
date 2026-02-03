import pycolmap
from hloc.utils import viz_3d
from pathlib import Path

def visualize_sfm_3d(sfm_dir: Path, scene: str, html_dir: Path):
    if not sfm_dir.exists():
        print(f"Error: Directory {str(sfm_dir)} does not exist!")
    else:
        reconstruction = pycolmap.Reconstruction(sfm_dir)
        ply_path = sfm_dir / f"reconstruction_{scene}.ply"

        if not ply_path.exists():
            print(f"Exporting PLY model for scene {scene}...")
            reconstruction.export_PLY(str(ply_path))
            print(f"PLY model saved to {ply_path}.")
        else:
            print(f"PLY model for scene {scene} already exists.")

    fig = viz_3d.init_figure()
    viz_3d.plot_reconstruction(fig, reconstruction, color='rgba(255,0,0,0.5)', name=f"triangulation of scene {scene}")
    # fig.show()

    html_path = html_dir / f"viz_{scene}_0.1_0.5.html"
    fig.write_html(str(html_path))
    print(f"HTML for 3D visualization saved to {html_path}.")