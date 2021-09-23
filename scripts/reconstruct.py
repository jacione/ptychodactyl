"""
Main script for reconstructing ptycho data.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from ptycho.recon import Reconstruction
from ptycho.specs import ReconstructionSpecs


SPECS = ReconstructionSpecs(
    title='test-2021-07-22.pty'
)
SPECS.add_cycle('epie', 10)


def reconstruct(specs):
    recon = Reconstruction(specs)

    for cycle in specs.cycles:
        recon.run_cycle(cycle)

    recon.save_reconstruction()
    recon.show_object_and_probe()


if __name__ == '__main__':
    reconstruct(SPECS)
