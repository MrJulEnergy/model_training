import ase.calculators
import zntrack
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import pathlib
import znh5md
from ipsuite.utils.ase_sim import freeze_copy_atoms
import typing
import h5py


class RecalculateData(zntrack.Node):
    data: list[ase.Atoms] = zntrack.deps()
    data_id: int = zntrack.params()
    calc = zntrack.deps()
    frames_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "frames.h5")

    def run(self):
        self.atoms = self.data[self.data_id]
        self.atoms.calc = self.calc.get_calculator()
        self.atoms.get_potential_energy()
        io = znh5md.IO(self.frames_path)
        io.append(self.atoms)

    @property
    def frames(self) -> typing.List[ase.Atoms]:
        with self.state.fs.open(self.frames_path, "rb") as f:
            with h5py.File(f) as file:
                return znh5md.IO(file_handle=file)[:]


class CombineRecalculations(zntrack.Node):
    data: list[RecalculateData] = zntrack.deps()
    frames_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "frames.h5")

    def run(self):
        io = znh5md.IO(self.frames_path)
        self.atoms = []
        n_calculations = len(self.data)
        for i in range(n_calculations):
            self.atoms.append(self.data[i].atoms)
        io.extend(self.atoms)
