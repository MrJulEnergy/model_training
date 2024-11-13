import zntrack
import ase
from ase.calculators.singlepoint import SinglePointCalculator
import h5py
import numpy as np
import tqdm
import functools
from ipsuite import fields
import pint
import random
import pathlib
import znh5md
import typing


class ShuffleAndSelect(zntrack.Node):
    data: list[ase.Atoms] = zntrack.deps()
    frames: list[ase.Atoms] = fields.Atoms()

    n_train: int = zntrack.params()
    n_test: int = zntrack.params()
    n_validate: int = zntrack.params()
    seed: int = zntrack.params(1234)

    def run(self):
        random.seed(self.seed)
        random.shuffle(self.data)
        self.frames = self.data

    @functools.cached_property
    def train_frames(self):
        return self.frames[0 : self.n_train]

    @functools.cached_property
    def test_frames(self):
        return self.frames[self.n_train : self.n_train + self.n_test]

    @functools.cached_property
    def validate_frames(self):
        return self.frames[
            self.n_train + self.n_test : self.n_train + self.n_test + self.n_validate
        ]


class FixEnergy(zntrack.Node):
    data: list[ase.Atoms] = zntrack.deps()
    frames_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "frames.h5")

    def run(self):
        io = znh5md.IO(self.frames_path)
        for atoms in self.data:
            atoms.calc.results["energy"] = atoms.calc.results["energy"][0]
            io.append(atoms)

    @property
    def frames(self) -> typing.List[ase.Atoms]:
        with self.state.fs.open(self.frames_path, "rb") as f:
            with h5py.File(f) as file:
                return znh5md.IO(file_handle=file)[:]
