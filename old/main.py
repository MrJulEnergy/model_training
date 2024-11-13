import ipsuite as ips
import numpy as np
from ipsuite.calculators import OrcaSinglePoint
from src import RecalculateData, CombineRecalculations

# Initiliaze Calculator
dft = OrcaSinglePoint(
    data=None,
    orcasimpleinput="PBE def2-TZVP TightSCF EnGrad",
    orcablocks="%pal nprocs 2 end",
    ASE_ORCA_COMMAND="/data/fzills/tools/orca_5_0_4/orca",
)


project = ips.Project(remove_existing_graph=True, automatic_node_names=True)
with project.group("DataGeneration"):
    n_data = 4
    n_batches = 2
    assert n_data % n_batches == 0
    # Load MLIP-MD traj
    trajectory = ips.data_loading.AddDataH5MD(file="data/traditional_md.h5")

    # Select Samples to recalculate
    dataset = ips.configuration_selection.RandomSelection(
        data=trajectory, n_configurations=n_data
    )

    # split into batches to reduce node amounts
    batches = []
    batch_size = n_data // n_batches
    for i in range(n_batches):
        idx1 = i * batch_size
        idx2 = idx1 + batch_size - 1
        batches.append(
            ips.configuration_selection.IndexSelection(
                data=dataset.atoms, indices=[i for i in range(idx1, idx2 + 1)]
            )
        )
        print([i for i in range(idx1, idx2 + 1)])
    # Recalculate Selected Batches using DFT
    recalculations = []
    for i in range(n_batches):
        recalculations.append(
            RecalculateData(
                data=batches[i].atoms,
                data_id=i,
                calc=dft,
            )
        )
    combine = sum([recalculations[i].frames for i in range(n_batches)], [])

project.build()
