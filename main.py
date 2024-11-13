import ipsuite as ips
import numpy as np
from ipsuite.calculators import OrcaSinglePoint
from src import ShuffleAndSelect, FixEnergy
from apax.nodes import Apax, ApaxBatchPrediction, ApaxJaxMD

project = ips.Project(remove_existing_graph=True, automatic_node_names=True)
with project.group("DataGeneration"):
    # Load entire MLIP-MD trajectory
    trajectory = ips.data_loading.AddDataH5MD(file="data/traditional_md.h5")

    # Select Samples to recalculate using DFT
    dataset = ips.configuration_selection.RandomSelection(
        data=trajectory, n_configurations=500
    )

    fix = FixEnergy(data=dataset.atoms)

    # Initiliaze Calculator & Recalculate
    dft = OrcaSinglePoint(
        data=fix.frames,
        orcasimpleinput="PBE def2-TZVP TightSCF EnGrad",
        orcablocks="%pal nprocs 8 end",
        ASE_ORCA_COMMAND="/data/fzills/tools/orca_5_0_4/orca",
    )

    # Divide Dataset into Train, Test and Validate data
    split_data = ShuffleAndSelect(
      data=dft.atoms, n_train=200, n_test=100, n_validate=200
    )

with project.group("PreTraining"):
    model = Apax(
        data=split_data.train_frames,
        validation_data=split_data.validate_frames,
        config="configs/train.yaml",
    )
    prediction = ApaxBatchPrediction(
        data=split_data.test_frames, model=model, batch_size=10
    )
    metrics = ips.nodes.PredictionMetrics(split_data.test_frames, prediction.atoms)

# Random Transfer Learining (RTL)
with project.group("RTL", "DataGeneration"):
    new_dataset = ips.configuration_selection.RandomSelection(
        data=dataset.excluded_atoms, n_configurations=150
    )

    fix = FixEnergy(data=new_dataset.atoms)
    dft = OrcaSinglePoint(
        data=fix.frames,
        orcasimpleinput="PBE def2-TZVP TightSCF EnGrad",
        orcablocks="%pal nprocs 8 end",
        ASE_ORCA_COMMAND="/data/fzills/tools/orca_5_0_4/orca",
    )
    split_data = ShuffleAndSelect(
      data=dft.atoms, n_train=20, n_test=100, n_validate=20
    )

with project.group("RTL", "TransferLearning"):
    random_transfer = Apax(
        model = model,
        data=split_data.train_frames,
        validation_data=split_data.validate_frames,
        config="configs/transfer.yaml",
    )
    prediction = ApaxBatchPrediction(
        data=split_data.test_frames, model=random_transfer, batch_size=10
    )
    metrics = ips.nodes.PredictionMetrics(split_data.test_frames, prediction.atoms)


"""with project.group("MD"):
    md = ApaxJaxMD(
        data = split_data.train_frames, # Random Frame from Trajectory
        data_id = -1,
        model = model,
        config = "configs/md.yaml",
    )"""


project.build()
