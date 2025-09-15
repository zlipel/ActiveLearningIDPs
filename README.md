# ActiveLearningIDPs

Python package providing an active learning pipeline for optimizing protein sequences.

## Installation

Clone the repository and install in editable mode:

```bash
pip install .
```

## Configuration Example

```python
from al_pipeline.features import SequenceFeaturizer

featurizer = SequenceFeaturizer(model_name="mpipi", db_path="/path/to/database")
```

## Slurm Usage

Submit the pipeline as a batch job:

```bash
#!/bin/bash
#SBATCH --job-name=al-pipeline
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00

module load python
srun active-learning-pipeline
```
