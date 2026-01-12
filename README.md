# A Multimodal Biomechanics Dataset with Synchronized Kinematics and Internal Tissue Motions during Reaching
This repository provides lightweight tutorials and reference code to **download**, **unpack**, **load**, **visualize**, and **reproduce derived metrics** for the dataset:
**A Multimodal Biomechanics Dataset with Synchronized Kinematics and Internal Tissue Motions during Reaching**

For a detailed description of the dataset please visit the dataset descriptor paper: [].

The dataset files themselves (HDF5 + ultrasound videos + metadata) are hosted on figshare and are not stored in this GitHub repository.

---

## Dataset (figshare)

- **Dataset landing page / DOI:** [https://doi.org/10.6084/m9.figshare.31030252](https://doi.org/10.6084/m9.figshare.31030252)

---

The figshare deposit contains:

- `dataset.csv` — participant-level metadata table (one row per participant) with paths to the corresponding data files  
- `hdf5_files/` — 36 (one per participant) `.h5` files containing synchronized timeseries, event annotations, derived signals, and metadata  
- `us_videos/` — 36 (one per participant) ultrasound `.mp4` videos (60 fps)  
- `exceptions.txt` — subject-specific notes/exceptions (missing channels, irregular segments, etc.)  
- `hdf5_structure.txt` — overview of the HDF5 internal layout (groups/datasets/attributes)  
- `SHA256SUMS.txt` — checksum manifest (optional integrity verification)  
- `Dataset.zip` — archive containing the data

---

## Repository contents

```
reaching-dataset/
├── python/
│   ├── hdf5_tutorial.ipynb          # Python/Jupyter tutorial to load and visualize the data
│   └── process_deriveddata.py       # Script to compute derived metrics from original recordings
├── matlab/
│   └── hdf5_tutorial.mlx            # MATLAB live script tutorial to load and visualize the data
├── scripts/
│   ├── download_dataset.ps1         # Windows download/unzip scripts
│   └── download_dataset.sh          # Unix/Linux/macOS download/unzip scripts
├── LICENSE
└── README.md
```

---

## Quick start

### 1) Download and unzip the dataset

#### Option A: Manual download

Download the dataset from figshare and unzip it to a local folder.

#### Option B: Using download scripts

This repository includes helper scripts to download and unzip the dataset:

**Windows:**

```powershell
# If you encounter execution policy errors, run this first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then run the download script:
.\scripts\download_dataset.ps1 -DestPath "C:\path\to\dataset\folder"
```

**Unix/Linux/macOS:**

```bash
./scripts/download_dataset.sh /path/to/dataset/folder
```

After extraction, your dataset folder should contain:

- `dataset.csv`
- `hdf5_files/`
- `us_videos/`
- `exceptions.txt`
- `hdf5_structure.txt`

> Tip: Start with `dataset.csv` — it is the recommended "index" file (one row per participant) and contains paths to each participant's `.h5` and `.mp4`.

### 2) Tutorials

#### MATLAB tutorial

**Requirements:** MATLAB (Live Script support)

Steps:

1. Open MATLAB
2. Set MATLAB working directory to the dataset folder (the folder containing `dataset.csv`)
3. Open and run: `matlab/hdf5_tutorial.mlx`

#### Python tutorial

**Requirements:** Miniconda (recommended), Python 3.10+

Create environment and install dependencies:

```bash
conda create -n <enviroment_name> python=3.10 -y
conda activate <enviroment_name>
pip install opencv-python numpy matplotlib h5py jupyterlab ipympl pysampled scikit-learn
```

Open the tutorial:

```bash
jupyter lab python/hdf5_tutorial.ipynb
```

### 3) Reproduce derived metrics

To reproduce the derived metrics (tremor detection, arm speed, EMG envelopes, etc.) from the original recordings:

```bash
python python/process_deriveddata.py --input /path/to/dataset/hdf5_folder/subject_id.h5 --tremor_location <triceps, biceps or palm>
```

---

## Notes

Please review `exceptions.txt` before running analyses across all participants. This file documents subject-specific notes, missing channels, and other important exceptions.

The HDF5 internal structure (fields, units, and array shapes) is documented in `hdf5_structure.txt` and in the Supplementary Materials of the dataset descriptor paper.

---

## Citation

If you use this dataset in your research, please cite:

**Plain text format:**
> [Authors]. (2025). A Multimodal Biomechanics Dataset with Synchronized Kinematics and Internal Tissue Motions during Reaching. *[Journal]*. https://doi.org/10.6084/m9.figshare.31030252

**BibTeX format:**
```bibtex
@article{reaching_dataset_2026,
  title={A Multimodal Biomechanics Dataset with Synchronized Kinematics and Internal Tissue Motions during Reaching},
  author={[Authors to be added]},
  journal={[Journal to be added]},
  year={2026},
  doi={10.6084/m9.figshare.31030252}
}
```

---

## Contact

For questions, issues, or feedback regarding this dataset:

- **GitHub Issues:** [https://github.com/RogerPallares/reaching-dataset/issues](https://github.com/RogerPallares/reaching-dataset/issues)
- **Email:** roger31@mit.edu

We welcome suggestions for improving the tutorials and documentation.
