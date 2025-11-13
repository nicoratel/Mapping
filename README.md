
---

# 🧩 `Mapping` Class Documentation

The `Mapping` class is the **core analytical engine** of the `Mapping_V2.py` module.
It manages every step of the scattering data analysis workflow — from file loading to azimuthal integration, model fitting, and result export.

---

## 🧭 Purpose

`Mapping` automates the transformation of raw 2D X-ray scattering images (EDF or HDF5 format) into:

* Azimuthal intensity profiles (`I(χ)`),
* Fitted peak parameters (position, width, amplitude, etc.),
* Orientation and anisotropy maps,
* Standardized data files (CSV, SASView, PNG).

It is particularly suited for **SAXS/WAXS** and **fiber diffraction** mapping experiments, where signal anisotropy and texture are key observables.

---

## ⚙️ Class Initialization

```python
Mapping(
    file: str,
    cif_file: str,
    reflections: np.ndarray,
    qvalues: np.ndarray,
    threshold: float = 0.01,
    binning: int = 1,
    mask: str | None = None,
    skipcalib: bool = False,
    mapping: bool = True,
)
```

### Parameters

| Parameter       | Type              | Description                                                                               |
| --------------- | ----------------- | ----------------------------------------------------------------------------------------- |
| **file**        | `str`             | Path to the input EDF or HDF5 image.                                                      |
| **cif_file**    | `str`             | Path to the CIF file providing crystal structure information.                             |
| **reflections** | `np.ndarray`      | Array of Miller indices for reflections of interest.                                      |
| **qvalues**     | `np.ndarray`      | List of scattering-vector magnitudes at which azimuthal profiles will be extracted.       |
| **threshold**   | `float`, optional | Relative tolerance for matching experimental and theoretical q-values. Default is `0.01`. |
| **binning**     | `int`, optional   | Downsampling factor applied to the detector images. Default is `1`.                       |
| **mask**        | `str`, optional   | Path to a detector mask file compatible with pyFAI.                                       |
| **skipcalib**   | `bool`, optional  | If `True`, skips calibration (useful for visualization or testing).                       |
| **mapping**     | `bool`, optional  | Enables or disables mapping workflow. Default is `True`.                                  |

---

## 🧩 Methods Overview

Below are the main methods of the `Mapping` class with detailed explanations, argument descriptions, and return values.

---

### 🔹 `bin_mask(mask, binning)`

Downsamples a binary mask array by a specified binning factor.

```python
bin_mask(mask: np.ndarray, binning: int) -> np.ndarray
```

**Parameters**

| Name        | Type         | Description                                                   |
| ----------- | ------------ | ------------------------------------------------------------- |
| **mask**    | `np.ndarray` | Binary mask array (1 for valid pixels, 0 for masked regions). |
| **binning** | `int`        | Downsampling factor.                                          |

**Returns**

| Type         | Description                       |
| ------------ | --------------------------------- |
| `np.ndarray` | Binned (downsampled) binary mask. |

**Usage Example**

```python
binned_mask = map_obj.bin_mask(mask, binning=2)
```

---

### 🔹 `extract_azimuthal_profiles()`

Performs azimuthal integration of the 2D scattering image(s) using `pyFAI`.

**Purpose**

* Converts intensity as a function of azimuthal angle `χ` for a given `q` region.
* Generates one azimuthal profile per q-value defined in `qvalues`.

**Returns**

| Type   | Description                                                             |
| ------ | ----------------------------------------------------------------------- |
| `dict` | Dictionary containing azimuthal profiles `{q_value: (chi, intensity)}`. |

**Notes**

* Internally applies the detector mask if provided.
* Automatically handles EDF or HDF5 geometries.

---

### 🔹 `fit_profiles(model='pseudoVoigt')`

Fits the azimuthal intensity profiles using the specified analytical model.

```python
fit_profiles(model: str = "pseudoVoigt") -> dict
```

**Parameters**

| Name      | Type  | Description                                                                      |
| --------- | ----- | -------------------------------------------------------------------------------- |
| **model** | `str` | Fitting model: `"pseudoVoigt"`, `"MaierSaupe"`, `"Gaussian"`, or `"Lorentzian"`. |

**Returns**

| Type   | Description                                                                                      |
| ------ | ------------------------------------------------------------------------------------------------ |
| `dict` | Dictionary of fit parameters for each q-value (e.g., amplitude, center, width, orientation, R²). |

**Notes**

* Fits are performed using SciPy’s `curve_fit` function.
* If `MaierSaupe` is selected, it estimates the orientation distribution function (ODF).

---

### 🔹 `compute_orientation()`

Computes local orientation parameters (main azimuthal direction, FWHM, anisotropy index) based on fitted profiles.

**Returns**

| Type           | Description                                                          |
| -------------- | -------------------------------------------------------------------- |
| `pd.DataFrame` | DataFrame containing orientation metrics per reflection and q-value. |

---

### 🔹 `export2sasview(q, intensity, filename)`

Exports integrated scattering data to a SASView-compatible ASCII file.

```python
export2sasview(q: np.ndarray, intensity: np.ndarray, filename: str) -> None
```

**Parameters**

| Name          | Type         | Description                               |
| ------------- | ------------ | ----------------------------------------- |
| **q**         | `np.ndarray` | 1D array of scattering vector magnitudes. |
| **intensity** | `np.ndarray` | Corresponding intensity values.           |
| **filename**  | `str`        | Output file path (`.dat` or `.txt`).      |

**Output Format**

```
# SASView 1D data
# q [1/Å]   I(q) [a.u.]
0.0010      112.5
0.0012      115.3
...
```

---

### 🔹 `extract_sample_name()`

Extracts the sample name from the file path or metadata.

**Returns**

| Type  | Description                                                   |
| ----- | ------------------------------------------------------------- |
| `str` | Clean sample name (without magnetic field or numeric suffix). |

**Example**

```python
>>> map_obj.extract_sample_name()
"PolymerBlend_A"
```

---

### 🔹 `extract_magnetic_field()`

Extracts the magnetic field value (in Tesla) from the filename or metadata.

**Returns**

| Type    | Description                        |
| ------- | ---------------------------------- |
| `float` | Magnetic field intensity in Tesla. |

**Example**

```python
>>> map_obj.extract_magnetic_field()
0.50
```

---

### 🔹 `save_results(output_dir)`

Saves all computed data (fits, maps, tables, figures) to the specified directory.

```python
save_results(output_dir: str) -> None
```

**Parameters**

| Name           | Type  | Description                             |
| -------------- | ----- | --------------------------------------- |
| **output_dir** | `str` | Directory where all results are stored. |

**Generated Files**

| File Type | Description                      |
| --------- | -------------------------------- |
| `.csv`    | Fitted parameters table          |
| `.png`    | Azimuthal maps and profile plots |
| `.h5`     | Processed datasets (optional)    |

---

## 🧠 Fitting Models Supported

| Model                     | Description                                                | Use Case                                     |
| ------------------------- | ---------------------------------------------------------- | -------------------------------------------- |
| **Pseudo-Voigt**          | Weighted sum of Gaussian and Lorentzian peaks.             | Standard diffraction peaks.                  |
| **Maier–Saupe**           | Orientation distribution function (ODF) for nematic order. | Orientation analysis in anisotropic systems. |
| **Gaussian / Lorentzian** | Simple symmetric profiles.                                 | Broad or isotropic features.                 |

All models return parameters such as **amplitude**, **center (χ₀)**, **width (FWHM)**, and **orientation index**.

---

## 🧩 Example Workflow

```python
from Mapping_V2 import Mapping

# Initialize mapping for a single dataset
map_obj = Mapping(
    file="sample_001.h5",
    cif_file="structure.cif",
    reflections=[[1, 0, 0]],
    qvalues=[1.2, 1.8],
    binning=2,
    mask="detector_mask.edf"
)

# Extract and fit azimuthal profiles
profiles = map_obj.extract_azimuthal_profiles()
fits = map_obj.fit_profiles(model="MaierSaupe")

# Compute orientation parameters and export results
orientation = map_obj.compute_orientation()
map_obj.save_results(output_dir="./results")
```

---

## 🧾 Typical Outputs

After running, the output folder includes:

| File                     | Description                               |
| ------------------------ | ----------------------------------------- |
| `azimuthal_profiles.csv` | Integrated I(χ) profiles for each q-value |
| `fit_parameters.csv`     | Fitted amplitude, width, orientation      |
| `orientation_map.png`    | Visual map of main orientation            |
| `*.dat`                  | SASView-compatible export files           |

---

## 🧩 Summary

The `Mapping` class:

* Provides an end-to-end data reduction pipeline.
* Integrates smoothly with **pyFAI**, **SciPy**, and **pandas**.
* Enables reproducible SAXS/WAXS and texture-mapping analyses.
* Can be combined with `BatchAzimProfileExtraction` for automation.

---

Would you like me to also include **the fitting model equations** (pseudo-Voigt and Maier–Saupe) in LaTeX form inside the README, so users can understand the analytical models behind the fits?


