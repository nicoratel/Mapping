# Scattering Data Analysis Toolkit

A comprehensive Python package for analyzing X-ray scattering data (SAXS, WAXS, USAXS) from synchrotron facilities, with particular focus on magnetic field-dependent measurements and azimuthal profile extraction.

## Overview

This toolkit provides a robust framework for processing 2D scattering patterns, extracting azimuthal profiles, fitting orientation distributions, and analyzing structural order parameters. It supports data from multiple beamlines and detector configurations, with specialized features for studying field-aligned systems.

## Features

- **Multi-format support**: HDF5 (ID02 beamline), EDF (LGC instrument)
- **Comprehensive Q-space mapping**: Automatic computation of q-parallel, q-perpendicular, and scattering angles
- **Azimuthal profile extraction**: Extract intensity distributions at constant q-values or crystallographic reflections
- **Advanced fitting routines**:
  - Pseudo-Voigt profile fitting
  - Maier-Saupe distribution fitting for liquid crystal systems
- **Nematic order parameter (S) calculation**
- **Batch processing**: Automated analysis of multiple files with magnetic field or time series
- **Data visualization**: 2D intensity maps, azimuthal profiles, fitting results
- **SasView export**: Convert data to SasView-compatible format

## Installation

### Requirements

```bash
pip install numpy scipy matplotlib h5py hdf5plugin scikit-image pandas fabio pyFAI ase
```

### Dependencies

- **numpy**: Numerical computations
- **scipy**: Interpolation, optimization, integration
- **matplotlib**: Data visualization
- **h5py/hdf5plugin**: HDF5 file handling
- **scikit-image**: Image processing
- **pandas**: Data organization
- **fabio**: EDF file reading
- **pyFAI**: Azimuthal integration
- **ase**: CIF file parsing for crystallography

## Quick Start

```python
from Mapping_V2 import Mapping, BatchAzimProfileExtraction
import numpy as np

# Single file analysis
data = Mapping(
    file='path/to/data.h5',
    instrument='ID02',
    qvalues=np.array([0.05, 0.10, 0.15]),  # For SAXS
    threshold=0.0001,
    binning=2,
    mapping=True
)

# Extract and fit azimuthal profiles
results = data.azim_profile_fit(plotflag=True)

# Batch processing
batch = BatchAzimProfileExtraction(
    path='path/to/data/directory',
    qvalues=np.array([0.05, 0.10, 0.15]),
    instrument='ID02',
    binning=2,
    mapping=True
)

# Fit all files and generate summary plots
df = batch.fit_azimprofiles(plot=True, r2_threshold=0.85)
```

## Core Classes

### `Mapping` Class

The main class for single-file scattering data analysis.

#### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | str | None | Path to data file (.h5 or .edf) |
| `cif_file` | str | None | Path to CIF file for crystallographic analysis |
| `instrument` | str | 'ID02' | Instrument type ('ID02' or 'LGC') |
| `reflections` | np.ndarray | None | Miller indices for diffraction peaks, e.g., `[[1,1,0], [2,0,0]]` |
| `qvalues` | np.ndarray | None | Q-values (Å⁻¹) for SAXS analysis, e.g., `[0.05, 0.10]` |
| `threshold` | float | 0.0001 | Relative tolerance for Q-value matching |
| `binning` | int | 2 | Downsampling factor for images |
| `mask` | str | None | Path to mask file (.edf format) |
| `skipcalib` | bool | False | Skip detector calibration (for image extraction only) |
| `mapping` | bool | False | Enable Q-space mapping calculations |

#### Key Attributes

**Experimental Parameters:**
- `wl`: X-ray wavelength (m)
- `D`: Sample-to-detector distance (m)
- `x_center`, `z_center`: Beam center position (pixels)
- `pixel_size_x`, `pixel_size_z`: Pixel dimensions (m)
- `B`: Applied magnetic field (mT)
- `samplename`: Sample identifier
- `epoch`: Timestamp of measurement

**Data Arrays:**
- `data`: 2D intensity array
- `q_parr`, `q_perp`: Q-parallel and Q-perpendicular components (m⁻¹)
- `qx`, `qy`, `qz`: Cartesian Q-space components (m⁻¹)
- `norm_Q`: Magnitude of scattering vector (m⁻¹)
- `thetaB`: Bragg angle array (degrees)
- `beta`: Complementary angle to Q-parallel (degrees)
- `phi`: Azimuthal angle array (degrees)

**Crystallography (if CIF provided):**
- `lattice_parameters`: Unit cell parameters
- `a`, `b`, `c`: Lattice constants (Å)
- `alpha`, `beta_lattice`, `gamma`: Unit cell angles (degrees)
- `atom_positions`: Fractional atomic coordinates
- `atom_elements`: Chemical symbols

#### Primary Methods

##### Data Visualization

**`plot2D(qcircle=0.068, plotqcircle=False, vmin=1, vmax=5)`**
- Plot 2D data in both pixel space and Q-space
- Optionally overlay circles at constant Q-values
- Returns: None (displays matplotlib figure)

**`plot2d_vsq(prefix='SAXS', qmin=0, qmax=0.1, vmin=-3, vmax=0, qcircles=None, show=False, rotate=False, cmap='jet', time=None)`**
- Create high-quality Q-space maps with logarithmic intensity scaling
- `qcircles`: Tuple of Q-values for overlaying circular guides
- `time`: Optional timestamp for kinetic series
- Returns: Path to saved PNG file

**`plotcomponents()`**
- Visualize Q-parallel, Q-perpendicular, beta angle, and |Q| distributions
- Returns: None (displays matplotlib figures)

**`plot_savedata(path, vmin=0, vmax=5, prefix='WAXS', qvalue=None)`**
- Save 2D intensity map as PNG
- Returns: None (saves file to disk)

##### Azimuthal Profile Extraction

**`compute_datavsbeta(reflection=None, qvalue=None)`**
- Extract intensity vs. angle (beta) at constant Q
- For diffraction: specify `reflection` (e.g., `[1,1,0]`)
- For SAXS: specify `qvalue` (e.g., `0.05`)
- Returns: `(beta_array, intensity_array)` tuple

**`pyFAI_extract_azimprofiles(qvalue)`**
- Use pyFAI library for azimuthal integration
- More robust for masked data
- Returns: `(chi_array, intensity_array)` where chi is azimuthal angle

**`pixelindexes_constantq(reflection=None, qvalue=None)`**
- Find detector pixels at a given Q-value
- Returns: Nx2 array of (row, column) indices

##### Profile Fitting

**`azim_profile_fit(beta_target=None, plotflag=False, printResults=False, method=None, prefix='WAXS', kinetic=False, time=0)`**
- Fit azimuthal profiles with pseudo-Voigt function
- `beta_target`: Center fit around specific angle (degrees)
- `method`: Use 'pyFAI' for pyFAI-extracted profiles
- Returns: Dictionary with fit parameters per reflection/Q-value
  ```python
  {
      'reflection_or_qvalue': [
          y0,        # Background offset
          I,         # Peak intensity
          x0,        # Peak position (degrees)
          x0_S,      # Position for S calculation (-20° to 20°)
          gamma,     # Peak width (degrees)
          eta,       # Lorentzian/Gaussian mixing (0-1)
          slope,     # Background slope
          S,         # Nematic order parameter
          R²         # Coefficient of determination
      ]
  }
  ```

**`azim_profile_fit_MS(beta_target=None, plotflag=False, printResults=False, method=None, prefix='WAXS', kinetic=False, time=0)`**
- Fit with Maier-Saupe distribution (for liquid crystals)
- Better for highly oriented systems
- Returns: Dictionary with format:
  ```python
  {
      'reflection_or_qvalue': [
          I,         # Intensity scaling
          x0,        # Distribution center (degrees)
          x0_S,      # Adjusted center for S calculation
          kappa,     # Maier-Saupe order parameter
          a,         # Background slope
          b,         # Background offset
          S,         # Nematic order parameter
          R²         # Coefficient of determination
      ]
  }
  ```

##### Utility Methods

**`caving(max_iter=10)`**
- Fill masked pixels using symmetry around beam center
- Iterative algorithm for scattered mask regions
- Returns: None (modifies `self.data` in-place)

**`export2sasview()`**
- Export data in SasView format (Qx, Qy, I)
- Saves .dat file in same directory as input
- Returns: None

**`d_hkl(reflection)`, `theta_hkl(reflection)`, `q_hkl(reflection)`**
- Calculate crystallographic parameters from Miller indices
- Requires CIF file to be loaded
- Returns: Float value (d-spacing in Å, angle in radians, Q in m⁻¹)

##### Radial Profile Extraction

**`save_radialprofiles(nb_azim=2, offset=0, delta=90, width=10, prefix='SAXS', plot=True, kinetic=False)`**
- Extract I(Q) along multiple azimuthal sectors
- `nb_azim`: Number of angular directions
- `offset`: Starting angle (degrees, from horizontal)
- `delta`: Angle increment between sectors
- `width`: Angular width of each sector
- Returns: None (saves CSV files)

---

### `BatchAzimProfileExtraction` Class

Automated processing of multiple scattering files.

#### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | str | - | Directory containing data files |
| `cif` | str | None | Path to CIF file |
| `reflections` | np.ndarray | None | Miller indices list |
| `qvalues` | np.ndarray | None | Q-values for SAXS |
| `instrument` | str | 'ID02' | Instrument identifier |
| `file_filter` | str | '*_waxs*_raw.h5' | Glob pattern for file selection |
| `threshold` | float | 0.0001 | Q-value tolerance |
| `binning` | int | 2 | Image downsampling factor |
| `plotflag` | bool | False | Enable automatic plotting |
| `mask` | str | None | Path to mask file |
| `skipcalib` | bool | False | Skip calibration |
| `mapping` | bool | False | Enable Q-space mapping |

#### Key Methods

**`fit_azimprofiles(beta_target=None, plot=True, r2_threshold=0.85, prefix='WAXS', method=None)`**
- Fit azimuthal profiles for all files (pseudo-Voigt)
- `r2_threshold`: Minimum R² for including data in plots
- Returns: pandas DataFrame with all fit results
- Automatically generates:
  - CSV file with fit parameters
  - Summary plots of position, width, and S vs. magnetic field

**`fit_azimprofiles_MS(beta_target=None, plot=True, r2_threshold=0.85, prefix='WAXS', method=None)`**
- Fit with Maier-Saupe distribution for all files
- Same return format as `fit_azimprofiles()`

**`fit_azimprofiles_vs_time(beta_target=None, plot=True, r2_threshold=0.85, prefix='WAXS', method=None)`**
- Time-resolved analysis (uses file timestamps)
- Generates plots vs. time instead of magnetic field
- Returns: pandas DataFrame

**`plot_save_azim_profiles_vs_B(prefix='WAXS', method=None)`**
- Extract and plot azimuthal profiles for all files
- Creates multi-panel figures showing evolution with field
- Saves individual profiles as CSV files

**`plot_save_azim_profiles_vs_time(prefix='WAXS', method=None)`**
- Time-resolved version of above
- Uses elapsed time from first measurement

**`plot_savedata(prefix='SAXS', vmin=-3, vmax=0, qmin=0, qmax=0.15, qcircles=None, cmap='jet', kinetic=False)`**
- Batch export of 2D Q-space maps
- `kinetic`: Use timestamps instead of field values in filenames

**`plot_savedata_zoom(prefix='WAXS', vmin=None, vmax=None, qvalue=None, zoom_factor=4, kinetic=False)`**
- Export zoomed regions around beam center
- Useful for highlighting specific features

**`save_radialprofiles(nb_azim=2, offset=0, delta=90, width=10, prefix='SAXS', plot=True, kinetic=False)`**
- Batch extraction of I(Q) profiles along sectors
- Generates comparison plots for all files

**`build_timearray()`**
- Calculate elapsed time from first measurement
- Populates `self.epoch` array (seconds)

**`extract_titles()`**
- Extract sample names from all files
- Saves to 'Sample_list.txt'

## Advanced Usage Examples

### Diffraction Analysis with Crystallography

```python
# Load data with CIF file for peak indexing
data = Mapping(
    file='sample_waxs.h5',
    cif_file='structure.cif',
    reflections=np.array([[1,1,0], [2,0,0], [2,2,0]]),
    instrument='ID02',
    threshold=0.0005,
    binning=2,
    mapping=True
)

# Calculate d-spacings
for refl in data.reflections:
    d = data.d_hkl(refl)
    q = data.q_hkl(refl) * 1e-10  # Convert to Å⁻¹
    print(f"Reflection {refl}: d = {d:.3f} Å, q = {q:.4f} Å⁻¹")

# Fit azimuthal profiles and calculate order parameters
results = data.azim_profile_fit(plotflag=True)
for refl in data.reflections:
    S = results[str(refl)][7]
    print(f"Reflection {refl}: S = {S:.3f}")
```

### SAXS Analysis with Masking

```python
# Process SAXS data with beam stop mask
data = Mapping(
    file='sample_saxs.h5',
    qvalues=np.array([0.03, 0.05, 0.08, 0.12]),
    instrument='ID02',
    binning=2,
    mask='beamstop_mask.edf',
    mapping=True
)

# Apply caving algorithm to fill masked regions
data.caving(max_iter=15)

# Create publication-quality Q-space map
data.plot2d_vsq(
    qmin=0.02,
    qmax=0.15,
    vmin=-3,
    vmax=0,
    qcircles=(0.05, 0.08, 0.12),
    cmap='viridis',
    show=True
)

# Fit profiles with Maier-Saupe distribution
results = data.azim_profile_fit_MS(plotflag=True, printResults=True)
```

### Magnetic Field Series Analysis

```python
# Batch process field-dependent measurements
batch = BatchAzimProfileExtraction(
    path='/data/magnetic_series/',
    qvalues=np.array([0.05, 0.10]),
    file_filter='*_eiger2_*_raw.h5',
    instrument='ID02',
    binning=2,
    mask='mask.edf',
    mapping=True
)

# Fit all profiles and generate summary
df = batch.fit_azimprofiles_MS(
    plot=True,
    r2_threshold=0.90,
    prefix='SAXS'
)

# Access results
print(df[['B (mT)', 'qvalue', 'Order Parameter S', 'R_squared']])

# Extract radial profiles at multiple angles
batch.save_radialprofiles(
    nb_azim=4,
    offset=0,
    delta=45,
    width=10,
    plot=True
)
```

### Time-Resolved Kinetic Study

```python
# Analyze temporal evolution
batch = BatchAzimProfileExtraction(
    path='/data/kinetics/',
    qvalues=np.array([0.08]),
    instrument='ID02',
    binning=1,
    mapping=True
)

# Fit vs. time
df_time = batch.fit_azimprofiles_vs_time(
    plot=True,
    prefix='SAXS_kinetic'
)

# Generate time-series images
batch.plot_savedata(
    prefix='SAXS',
    vmin=-2,
    vmax=1,
    qmin=0,
    qmax=0.2,
    kinetic=True
)
```

## Output Files

The toolkit generates several types of output files:

### CSV Files

1. **Azimuthal profiles**: `Azimuthal_Profiles/file_XXXXX_sample_BmT_hkl.csv`
   - Columns: beta (degrees), intensity

2. **Fit parameters**: `prefix_azim_profiles_refinements.csv`
   - All fit parameters for batch processing
   - Includes: filename, sample, field, reflection/Q, fit parameters, S, R²

3. **Radial profiles**: `Radial_Profiles_(U)SAXS/sample_BmT_sectorXX.csv`
   - Columns: Q (Å⁻¹), intensity

### Images

1. **2D maps**: `png_images/samplename_BmT_ImgXXXXX.png`
   - Q-space intensity maps

2. **Fit plots**: `Azim_Profile_Fittings/sample_BmT_ImgXXXXX_hkl.png`
   - Experimental data with fit overlay

3. **Summary plots**: `prefix_Fitting_results.png`
   - Multi-panel evolution of fit parameters vs. field/time

### Data Arrays

- **NPZ files**: `png_images/prefix_sample_BmT_ImgXXXXX.npz`
  - Contains: Qx, QZ, Z (intensity) arrays for reprocessing

## Nematic Order Parameter Calculation

The order parameter S quantifies molecular alignment:

### Pseudo-Voigt Method

$$S = \frac{\int_0^\pi P_2(\cos\theta) f(\theta) \sin\theta \, d\theta}{\int_0^\pi f(\theta) \sin\theta \, d\theta}$$

Where $P_2(x) = \frac{1}{2}(3x^2 - 1)$ is the second Legendre polynomial and $f(\theta)$ is the fitted pseudo-Voigt function.

### Maier-Saupe Method

For liquid crystals, the distribution follows:

$$f(\theta) = \frac{1}{Z(\kappa)} \exp(\kappa \cos^2\theta)$$

The order parameter is then:

$$S = \frac{\int_0^\pi P_2(\cos\theta) \exp(\kappa\cos^2\theta) \sin\theta \, d\theta}{\int_0^\pi \exp(\kappa\cos^2\theta) \sin\theta \, d\theta}$$

Both methods center the distribution around 0° before calculation to ensure 0 ≤ S ≤ 1.

## Tips and Best Practices

1. **Memory management**: Use `binning=2` or higher for large datasets
2. **Mask optimization**: Create masks with beam stop and detector gaps for accurate results
3. **Fit quality**: Always check R² values; set `r2_threshold=0.85` or higher
4. **Q-range selection**: Choose Q-values where peaks are well-defined
5. **Caving iterations**: Increase `max_iter` if many pixels remain masked after symmetrization
6. **pyFAI vs. custom**: Use `method='pyFAI'` for complex masks, custom method for speed
7. **Maier-Saupe fitting**: Best for highly aligned systems (S > 0.5)

## Troubleshooting

**Problem**: "Phi values could not be computed"
- **Solution**: Some geometric configurations cause division by zero. Use beta angle instead of phi.

**Problem**: Poor fit quality (low R²)
- **Solution**: Adjust `width` parameter in fitting range or increase data quality (longer exposure)

**Problem**: Negative order parameters
- **Solution**: Check if peak position is correctly centered; may need manual `beta_target`

**Problem**: Memory error with large files
- **Solution**: Increase `binning` parameter or process files individually



