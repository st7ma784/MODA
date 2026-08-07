# Installation

## Requirements

MATLAB R2017a or higher is required, but newer versions are recommended.

The following MATLAB toolboxes are needed:

- Signal Processing Toolbox
- Statistics and Machine Learning Toolbox
- Wavelet Toolbox

You can check which toolboxes are currently installed by running the `ver` command in
the MATLAB Command Window.

## Downloading MODA

**As a user**, download a zip of the code:

- [Click here](https://github.com/luphysics/MODA/zipball/master) to download the code
  as a `.zip` file.
- Extract the zip file to a desired location.
- For simplicity of instructions, rename the folder to `MODA`.

**As a developer**, clone the repository instead so you can commit changes:

```bash
git clone https://github.com/luphysics/MODA.git
```

This downloads the code into a folder named `MODA`. See the
[Developer Guide](../developer-guide/index.md) for git hook setup and contribution
workflow.

## FastMODA (web app) requirements

FastMODA is a separate Python/Flask application under `FastMODA/`. It requires:

- Python 3.9+
- The packages listed in `FastMODA/requirements.txt`
- (Optional, for GPU-accelerated coherence/bispectrum/Bayesian analysis) a CUDA-capable
  GPU with PyTorch installed

Install and run it with:

```bash
cd FastMODA
pip install -r requirements.txt
python app.py
```

By default it serves at `http://localhost:5000`. See
[Using MODA → The Web App](../using-moda/web-app.md) for a tour of the interface, and
[API & Machine Learning](../api-and-ml/rest-api-reference.md) for the REST API.

## Next step

Continue to the [Quickstart](quickstart.md) to load your first signal and run an
analysis.
