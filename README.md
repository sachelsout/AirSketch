# AirSketch

Real-time finger-drawing overlay for video calls using a Spatial-Temporal
Transformer for hand landmark sequence prediction.

![AirSketch Banner](demo/airsketch-banner.png)

## What it does

AirSketch lets you draw diagrams mid-air during a live video call using only
your webcam. A Spatial-Temporal Transformer tracks your index fingertip and
overlays your strokes directly onto your video feed in real time - no stylus,
tablet, or depth sensor required.

## Requirements

- Python 3.10+
- macOS 13+ or Ubuntu 22.04 (Windows untested)
- Webcam (720p or higher recommended)

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/sachelsout/airsketch.git
cd airsketch
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate          # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. (HPC only) Install on Zaratan**
```bash
module load python/3.10
conda create -n airsketch python=3.10
conda activate airsketch
pip install -r requirements.txt
```

## Project structure

```
src/                  Model, training, inference, overlay code
data/                 Raw datasets (gitignored) and split index files
notebooks/            Exploration and visualization notebooks
tests/                Pytest unit tests
configs/              Hyperparameter YAML files
demo/                 Demo video and thumbnail
report/               Final report PDF, LaTeX source, figures
```

## Running tests

```bash
pytest tests/
```

## Dataset setup

See [Issue #4](https://github.com/sachelsout/AirSketch/issues/4) for FreiHAND download instructions and
[Issue #7](https://github.com/sachelsout/AirSketch/issues/7) for custom gesture data collection protocol.

## Team

| Name | GitHub |
|------|--------|
| Ayush Sinha LNU | https://github.com/Cheetahzzzz1 |
| Pravalika Sure | https://github.com/Pravalikasure11 |
| Rohan Dawkhar | https://github.com/sachelsout |
| Siddhardh Chochipatla | https://github.com/Snch0718 |

## License

MIT