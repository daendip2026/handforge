# HandForge-Python

Real-time, extreme low-latency hand pose tracking pipeline using MediaPipe and OpenCV.
*(Currently: Core Engine Implementation)*

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd handforge-python
   ```

2. **Set up a virtual environment:**
   ```bash
   # Windows
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies:**
   Install the package in editable mode with development tools:
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```

## Usage

The tracking pipeline can be executed via the command line interface. It is driven by `config.yaml`.

### 1. Run via Developer Script (Recommended for Dev)
Runs the tracker directly from the source tree.
```bash
python scripts/run_tracker.py
```

### 2. Run as a Module
```bash
python -m hand_tracker
```

### CLI Options
You can control the behavior of the tracker using the following flags:

- `--view` : Enable real-time OpenCV debug visualization (shows the camera feed and landmarks).
- `--full-dump` : Print the coordinates of all 21 landmarks per frame to the console.
- `--log-level [DEBUG|INFO|WARNING|ERROR]` : Override the log level specified in `config.yaml`.
- `--config PATH` : Specify a custom config file path (default: `config.yaml`).

**Example:**
Run the tracker with the debug viewer enabled and detailed logging:
```bash
python scripts/run_tracker.py --view --log-level DEBUG
```

---

## Development Workflow

### Linting & Formatting
We use [Ruff](https://github.com/astral-sh/ruff) for extremely fast linting and code formatting.
```bash
ruff check .
ruff format .
```

### Static Type Checking
We use [Mypy](https://github.com/python/mypy) with strict configuration.
```bash
mypy src tests scripts
```

### Running Tests
Execute the test suite with coverage reporting:
```bash
pytest
```

---

## License
This project is licensed under the MIT License.
