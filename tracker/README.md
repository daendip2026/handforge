# HandForge-Python

Real-time, extreme low-latency hand pose tracking pipeline using MediaPipe and OpenCV.
*(Currently: Core Engine Implementation)*

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd tracker/
   ```

2. **Install dependencies using `uv`:**
   This project uses `uv` for fast dependency management. Running `uv sync` will automatically create a virtual environment (`.venv`) and install all required packages (including development tools) from `uv.lock`.
   ```bash
   uv sync --all-extras
   ```

3. **Activate the environment and set up pre-commit:**
   ```bash
   # Windows
   source .venv/Scripts/activate
   # macOS/Linux: source .venv/bin/activate

   pre-commit install
   ```

4. **Download the MediaPipe Model:**
   Before running the tracker, you must download the HandLandmarker model file:
   ```bash
   uv run python scripts/download_models.py
   ```

## Usage

The tracking pipeline can be executed via the command line interface. It is driven by `config.yaml`.

### 1. Run via Developer Script (Recommended for Dev)
Runs the tracker directly from the source tree.
```bash
uv run python scripts/run_tracker.py
```

### 2. Run as a Module
```bash
uv run python -m hand_tracker
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
uv run python scripts/run_tracker.py --view --log-level DEBUG
```

---

## Development Workflow

### Linting & Formatting
We use [Ruff](https://github.com/astral-sh/ruff) for extremely fast linting and code formatting.
```bash
uv run ruff check .
uv run ruff format .
```

### Static Type Checking
We use [Mypy](https://github.com/python/mypy) with strict configuration.
```bash
uv run mypy src tests scripts
```

### Running Tests
Execute the test suite with coverage reporting:
```bash
uv run pytest
```

---

## License
This project is licensed under the MIT License.
