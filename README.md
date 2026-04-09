# HandForge-Python

Real-time hand pose tracking pipeline: **MediaPipe** → **Protobuf/WebSocket** → **Unity VRM**.

This repository contains the Python-based hand tracking engine designed to process camera input and stream optimized pose data to a Unity-based VRM avatar.

## Tech Stack

- **Inference:** MediaPipe, OpenCV, NumPy
- **Communication:** Python-OSC, WebSockets, Protobuf
- **Configuration:** Pydantic (Settings), PyYAML
- **Tooling:** Ruff (Linting/Formatting), Mypy (Type Checking), Pytest

---

## Getting Started

### Prerequisites
- **Python 3.11** or higher.

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd handforge-python
   ```

2. **Set up a virtual environment:**
   ```bash
   # Windows (using Python Launcher)
   py -3.11 -m venv .venv
   .venv\Scripts\activate

   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   Install the package in editable mode with development tools:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Initialize Git hooks:**
   Set up `pre-commit` to ensure code quality on every commit:
   ```bash
   pre-commit install
   ```

---

## Development Workflow

To maintain high code quality, we enforce strict linting and type checking.

### Linting & Formatting
We use [Ruff](https://github.com/astral-sh/ruff) for extremely fast linting and code formatting.
```bash
# Check for issues
ruff check .

# Format code
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
