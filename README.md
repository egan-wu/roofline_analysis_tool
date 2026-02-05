# Roofline Analysis Tool

A powerful, interactive visualization tool built with **Dash** and **Plotly** to analyze NPU/Hardware performance using the [Roofline Model](https://en.wikipedia.org/wiki/Roofline_model). This tool helps identify whether operations are **Compute Bound** or **Memory Bound** and diagnoses efficiency issues.

## 🚀 Features

- **Interactive Roofline Chart**: Zoom, pan, and hover over operations to see detailed metrics.
- **Real-time Configuration**: Adjust hardware parameters (Peak TOPS, Memory Bandwidth) and efficiency thresholds on the fly.
- **Plugin System**: extensible operation support via `op_registry.py`.
- **Diagnostic Panel**: Click on any node to view detailed statistics (MACs, Bytes, Utilization, Bandwidth) with clear definitions.
- **Category Filter**: Filter operations by efficiency status (Compute Bound, Memory Inefficient, etc.).
- **Auto-Reload**: Automatically detects and reloads changes in `commands.json`.
- **Config Persistence**: Saves your hardware settings to `roofline_config.json`.

## 📂 File Structure

- `roofline_analysis_tool.py`: The main application script (Dash app).
- `commands.json`: Input data file containing the list of operations to analyze.
- `op_registry.py`: External plugin file for defining custom metrics (MACs, Bytes) and display symbols.
- `roofline_config.json`: user configuration file for hardware specs (auto-generated).

## 🛠️ Installation & Requirements

If running from source, ensure you have Python installed along with the required dependencies:

```bash
pip install dash pandas plotly numpy
```

## 📖 Usage

### Option 1: Running from Source
Run the script directly with Python:

```bash
python roofline_analysis_tool.py
```
The tool will automatically open in your default browser at `http://127.0.0.1:8050`.

## 📝 Input Format (commands.json)

The tool reads a JSON file (`commands.json`) containing a list of operations. Each operation acts as a data point on the roofline chart.

**Example Structure:**

```json
[
    {
        "type": "ID_CONV",
        "name": "#1",
        "data_type": "INT8",
        "input_shape": [1, 64, 1, 224, 224],
        "output_channels": 128,
        "kernel_size": [1, 3, 3],
        "stride": [1, 2, 2],
        "padding": [0, 1, 1],
        "time_ns": 850000.0
    },
    {
        "type": "ID_MUL",
        "name": "#2",
        "data_type": "FP32",
        "input_self": [1, 1, 1, 1, 2048],
        "input_other": [1, 1, 1, 1, 2048],
        "time_ns": 2100.0
    }
]
```

### Supported Types:
- **Built-in**: `ID_CONV` (Convolution), `ID_MM` (Matrix Mult), etc. can be customized in `op_registry.py`.
- **Custom**: Any type (e.g., `ID_MUL`, `ID_ADDMM`) can be supported by adding logic to `get_custom_metrics` in `op_registry.py`.

## 🔌 Plugin System (`op_registry.py`)

The tool supports a flexible plugin system to define how MACs and Bytes are calculated for different operation types.

**How to extend:**
1. Open `op_registry.py`.
2. Add a new conditions in `get_custom_metrics(cmd, dt_size)`.
3. Return `(macs, bytes, symbol)` where symbol can be one of:
    - `"circle"`, `"square"`, `"diamond"`, `"cross"`, `"x"`, `"triangle-up"`, etc. (Plotly symbols)

**Example:**

```python
elif ctype == "ID_MY_OP":
    # Custom logic to calculate macs and bytes
    macs = ...
    bytes_moved = ...
    return macs, bytes_moved, "diamond"
```

## ⚙️ Configuration

The tool allows you to configure the following parameters via the UI (or `roofline_config.json`):

- **Peak TOPS**: Theoretical peak performance of the NPU (in Tera Operations Per Second).
- **Mem BW (GB/s)**: Theoretical maximum memory bandwidth.
- **Memory Inefficient Threshold (%)**: Custom threshold to flag operations that are not fully utilizing memory bandwidth.
- **Compute Inefficient Threshold (%)**: Custom threshold to flag operations that are not fully utilizing compute units.

## 📊 Interpretation

- **Green Nodes**: **Bound** (Good). The operation is hitting the hardware limits (either Compute or Memory).
- **Red Nodes**: **Inefficient** (Bad). The operation is significantly below the roofline boundary, indicating software or architectural bottlenecks.
- **Blue Nodes**: **Normal**. Operations falling between the efficiency thresholds.
- **Shapes**:
    - 🟦 Square: Convolution (`ID_CONV`)
    - 🔵 Circle: Matrix Mult (`ID_MM`)
    - 🔺 Triangle: Element-wise (`ID_MUL`)
    - 🔷 Diamond: Other/Generic
    - ⭐ Star: Error/Unknown
