# Roofline Analysis Tool

A powerful, interactive visualization tool built with **Dash** and **Plotly** to analyze NPU/Hardware performance using the [Roofline Model](https://en.wikipedia.org/wiki/Roofline_model). This tool helps identify whether operations are **Compute Bound** or **Memory Bound** and diagnoses efficiency issues.

## 🚀 Features

- **Interactive Roofline Chart**: Zoom, pan, and hover over operations to see detailed metrics.
- **Real-time Configuration**: Adjust hardware parameters (Peak TOPS, Memory Bandwidth) and efficiency thresholds on the fly.
- **Diagnostic Panel**: Click on any node to view detailed statistics (MACs, Bytes, Utilization, Bandwidth).
- **Auto-Reload**: Automatically detects and reloads changes in `commands.json`.
- **Config Persistence**: Saves your hardware settings to `roofline_config.json`.

## 📂 File Structure

- `roofline_analysis_tool.py`: The main application script (Dash app).
- `commands.json`: Input data file containing the list of operations to analyze.
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
        "time_us": 850.0
    },
    {
        "type": "ID_MM",
        "name": "#2",
        "data_type": "FP16",
        "input_mat1": [1, 1, 1, 512, 1024],
        "input_mat2": [1, 1, 1, 1024, 512],
        "time_us": 320.0
    }
]
```

### Supported Types:
- **ID_CONV**: Convolution operations (requires `input_shape`, `kernel_size`, `stride`, etc.)
- **ID_MM**: Matrix Multiplication (requires `input_mat1`, `input_mat2`)
- **Generic**: Other operations (fallback calculation based on `input_shape`)

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
    - 🟥 Square: Convolution (`ID_CONV`)
    - 🔵 Circle: Matrix Mult (`ID_MM`)
    - 🔺 Triangle: Other
