import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import datetime
import os
import sys
import webbrowser
from threading import Timer
import importlib.util

# --- 1. Path and Environment Adaptation ---
def get_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

CONFIG_FILE = os.path.join(get_base_path(), 'roofline_config.json')
DATA_FILE = os.path.join(get_base_path(), 'commands.json')
PLUGIN_FILE = os.path.join(get_base_path(), 'op_registry.py')

# --- 2. Core Settings ---
DT_MAP = {"INT8": 1, "FP16": 2, "BF16": 2, "FP32": 4}
COLOR_MAP = {
    "Compute Bound": "#27ae60", 
    "Memory Bound": "#2ecc71",
    "Compute Inefficient": "#e74c3c", 
    "Memory Inefficient": "#c0392b", 
    "Normal": "#2980b9"
}

def load_config():
    default = {"peak_tops": 32.0, "mem_bw": 50.0, "mem_thresh": 40, "comp_thresh": 40}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return {**default, **json.load(f)}
        except:
            return default
    return default

# --- 3. Plugin Integration Logic ---
def get_metrics_from_plugin(cmd, dt_size):
    if not os.path.exists(PLUGIN_FILE):
        return None
    try:
        spec = importlib.util.spec_from_file_location("op_registry", PLUGIN_FILE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.get_custom_metrics(cmd, dt_size)
    except Exception as e:
        print(f"Plugin Error: {e}")
        return None

# --- 4. Core Calculation Function ---
def calculate_metrics(raw_data, peak_tops, mem_bw, mem_thresh, comp_thresh):
    if not raw_data: return pd.DataFrame()
    processed = []
    turning_point = peak_tops / (mem_bw / 1000)

    for i, cmd in enumerate(raw_data):
        dt_size = DT_MAP.get(cmd.get('data_type', "INT8"), 1)
        
        custom_res = get_metrics_from_plugin(cmd, dt_size)
        if custom_res:
            macs, bytes_moved, symbol = custom_res
        else:
            print("Warning: No plugin found for command: ", cmd)
            continue

        ai = macs / bytes_moved
        # Use time_ns as requested
        time_sec = cmd['time_ns'] / 1e9
        perf = (macs / time_sec) / 1e12
        util = (perf / min(peak_tops, (mem_bw * ai) / 1000)) * 100
        actual_bw = (bytes_moved / time_sec) / 1e9

        if ai >= turning_point:
            status = "Compute Bound" if util > 70 else ("Compute Inefficient" if util < comp_thresh else "Normal")
        else:
            status = "Memory Bound" if util > 70 else ("Memory Inefficient" if actual_bw < (mem_bw * mem_thresh / 100) else "Normal")

        processed.append({
            "name": cmd.get('name', f"#{i}"), "type": cmd['type'], "ai": ai, "perf": perf, "util": util,
            "status": status, "status_color": COLOR_MAP[status], "symbol": symbol,
            "actual_bw": actual_bw, "macs": macs, "bytes": bytes_moved, "raw": cmd
        })
    return pd.DataFrame(processed)

# --- 5. Dash App Layout ---
app = Dash(__name__)
current_cfg = load_config()

app.layout = html.Div(style={'display': 'flex', 'height': '100vh', 'fontFamily': 'Segoe UI, Arial'}, children=[
    dcc.Store(id='raw-json-store'), dcc.Store(id='processed-df-store'),
    
    # Left Control Area and Chart
    html.Div(style={'flex': '75', 'padding': '20px', 'display': 'flex', 'flexDirection': 'column'}, children=[
        html.Div(style={'background': '#f1f3f4', 'padding': '15px', 'borderRadius': '8px', 'marginBottom': '10px'}, children=[
            html.Div(style={'display': 'flex', 'gap': '15px', 'alignItems': 'center', 'flexWrap': 'wrap'}, children=[
                html.Div([html.B("Peak TOPS: "), dcc.Input(id='tops-input', type='number', value=current_cfg['peak_tops'], style={'width': '60px'})]),
                html.Div([html.B("BW (GB/s): "), dcc.Input(id='bw-input', type='number', value=current_cfg['mem_bw'], style={'width': '60px'})]),
                html.Button('🔄 Reload JSON', id='reload-btn', n_clicks=0),
                html.Button('💾 Save Config', id='save-config-btn', n_clicks=0, style={'background': '#e8f0fe'}),
                html.Span(id='config-status', style={'fontSize': '12px', 'color': '#4285f4'})
            ]),
            
            # Threshold Sliders
            html.Div(style={'display': 'flex', 'gap': '30px', 'marginTop': '15px'}, children=[
                html.Div([html.Label("Memory Inefficient Threshold (%)"), dcc.Slider(0,100,5, value=current_cfg['mem_thresh'], id='mem-thresh-slider')], style={'flex': 1}),
                html.Div([html.Label("Compute Inefficient Threshold (%)"), dcc.Slider(0,100,5, value=current_cfg['comp_thresh'], id='comp-thresh-slider')], style={'flex': 1})
            ]),

            # New: Status Filter Checkbox
            html.Div(style={'marginTop': '15px', 'borderTop': '1px solid #ddd', 'paddingTop': '10px'}, children=[
                html.B("Show Diagnostic Categories: "),
                dcc.Checklist(
                    id='status-filter',
                    options=[
                        {'label': html.Span(f" {k}", style={'color': v, 'fontWeight': 'bold', 'marginRight': '15px'}), 'value': k}
                        for k, v in COLOR_MAP.items()
                    ],
                    value=list(COLOR_MAP.keys()), # Select all by default
                    inline=True
                )
            ])
        ]),
        
        dcc.Graph(id='roofline-chart', style={'flex-grow': '1'})
    ]),
    
    # Right Detail Panel
    html.Div(id='detail-panel', style={'flex': '25', 'padding': '25px', 'background': '#fdfdfd', 'borderLeft': '1px solid #eee', 'overflowY': 'auto'})
])

# --- 6. Callbacks ---

@app.callback(
    [Output('raw-json-store', 'data'), Output('detail-panel', 'children', allow_duplicate=True)], 
    [Input('reload-btn', 'n_clicks')], prevent_initial_call='initial_duplicate'
)
def load_file(n):
    if not os.path.exists(DATA_FILE): return None, html.P("❌ commands.json missing")
    with open(DATA_FILE, 'r') as f: data = json.load(f)
    return data, html.P(f"✅ Successfully loaded ({len(data)} commands).")

@app.callback(
    Output('processed-df-store', 'data'), 
    [Input('raw-json-store', 'data'), Input('tops-input', 'value'), Input('bw-input', 'value'), 
     Input('mem-thresh-slider', 'value'), Input('comp-thresh-slider', 'value')]
)
def update_calcs(raw, t, b, mt, ct):
    if not raw: return None
    return calculate_metrics(raw, t, b, mt, ct).to_dict('records')

@app.callback(
    Output('roofline-chart', 'figure'), 
    [Input('processed-df-store', 'data'), Input('tops-input', 'value'), Input('bw-input', 'value'),
     Input('status-filter', 'value')] # New filter input
)
def update_chart(data, tops, bw, selected_statuses):
    if not data: return go.Figure()
    
    df_all = pd.DataFrame(data)
    # Filter data based on selected statuses
    df_curr = df_all[df_all['status'].isin(selected_statuses)]
    
    fig = go.Figure()
    xr = np.logspace(-2, 4, 1000)
    
    # Draw Theoretical Roofline
    fig.add_trace(go.Scatter(
        x=xr, y=np.minimum(tops, (bw * xr) / 1000), 
        mode='lines', line=dict(color='red', width=3), 
        name='Theoretical Roofline', hoverinfo='skip'
    ))
    
    # Draw Data Points
    if not df_curr.empty:
        fig.add_trace(go.Scatter(
            x=df_curr['ai'], y=df_curr['perf'], 
            mode='markers', 
            marker=dict(size=18, color=df_curr['status_color'], symbol=df_curr['symbol'], line=dict(width=1, color='white')), 
            customdata=df_curr.index.tolist(), 
            hovertext=df_curr['name'],
            name='Ops'
        ))
    
    fig.update_xaxes(type="log", title="Arithmetic Intensity (MACs/Byte)", gridcolor='#eee')
    fig.update_yaxes(type="log", title="Performance (TOPS)", range=[-3.5, np.log10(tops*2)], gridcolor='#eee')
    fig.update_layout(template="plotly_white", margin=dict(l=60, r=20, t=10, b=60), clickmode='event+select')
    return fig

@app.callback(Output('detail-panel', 'children'), [Input('roofline-chart', 'clickData'), State('processed-df-store', 'data')])
def show_detail(click, data):
    # Definition Block
    definition_block = html.Div(style={'marginTop': '30px', 'padding': '15px', 'background': '#f8f9fa', 'borderRadius': '8px', 'border': '1px solid #ddd'}, children=[
        html.H4("📘 Diagnostic Definitions", style={'marginTop': '0px'}),
        html.Div(style={'fontSize': '13px', 'lineHeight': '1.6'}, children=[
            html.P([html.B("1. Turning Point (AI*): "), "Peak TOPS / Peak Bandwidth. The boundary between Memory and Compute bound regions."]),
            html.Ul([
                html.Li([html.B("Compute Bound (🟢): "), "In the plateau region with >70% utilization. PE array is fully utilized."]),
                html.Li([html.B("Memory Bound (🟢): "), "In the slope region with >70% utilization. DMA bandwidth is saturated."]),
                html.Li([html.B("Compute Inefficient (🔴): "), "In the plateau but below threshold. Usually caused by small op dimensions or poor tiling."]),
                html.Li([html.B("Memory Inefficient (🔴): "), "In the slope but below threshold. Caused by misalignment, large strides, or small DMA bursts."]),
            ]),
            html.Hr(style={'margin': '10px 0'}),
            html.B("💡 Calculation Formulas:"),
            html.Ul([
                html.Li("Intensity (AI) = MACs / Total Bytes Moved"),
                html.Li("Utilization = Actual TOPS / Theoretical Max at given AI"),
                html.Li("Actual BW = Bytes Moved / Execution Time")
            ])
        ])
    ])

    if not click or not data: 
        return [html.H3("Diagnostic Panel"), html.P("Click a node in the chart to view metrics."), definition_block]
    
    df_full = pd.DataFrame(data)
    idx = int(click['points'][0]['customdata'])
    row = df_full.iloc[idx]
    
    return [
        html.H2(row['name'], style={'color': row['status_color'], 'marginBottom': '0px'}),
        html.Div(row['status'], style={'fontSize': '18px', 'fontWeight': 'bold', 'color': row['status_color'], 'marginBottom': '10px'}),
        html.Hr(),
        html.B("📊 Performance Metrics"),
        html.Table([
            html.Tr([html.Td("Op Type:"), html.Td(html.B(row['type']))]),
            html.Tr([html.Td("Actual Performance:"), html.Td(html.B(f"{row['perf']:.4f} TOPS"))]),
            html.Tr([html.Td("Hardware Utilization:"), html.Td(html.B(f"{row['util']:.2f} %"))]),
            html.Tr([html.Td("Arithmetic Intensity:"), html.Td(f"{row['ai']:.2f} MACs/B")]),
        ], style={'width': '100%', 'lineHeight': '1.8'}),
        html.Br(),
        html.B("⚙️ Hardware Stats"),
        html.Table([
            html.Tr([html.Td("Actual DMA Bandwidth:"), html.Td(html.B(f"{row['actual_bw']:.2f} GB/s"))]),
            html.Tr([html.Td("Total MACs:"), html.Td(f"{row['macs']:.2e}")]),
            html.Tr([html.Td("Total Bytes Moved:"), html.Td(f"{row['bytes']:.2e}")]),
        ], style={'width': '100%', 'lineHeight': '1.8', 'fontSize': '14px', 'color': '#555'}),
        html.Hr(),
        html.B("📄 Raw Config:"),
        html.Pre(json.dumps(row['raw'], indent=2), style={'background': '#2d3436', 'color': '#eee', 'padding': '10px', 'fontSize': '11px', 'marginTop': '15px'}),
        definition_block
    ]

# Save Config Callback
@app.callback(
    Output('config-status', 'children'),
    Input('save-config-btn', 'n_clicks'),
    [State('tops-input', 'value'), State('bw-input', 'value'),
     State('mem-thresh-slider', 'value'), State('comp-thresh-slider', 'value')],
    prevent_initial_call=True
)
def save_config_callback(n_clicks, tops, bw, mem_t, comp_t):
    new_cfg = {"peak_tops": tops, "mem_bw": bw, "mem_thresh": mem_t, "comp_thresh": comp_t}
    with open(CONFIG_FILE, 'w') as f:
        json.dump(new_cfg, f, indent=4)
    return f"✅ Config Saved ({datetime.datetime.now().strftime('%H:%M:%S')})"

if __name__ == '__main__':
    # Set timer to auto-open web page
    Timer(1.5, lambda: webbrowser.open('http://127.0.0.1:8050')).start()
    app.run(debug=False, port=8050)