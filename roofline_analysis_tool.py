import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update
import datetime
import os
import sys
import webbrowser
import base64
import io
from threading import Timer
import importlib.util

# --- 1. Path and Environment Adaptation ---
def get_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

CONFIG_FILE = os.path.join(get_base_path(), 'roofline_config.json')
DEFAULT_FILENAME = 'commands.json' 
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
        
        # 1. Get Theoretical (Golden) Metrics from Registry
        custom_res = get_metrics_from_plugin(cmd, dt_size)
        if custom_res:
            theo_macs, theo_bytes, symbol = custom_res
        else:
            continue

        # 2. Get Real HW Metrics (fallback to theo if missing)
        real_bytes = cmd.get('hw_bytes', theo_bytes) 
        time_sec = cmd['time_ns'] / 1e9

        # 3. Calculate Two Intensities
        ai_algo = theo_macs / theo_bytes if theo_bytes > 0 else 0 
        ai_real = theo_macs / real_bytes if real_bytes > 0 else 0 

        # Performance is based on Real Time but Theo Work (Effective Throughput)
        perf = (theo_macs / time_sec) / 1e12
        
        # 4. Calculate Efficiency Metrics
        data_efficiency = (theo_bytes / real_bytes) * 100 if real_bytes > 0 else 0
        actual_bw_used = (real_bytes / time_sec) / 1e9
        bw_efficiency = (actual_bw_used / mem_bw) * 100 if mem_bw > 0 else 0
        
        # Utilization calculation (Local Util against Ceiling)
        roofline_ceiling = min(peak_tops, (mem_bw * ai_real) / 1000)
        util = (perf / roofline_ceiling) * 100 if roofline_ceiling > 0 else 0
        
        # Global Util for status check logic
        global_util = (perf / peak_tops) * 100

        # Determine status based on REAL behavior
        if ai_real >= turning_point:
            status = "Compute Bound" if global_util > 70 else ("Compute Inefficient" if global_util < comp_thresh else "Normal")
        else:
            status = "Memory Bound" if global_util > 70 else ("Memory Inefficient" if actual_bw_used < (mem_bw * mem_thresh / 100) else "Normal")

        processed.append({
            "id": i, 
            "name": cmd.get('name', f"#{i}"), 
            "type": cmd['type'], 
            "label": f"{cmd.get('name', f'#{i}')} ({cmd['type']})", 
            "ai_real": ai_real,      
            "ai_algo": ai_algo,      
            "perf": perf,            
            "util": util, 
            "data_eff": data_efficiency,
            "bw_eff": bw_efficiency,
            "status": status, 
            "status_color": COLOR_MAP[status], 
            "symbol": symbol,
            "actual_bw": actual_bw_used, 
            "theo_macs": theo_macs, 
            "theo_bytes": theo_bytes,
            "real_bytes": real_bytes,
            "real_ceiling": roofline_ceiling, 
            "raw": cmd
        })
    return pd.DataFrame(processed)

# --- 5. Dash App Layout ---
app = Dash(__name__)
current_cfg = load_config()

app.layout = html.Div(style={'display': 'flex', 'height': '100vh', 'fontFamily': 'Segoe UI, Arial', 'overflow': 'hidden'}, children=[
    dcc.Store(id='raw-json-store'), 
    dcc.Store(id='processed-df-store'),
    # Store to track which file is currently active
    dcc.Store(id='active-filename-store', data=DEFAULT_FILENAME), 
    
    # --- PANEL 1: Node Filter Sidebar (Left, 20%) ---
    html.Div(style={'flex': '15', 'minWidth': '200px', 'maxWidth': '20%', 'background': '#f8f9fa', 'borderRight': '1px solid #ddd', 'display': 'flex', 'flexDirection': 'column'}, children=[
        html.Div(style={'padding': '15px', 'borderBottom': '1px solid #ddd', 'background': '#fff'}, children=[
            html.H3("🔍 Search", style={'margin': '0 0 10px 0', 'fontSize': '16px'}),
            dcc.Input(id='node-search', type='text', placeholder='Filter by name...', style={'width': '100%', 'padding': '5px', 'boxSizing': 'border-box', 'marginBottom': '10px'}),
            html.Div(style={'display': 'flex', 'gap': '5px'}, children=[
                html.Button("Select All", id='btn-select-all', n_clicks=0, style={'flex': 1, 'fontSize': '11px', 'padding': '4px'}),
                html.Button("Clear All", id='btn-clear-all', n_clicks=0, style={'flex': 1, 'fontSize': '11px', 'padding': '4px'}),
            ])
        ]),
        # Scrollable Checklist Area
        html.Div(style={'flex': '1', 'overflowY': 'auto', 'padding': '10px'}, children=[
            dcc.Checklist(
                id='node-checklist',
                options=[],
                value=[],
                style={'fontSize': '12px', 'lineHeight': '1.5'},
                labelStyle={'display': 'block', 'marginBottom': '4px', 'cursor': 'pointer', 'whiteSpace': 'nowrap', 'overflow': 'hidden', 'textOverflow': 'ellipsis'}
            )
        ])
    ]),

    # --- PANEL 2: Main Chart & Controls (Middle, 55%) ---
    html.Div(style={'flex': '60', 'padding': '20px', 'display': 'flex', 'flexDirection': 'column', 'borderRight': '1px solid #ddd'}, children=[
        # Controls Header
        html.Div(style={'background': '#f1f3f4', 'padding': '15px', 'borderRadius': '8px', 'marginBottom': '10px'}, children=[
            html.Div(style={'display': 'flex', 'gap': '10px', 'alignItems': 'center', 'flexWrap': 'wrap'}, children=[
                html.Div([html.B("TOPS: "), dcc.Input(id='tops-input', type='number', value=current_cfg['peak_tops'], style={'width': '50px'})]),
                html.Div([html.B("BW: "), dcc.Input(id='bw-input', type='number', value=current_cfg['mem_bw'], style={'width': '50px'})]),
                
                # --- RELOAD (Resets to Default) ---
                html.Button('🔄 Reload', id='reload-btn', n_clicks=0, style={'background': '#e8f0fe'}),

                # --- UPLOAD (Loads New File) ---
                dcc.Upload(
                    id='upload-data',
                    children=html.Button('📂 Open', style={'background': '#e8f0fe'}),
                    multiple=False,
                    style={'display': 'inline-block'}
                ),
                
                html.Button('💾 Save Config', id='save-config-btn', n_clicks=0, style={'background': '#e8f0fe'}),
                html.Span(id='config-status', style={'fontSize': '12px', 'color': '#4285f4', 'marginLeft': '10px', 'whiteSpace': 'nowrap'})
            ]),
            
            # Threshold Sliders
            html.Div(style={'display': 'flex', 'gap': '20px', 'marginTop': '10px'}, children=[
                html.Div([html.Label("Memory Thresh%"), dcc.Slider(0,100,5, value=current_cfg['mem_thresh'], id='mem-thresh-slider')], style={'flex': 1}),
                html.Div([html.Label("Compute Thresh%"), dcc.Slider(0,100,5, value=current_cfg['comp_thresh'], id='comp-thresh-slider')], style={'flex': 1})
            ]),

            # Status Filter and Gap Toggle
            html.Div(style={'marginTop': '10px', 'borderTop': '1px solid #ddd', 'paddingTop': '5px', 'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'fontSize': '12px'}, children=[
                dcc.Checklist(
                    id='status-filter',
                    options=[{'label': html.Span(f" {k}", style={'color': v, 'fontWeight': 'bold', 'marginRight': '5px'}), 'value': k} for k, v in COLOR_MAP.items()],
                    value=list(COLOR_MAP.keys()),
                    inline=True
                ),
                dcc.Checklist(
                    id='show-gap-toggle',
                    options=[{'label': ' 📈 Show Efficiency Gap', 'value': 'SHOW_H'}],
                    value=[],
                    inline=True,
                    style={'fontWeight': 'bold', 'color': '#e74c3c'}
                )
            ])
        ]),
        
        dcc.Graph(id='roofline-chart', style={'flex-grow': '1'})
    ]),
    
    # --- PANEL 3: Detail Panel (Right, 25%) ---
    html.Div(id='detail-panel', style={'flex': '25', 'padding': '20px', 'background': '#fdfdfd', 'overflowY': 'auto'})
])

# --- 6. Callbacks ---

# A. Helper Function for Bytes
def format_bytes(size):
    power = 2**10
    n = 0
    power_labels = {0 : '', 1: 'KB', 2: 'MB', 3: 'GB'}
    while size > power:
        size /= power
        n += 1
    return f"{size:.2f} {power_labels.get(n, 'TB')}"

# B. MAIN FILE LOADER LOGIC (STATEFUL)
@app.callback(
    [Output('raw-json-store', 'data'), 
     Output('config-status', 'children', allow_duplicate=True),
     Output('active-filename-store', 'data')], 
    [Input('upload-data', 'contents'), 
     Input('upload-data', 'filename'), 
     Input('reload-btn', 'n_clicks')],
    [State('active-filename-store', 'data')], 
    prevent_initial_call='initial_duplicate'
)
def manage_file_state(contents, uploaded_filename, reload_clicks, current_active_file):
    trigger_id = ctx.triggered_id
    
    # 1. HANDLE OPEN/UPLOAD
    if trigger_id == 'upload-data' and contents:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            data = json.load(io.BytesIO(decoded))
            msg = f"📂 Opened: {uploaded_filename} ({len(data)} nodes)"
            return data, msg, uploaded_filename 
        except Exception as e:
            return no_update, f"❌ Error: {e}", current_active_file

    # 2. HANDLE RELOAD or INITIALIZATION
    target_file = current_active_file if current_active_file else DEFAULT_FILENAME
    full_path = os.path.join(get_base_path(), target_file)
    
    if os.path.exists(full_path):
        try:
            with open(full_path, 'r') as f:
                data = json.load(f)
            msg = f"🔄 Reloaded: {target_file} ({len(data)} nodes)" if trigger_id == 'reload-btn' else f"✅ Loaded: {target_file}"
            return data, msg, target_file
        except Exception as e:
             return no_update, f"❌ Error reading {target_file}", target_file
    else:
        if trigger_id == 'reload-btn':
            return no_update, f"⚠️ Cannot reload {target_file} (File not found locally)", target_file
        
    return no_update, "⚠️ Ready", target_file

# Trigger initial load
@app.callback(
    Output('upload-data', 'contents'),
    Input('roofline-chart', 'id') 
)
def init_load(_):
    return None 

# C. Process Data
@app.callback(
    Output('processed-df-store', 'data'), 
    [Input('raw-json-store', 'data'), Input('tops-input', 'value'), Input('bw-input', 'value'), 
     Input('mem-thresh-slider', 'value'), Input('comp-thresh-slider', 'value')]
)
def update_calcs(raw, t, b, mt, ct):
    if not raw: return None
    return calculate_metrics(raw, t, b, mt, ct).to_dict('records')

# D. Update Node Checklist (Filter Logic)
@app.callback(
    [Output('node-checklist', 'options'), Output('node-checklist', 'value')],
    [Input('processed-df-store', 'data'), 
     Input('node-search', 'value'),
     Input('btn-select-all', 'n_clicks'),
     Input('btn-clear-all', 'n_clicks')],
    [State('node-checklist', 'value')]
)
def update_node_list(data, search_term, btn_sel, btn_clr, current_selection):
    if not data: return [], []
    
    df = pd.DataFrame(data)
    options = [{'label': row['label'], 'value': row['id']} for _, row in df.iterrows()]
    all_ids = df['id'].tolist() # <--- FIXED: Defined this variable here
    
    triggered_id = ctx.triggered_id if ctx.triggered_id else ""
    
    if triggered_id == 'btn-clear-all':
        return options, []
    if triggered_id == 'btn-select-all':
        if search_term:
            filtered_ids = [opt['value'] for opt in options if search_term.lower() in opt['label'].lower()]
            new_selection = list(set((current_selection or []) + filtered_ids))
            return options, new_selection
        return options, all_ids
        
    if search_term:
        options = [opt for opt in options if search_term.lower() in opt['label'].lower()]
    
    if triggered_id == 'processed-df-store':
         return options, all_ids

    return options, current_selection

# E. Update Chart
@app.callback(
    Output('roofline-chart', 'figure'), 
    [Input('processed-df-store', 'data'), Input('tops-input', 'value'), Input('bw-input', 'value'),
     Input('status-filter', 'value'), Input('show-gap-toggle', 'value'),
     Input('node-checklist', 'value')]
)
def update_chart(data, tops, bw, selected_statuses, show_gap, selected_node_ids):
    if not data: return go.Figure()
    
    df_all = pd.DataFrame(data)
    
    # 1. Filter by Status
    df_curr = df_all[df_all['status'].isin(selected_statuses)]
    
    # 2. Filter by Node Checklist
    if selected_node_ids is not None:
        df_curr = df_curr[df_curr['id'].isin(selected_node_ids)]
    
    fig = go.Figure()
    xr = np.logspace(-2, 4, 1000)
    
    # Draw Theoretical Roofline (Concrete Black Line)
    fig.add_trace(go.Scatter(
        x=xr, y=np.minimum(tops, (bw * xr) / 1000), 
        mode='lines', line=dict(color='black', width=3), 
        name='Roofline', hoverinfo='skip'
    ))
    
    # --- Efficiency Gap Lines (Horizontal Only - Red Dotted) ---
    if 'SHOW_H' in show_gap and not df_curr.empty:
        for _, row in df_curr.iterrows():
            if abs(row['ai_algo'] - row['ai_real']) / (row['ai_algo'] + 1e-9) > 0.05:
                # Line connecting Real to Ideal
                fig.add_trace(go.Scatter(
                    x=[row['ai_real'], row['ai_algo']],
                    y=[row['perf'], row['perf']],
                    mode='lines',
                    line=dict(color='red', width=2, dash='dot'), # Red Dotted Line
                    showlegend=False, hoverinfo='skip'
                ))
                # Ghost Point (Ideal)
                fig.add_trace(go.Scatter(
                    x=[row['ai_algo']], y=[row['perf']],
                    mode='markers',
                    marker=dict(size=6, color=row['status_color'], symbol='circle-open', opacity=0.5),
                    showlegend=False,
                    hovertext=f"Ideal AI: {row['ai_algo']:.2f}"
                ))

    # --- Draw Real Data Points ---
    if not df_curr.empty:
        fig.add_trace(go.Scatter(
            x=df_curr['ai_real'], y=df_curr['perf'], mode='markers', 
            marker=dict(size=14, color=df_curr['status_color'], symbol=df_curr['symbol'], line=dict(width=1, color='white'), opacity=0.9), 
            customdata=df_curr.index.tolist(), text=df_curr['label'],
            hovertemplate="<b>%{text}</b><br>AI: %{x:.2f}<br>Perf: %{y:.2f} TOPS", name='Ops'
        ))
    
    fig.update_xaxes(type="log", title="Arithmetic Intensity", gridcolor='#eee')
    fig.update_yaxes(type="log", title="Performance (TOPS)", range=[-3.5, np.log10(tops*2)], gridcolor='#eee')
    fig.update_layout(template="plotly_white", margin=dict(l=60, r=20, t=10, b=60), clickmode='event+select')
    return fig

# F. Show Detail Panel (Redesigned)
@app.callback(Output('detail-panel', 'children'), [Input('roofline-chart', 'clickData'), State('processed-df-store', 'data')])
def show_detail(click, data):
    definition_block = html.Div(style={'marginTop': '20px', 'padding': '10px', 'background': '#f8f9fa', 'borderRadius': '5px', 'border': '1px solid #ddd', 'fontSize': '12px'}, children=[
        html.B("ℹ️ Quick Guide:"),
        html.Ul(style={'paddingLeft': '20px', 'margin': '5px 0'}, children=[
            html.Li("Roofline Eff: How close to the red line (Hardware Potential)."),
            html.Li("Bandwidth Eff: Actual Bandwidth / Peak Bandwidth (Bus Saturation)."),
            html.Li("Memory Eff: Ideal Bytes / Real Bytes (Software/Tiling Quality)."),
        ])
    ])

    if not click or not data: 
        return [html.H3("Detail Panel"), html.P("Select a node to view details."), definition_block]
    
    df_full = pd.DataFrame(data)
    idx = int(click['points'][0]['customdata'])
    row = df_full.iloc[idx]
    
    # Analysis Logic
    compute_gap_tops = row['real_ceiling'] - row['perf']
    mem_waste_percent = 100 - row['data_eff']
    is_wasteful = mem_waste_percent > 20 
    
    return [
        # Header
        html.H2(row['name'], style={'color': row['status_color'], 'marginBottom': '0px', 'fontSize': '20px'}),
        html.Div(row['type'], style={'fontSize': '14px', 'color': '#666', 'marginBottom': '5px'}),
        html.Div(row['status'], style={'fontSize': '16px', 'fontWeight': 'bold', 'color': row['status_color'], 'marginBottom': '15px'}),
        
        html.Hr(style={'margin': '10px 0'}),
        
        # Section 1: Efficiency Analysis
        html.B("📉 Efficiency Analysis"),
        html.Table([
            html.Tr([
                html.Td("Roofline Eff:"), 
                html.Td([
                    html.B(f"{row['util']:.1f}%"),
                    html.Span(f" (Lost: {compute_gap_tops:.2f} TOPS)", style={'color': '#e74c3c', 'fontSize': '11px', 'marginLeft': '5px'}) if compute_gap_tops > 0.1 else None
                ])
            ]),
            html.Tr([
                html.Td("Bandwidth Eff:"), 
                html.Td([
                    html.B(f"{row['bw_eff']:.1f}%"),
                ])
            ]),
            html.Tr([
                html.Td("Memory Eff:"), 
                html.Td([
                    html.B(f"{row['data_eff']:.1f}%"),
                    html.Span(" ⚠️ Waste", style={'color': '#e74c3c', 'fontWeight': 'bold', 'fontSize': '11px'}) if is_wasteful else html.Span(" ✅ Good", style={'color': 'green', 'fontSize': '11px'})
                ])
            ]),
        ], style={'width': '100%', 'lineHeight': '1.8', 'marginBottom': '15px'}),

        # Section 2: Physical Stats
        html.B("⚙️ Physical Stats"),
        html.Table([
            html.Tr([html.Td("Bandwidth:"), html.Td(f"{row['actual_bw']:.2f} GB/s")]),
            html.Tr([html.Td("Intensity (AI):"), html.Td(f"{row['ai_real']:.2f} MACs/Byte")]),
            html.Tr([html.Td("Performance (Real):"), html.Td(f"{row['perf']:.2f} TOPS")]),
            html.Tr([html.Td("Performance (Limit):"), html.Td(f"{row['real_ceiling']:.2f} TOPS")]),
            html.Tr([html.Td("Traffic (Real):"), html.Td(format_bytes(row['real_bytes']))]),
            html.Tr([html.Td("Traffic (Ideal):"), html.Td(format_bytes(row['theo_bytes']))]),
        ], style={'width': '100%', 'lineHeight': '1.6', 'fontSize': '13px', 'color': '#333'}),

        # Section 3: Debug Raw
        html.Hr(style={'margin': '10px 0'}),
        html.Details([
            html.Summary("View Raw JSON", style={'cursor': 'pointer', 'fontSize': '12px'}),
            html.Pre(json.dumps(row['raw'], indent=2), style={'background': '#eee', 'padding': '10px', 'fontSize': '10px', 'overflowX': 'auto'})
        ]),
        
        definition_block
    ]

# G. Save Config
@app.callback(
    Output('config-status', 'children'),
    Input('save-config-btn', 'n_clicks'),
    [State('tops-input', 'value'), State('bw-input', 'value'), State('mem-thresh-slider', 'value'), State('comp-thresh-slider', 'value')],
    prevent_initial_call=True
)
def save_cfg(n, t, b, mt, ct):
    with open(CONFIG_FILE, 'w') as f: json.dump({"peak_tops": t, "mem_bw": b, "mem_thresh": mt, "comp_thresh": ct}, f)
    return f"✅ Saved at {datetime.datetime.now().strftime('%H:%M:%S')}"

if __name__ == '__main__':
    Timer(1.5, lambda: webbrowser.open('http://127.0.0.1:8050')).start()
    app.run(debug=False, port=8050)