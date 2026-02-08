import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update, ALL, MATCH
import datetime
import os
import sys
import base64
import io
import webbrowser
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
    default = {
        "peak_tops": 32.0, "mem_bw": 50.0, "mem_thresh": 40, "comp_thresh": 40,
        "issue_eff_limit": 50, "issue_waste_limit": 20, "issue_perf_limit": 1.0
    }
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
            theo_macs, theo_bytes, symbol = custom_res
        else:
            continue

        real_bytes = cmd.get('hw_bytes', theo_bytes) 
        time_sec = cmd['time_ns'] / 1e9

        ai_algo = theo_macs / theo_bytes if theo_bytes > 0 else 0 
        ai_real = theo_macs / real_bytes if real_bytes > 0 else 0 

        perf = (theo_macs / time_sec) / 1e12
        data_efficiency = (theo_bytes / real_bytes) * 100 if real_bytes > 0 else 0
        actual_bw_used = (real_bytes / time_sec) / 1e9
        bw_efficiency = (actual_bw_used / mem_bw) * 100 if mem_bw > 0 else 0
        
        roofline_ceiling = min(peak_tops, (mem_bw * ai_real) / 1000)
        util = (perf / roofline_ceiling) * 100 if roofline_ceiling > 0 else 0
        global_util = (perf / peak_tops) * 100

        # --- Status Logic (Corrected) ---
        if ai_real >= turning_point:
            # Compute Region
            if util > 80: status = "Compute Bound"
            elif global_util < comp_thresh: status = "Compute Inefficient"
            else: status = "Normal"
        else:
            # Memory Region
            if util > 80: status = "Memory Bound"
            else:
                bw_thresh_gb = mem_bw * (mem_thresh / 100.0)
                if actual_bw_used < bw_thresh_gb: status = "Memory Inefficient"
                else: status = "Normal"

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
            "note": cmd.get('note', ''),
            "raw": cmd
        })
    return pd.DataFrame(processed)

# --- 5. Dash App Layout ---
app = Dash(__name__, suppress_callback_exceptions=True) 
current_cfg = load_config()

# Styles
input_mini_style = {'width': '100%', 'fontSize': '11px', 'padding': '2px', 'border': '1px solid #ccc', 'borderRadius': '3px'}
label_mini_style = {'fontSize': '10px', 'fontWeight': 'bold', 'color': '#555', 'marginBottom': '2px', 'display': 'block', 'whiteSpace': 'nowrap'}

# Analysis Guide Block
def get_analysis_guide():
    return html.Div(style={'marginTop': '20px', 'padding': '15px', 'background': '#f8f9fa', 'borderRadius': '8px', 'border': '1px solid #ddd'}, children=[
        html.H4("📘 Analysis Guide", style={'marginTop': '0px', 'fontSize': '14px'}),
        html.Div(style={'fontSize': '12px', 'lineHeight': '1.6'}, children=[
            html.P([html.B("📈 Efficiency Gap: "), "Difference between Real and Ideal points. Indicates Memory Waste."]),
            html.Hr(style={'margin': '8px 0'}),
            html.B("Status Legend:"),
            html.Ul(style={'paddingLeft': '20px', 'margin': '5px 0'}, children=[
                html.Li([html.B("Compute Bound (🟢): "), "Good utilization in high AI region."]),
                html.Li([html.B("Memory Bound (🟢): "), "Good utilization in low AI region."]),
                html.Li([html.B("Inefficient (🔴): "), "Hardware is stalling or BW usage is low."]),
            ]),
        ])
    ])

app.layout = html.Div(style={'display': 'flex', 'height': '100vh', 'fontFamily': 'Segoe UI, Arial', 'overflow': 'hidden'}, children=[
    dcc.Store(id='raw-json-store'), 
    dcc.Store(id='processed-df-store'),
    dcc.Store(id='active-filename-store', data=DEFAULT_FILENAME),
    dcc.Store(id='selected-ids-store', data=[]), # Source of truth for selection
    dcc.Download(id="download-csv"), 
    
    # --- PANEL 1: Node Filter Sidebar (Left, 20%) ---
    html.Div(style={'flex': '20', 'minWidth': '240px', 'maxWidth': '25%', 'background': '#f8f9fa', 'borderRight': '1px solid #ddd', 'display': 'flex', 'flexDirection': 'column'}, children=[
        
        # 1. Search & Select
        html.Div(style={'padding': '15px', 'borderBottom': '1px solid #ddd', 'background': '#fff'}, children=[
            html.H3("🔍 Filter Nodes", style={'margin': '0 0 10px 0', 'fontSize': '16px'}),
            dcc.Input(id='node-search', type='text', placeholder='Search name...', style={'width': '100%', 'padding': '6px', 'boxSizing': 'border-box', 'marginBottom': '10px'}),
            html.Div(style={'display': 'flex', 'gap': '5px'}, children=[
                html.Button("Select All", id='btn-select-all', n_clicks=0, style={'flex': 1, 'fontSize': '11px', 'padding': '2px'}),
                html.Button("Clear", id='btn-clear-all', n_clicks=0, style={'flex': 1, 'fontSize': '11px', 'padding': '2px'}),
            ])
        ]),

        # 2. Metric Filters (Saved Config Loaded)
        html.Div(style={'padding': '10px 15px', 'borderBottom': '1px solid #ddd', 'background': '#fafafa'}, children=[
            html.B("📊 Metrics Filter", style={'fontSize': '13px'}),
            html.Div(style={'display': 'flex', 'gap': '8px', 'marginTop': '8px', 'alignItems': 'flex-end'}, children=[
                html.Div(style={'flex': 1}, children=[html.Label("Eff % <", style=label_mini_style), dcc.Input(id='filter-eff-limit', type='number', value=current_cfg.get('issue_eff_limit'), style=input_mini_style)]),
                html.Div(style={'flex': 1}, children=[html.Label("Waste % >", style=label_mini_style), dcc.Input(id='filter-waste-limit', type='number', value=current_cfg.get('issue_waste_limit'), style=input_mini_style)]),
                html.Div(style={'flex': 1}, children=[html.Label("Perf <", style=label_mini_style), dcc.Input(id='filter-perf-limit', type='number', value=current_cfg.get('issue_perf_limit'), style=input_mini_style)]),
            ]),
            html.Div(id='filter-count-label', style={'marginTop': '5px', 'fontSize': '10px', 'color': '#888', 'textAlign': 'right'})
        ]),

        # 3. Dynamic Node List (Custom Row Components)
        html.Div(id='node-list-container', style={'flex': '1', 'overflowY': 'auto', 'padding': '5px'})
    ]),

    # --- PANEL 2: Main Chart & Controls (Middle, 55%) ---
    html.Div(style={'flex': '55', 'padding': '20px', 'display': 'flex', 'flexDirection': 'column', 'borderRight': '1px solid #ddd'}, children=[
        html.Div(style={'background': '#f1f3f4', 'padding': '15px', 'borderRadius': '8px', 'marginBottom': '10px'}, children=[
            html.Div(style={'display': 'flex', 'gap': '10px', 'alignItems': 'center', 'flexWrap': 'wrap'}, children=[
                html.Div([html.B("TOPS: "), dcc.Input(id='tops-input', type='number', value=current_cfg['peak_tops'], style={'width': '50px'})]),
                html.Div([html.B("BW: "), dcc.Input(id='bw-input', type='number', value=current_cfg['mem_bw'], style={'width': '50px'})]),
                html.Button('🔄 Reload', id='reload-btn', n_clicks=0, style={'background': '#e8f0fe'}),
                dcc.Upload(id='upload-data', children=html.Button('📂 Open', style={'background': '#e8f0fe'}), multiple=False, style={'display': 'inline-block'}),
                html.Button('📊 Export CSV', id='export-csv-btn', n_clicks=0, style={'background': '#e8f0fe'}),
                html.Button('💾 Save Config', id='save-config-btn', n_clicks=0, style={'background': '#e8f0fe'}),
                html.Span(id='config-status', style={'fontSize': '12px', 'color': '#4285f4', 'marginLeft': '10px', 'whiteSpace': 'nowrap'})
            ]),
            # Added drag mode for sliders
            html.Div(style={'display': 'flex', 'gap': '20px', 'marginTop': '10px'}, children=[
                html.Div([html.Label("Mem Thresh%"), dcc.Slider(0,100,5, value=current_cfg['mem_thresh'], id='mem-thresh-slider', updatemode='drag')], style={'flex': 1}),
                html.Div([html.Label("Comp Thresh%"), dcc.Slider(0,100,5, value=current_cfg['comp_thresh'], id='comp-thresh-slider', updatemode='drag')], style={'flex': 1})
            ]),
            html.Div(style={'marginTop': '10px', 'borderTop': '1px solid #ddd', 'paddingTop': '5px', 'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'fontSize': '12px'}, children=[
                dcc.Checklist(id='status-filter', options=[{'label': html.Span(f" {k}", style={'color': v, 'fontWeight': 'bold', 'marginRight': '5px'}), 'value': k} for k, v in COLOR_MAP.items()], value=list(COLOR_MAP.keys()), inline=True),
                dcc.Checklist(id='show-gap-toggle', options=[{'label': ' 📈 Efficiency Gap', 'value': 'SHOW_H'}], value=[], inline=True, style={'fontWeight': 'bold', 'color': '#e74c3c'})
            ])
        ]),
        dcc.Graph(id='roofline-chart', style={'flex-grow': '1'})
    ]),
    
    # --- PANEL 3: Detail Panel (Right, 25%) ---
    html.Div(id='detail-panel', style={'flex': '25', 'padding': '20px', 'background': '#fdfdfd', 'overflowY': 'auto'})
])

# --- 6. Callbacks ---

def format_bytes(size):
    power = 2**10
    n = 0
    power_labels = {0 : '', 1: 'KB', 2: 'MB', 3: 'GB'}
    while size > power:
        size /= power
        n += 1
    return f"{size:.2f} {power_labels.get(n, 'TB')}"

# B. File Loading Logic
@app.callback(
    [Output('raw-json-store', 'data'), Output('config-status', 'children', allow_duplicate=True), Output('active-filename-store', 'data')],
    [Input('upload-data', 'contents'), Input('upload-data', 'filename'), Input('reload-btn', 'n_clicks')],
    [State('active-filename-store', 'data')], prevent_initial_call='initial_duplicate'
)
def manage_file_state(contents, uploaded_filename, reload_clicks, current_active_file):
    trigger_id = ctx.triggered_id
    data = []
    
    if trigger_id == 'upload-data' and contents:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            data = json.load(io.BytesIO(decoded))
            msg = f"📂 Opened: {uploaded_filename}"
            return data, msg, uploaded_filename 
        except Exception as e:
            return no_update, f"❌ Error: {e}", current_active_file

    target_file = current_active_file if current_active_file else DEFAULT_FILENAME
    full_path = os.path.join(get_base_path(), target_file)
    
    if os.path.exists(full_path):
        try:
            with open(full_path, 'r') as f:
                data = json.load(f)
            msg = f"🔄 Reloaded: {target_file}" if trigger_id == 'reload-btn' else f"✅ Loaded: {target_file}"
            return data, msg, target_file
        except:
             return no_update, f"❌ Error reading {target_file}", target_file
    else:
        if trigger_id == 'reload-btn': return no_update, f"⚠️ File not found locally", target_file
        
    return no_update, "⚠️ Ready", target_file

# Init Load
@app.callback(Output('upload-data', 'contents'), Input('roofline-chart', 'id'))
def init_load(_): return None 

# C. Process Data (Recalculates status based on sliders)
@app.callback(
    Output('processed-df-store', 'data'), 
    [Input('raw-json-store', 'data'), Input('tops-input', 'value'), Input('bw-input', 'value'), 
     Input('mem-thresh-slider', 'value'), Input('comp-thresh-slider', 'value')]
)
def update_calcs(raw, t, b, mt, ct):
    if not raw: return None
    return calculate_metrics(raw, t, b, mt, ct).to_dict('records')

# D. Update Node List (Rendering with Pattern Matching)
@app.callback(
    [Output('node-list-container', 'children'), Output('filter-count-label', 'children')],
    [Input('processed-df-store', 'data'), Input('node-search', 'value'), 
     Input('filter-eff-limit', 'value'), Input('filter-waste-limit', 'value'), Input('filter-perf-limit', 'value'),
     Input('selected-ids-store', 'data')], # Trigger on selection change to re-render boxes
    prevent_initial_call=True
)
def update_node_list_render(data, search_term, eff_limit, waste_limit, perf_limit, selected_ids):
    if not data: return [], ""
    df = pd.DataFrame(data)
    
    # 1. Apply Filters
    if eff_limit is not None: df = df[df['util'] <= eff_limit]
    if waste_limit is not None: df = df[(100 - df['data_eff']) >= waste_limit]
    if perf_limit is not None: df = df[df['perf'] <= perf_limit]
    if search_term: df = df[df['label'].str.contains(search_term, case=False)]

    # 2. Render List
    selected_set = set(selected_ids or [])
    children = []
    
    # Optimization: If too many nodes, slice? For now render all filtered.
    for _, row in df.iterrows():
        nid = row['id']
        is_selected = nid in selected_set
        
        row_div = html.Div(style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '2px'}, children=[
            dcc.Checklist(
                options=[{'label': '', 'value': nid}],
                value=[nid] if is_selected else [],
                id={'type': 'node-chk', 'index': nid},
                style={'marginRight': '5px'} 
            ),
            html.Div(
                row['label'],
                id={'type': 'node-lbl', 'index': nid},
                n_clicks=0,
                style={
                    'cursor': 'pointer', 'fontSize': '12px', 'whiteSpace': 'nowrap', 
                    'overflow': 'hidden', 'textOverflow': 'ellipsis', 'flex': 1,
                    'color': '#2980b9' if is_selected else 'black',
                    'fontWeight': 'bold' if is_selected else 'normal'
                }
            )
        ])
        children.append(row_div)

    return children, f"Showing {len(df)} nodes"

# D2. Sync Selection State
@app.callback(
    Output('selected-ids-store', 'data'),
    [Input({'type': 'node-chk', 'index': ALL}, 'value'),
     Input('btn-select-all', 'n_clicks'), Input('btn-clear-all', 'n_clicks')],
    [State('selected-ids-store', 'data'), State({'type': 'node-chk', 'index': ALL}, 'id')]
)
def sync_selection(checkbox_values, sel_all, clear_all, current_selection, checkbox_ids):
    trigger = ctx.triggered_id
    current_set = set(current_selection or [])
    
    # Get IDs of currently visible checkboxes
    visible_ids = [c_id['index'] for c_id in checkbox_ids]
    
    if trigger == 'btn-clear-all':
        return []
    
    if trigger == 'btn-select-all':
        current_set.update(visible_ids)
        return list(current_set)
    
    # For individual toggles:
    # 1. Identify which visible ones are checked
    visible_checked = set()
    for val in checkbox_values:
        if val: visible_checked.add(val[0])
        
    # 2. Merge: (Existing selection MINUS All Visible) UNION Visible Checked
    # This preserves selections of nodes that are currently filtered out
    final_set = (current_set - set(visible_ids)) | visible_checked
    
    return list(final_set)

# E. Update Chart
@app.callback(
    Output('roofline-chart', 'figure'), 
    [Input('processed-df-store', 'data'), Input('tops-input', 'value'), Input('bw-input', 'value'),
     Input('status-filter', 'value'), Input('show-gap-toggle', 'value'), 
     Input('selected-ids-store', 'data')] 
)
def update_chart(data, tops, bw, selected_statuses, show_gap, selected_node_ids):
    if not data: return go.Figure()
    df_all = pd.DataFrame(data)
    df_curr = df_all[df_all['status'].isin(selected_statuses)]
    
    if selected_node_ids:
        df_curr = df_curr[df_curr['id'].isin(selected_node_ids)]
    
    fig = go.Figure()
    xr = np.logspace(-2, 4, 1000)
    fig.add_trace(go.Scatter(x=xr, y=np.minimum(tops, (bw * xr) / 1000), mode='lines', line=dict(color='black', width=3), name='Roofline', hoverinfo='skip'))
    
    if 'SHOW_H' in show_gap and not df_curr.empty:
        for _, row in df_curr.iterrows():
            if abs(row['ai_algo'] - row['ai_real']) / (row['ai_algo'] + 1e-9) > 0.05:
                fig.add_trace(go.Scatter(x=[row['ai_real'], row['ai_algo']], y=[row['perf'], row['perf']], mode='lines', line=dict(color='red', width=2, dash='dot'), showlegend=False, hoverinfo='skip'))
                fig.add_trace(go.Scatter(x=[row['ai_algo']], y=[row['perf']], mode='markers', marker=dict(size=6, color=row['status_color'], symbol='circle-open', opacity=0.5), showlegend=False, hovertext=f"Ideal AI: {row['ai_algo']:.2f}"))

    if not df_curr.empty:
        fig.add_trace(go.Scatter(
            x=df_curr['ai_real'], y=df_curr['perf'], mode='markers', 
            marker=dict(size=14, color=df_curr['status_color'], symbol=df_curr['symbol'], line=dict(width=1, color='white'), opacity=0.9), 
            customdata=df_curr.index.tolist(), text=df_curr['label'],
            hovertemplate="<b>%{text}</b><br>AI: %{x:.2f}<br>Perf: %{y:.2f} TOPS", name='Ops'
        ))
    
    fig.update_xaxes(type="log", title="Arithmetic Intensity", gridcolor='#eee')
    fig.update_yaxes(type="log", title="Performance (TOPS)", range=[-3.5, np.log10(tops*2)], gridcolor='#eee', tickformat=".0f")
    fig.update_layout(template="plotly_white", margin=dict(l=60, r=20, t=10, b=60), clickmode='event+select')
    return fig

# F. Show Detail Panel
@app.callback(
    Output('detail-panel', 'children'),
    [Input('roofline-chart', 'clickData'), Input({'type': 'node-lbl', 'index': ALL}, 'n_clicks')],
    [State('processed-df-store', 'data')],
    prevent_initial_call=True
)
def update_detail_panel(chart_click, list_clicks, df_data):
    trigger = ctx.triggered_id
    if not df_data: return no_update

    df = pd.DataFrame(df_data)
    target_id = None

    if trigger == 'roofline-chart' and chart_click:
        target_id = int(chart_click['points'][0]['customdata'])
    elif isinstance(trigger, dict) and trigger['type'] == 'node-lbl':
        target_id = trigger['index']

    if target_id is None: return no_update

    try:
        row = df[df['id'] == target_id].iloc[0]
    except: return no_update

    compute_gap = row['real_ceiling'] - row['perf']
    mem_waste = 100 - row['data_eff']
    
    return [
        html.H2(row['name'], style={'color': row['status_color'], 'marginBottom': '0px', 'fontSize': '20px'}),
        html.Div(row['type'], style={'fontSize': '14px', 'color': '#666', 'marginBottom': '10px'}),
        
        html.B("📉 Efficiency Analysis"),
        html.Table([
            html.Tr([html.Td("Roofline Eff:"), html.Td([html.B(f"{row['util']:.1f}%"), html.Span(f" (Lost: {compute_gap:.2f} TOPS)", style={'color': '#e74c3c', 'fontSize': '11px', 'marginLeft': '5px'})])]),
            html.Tr([html.Td("Bandwidth Eff:"), html.Td(html.B(f"{row['bw_eff']:.1f}%"))]),
            html.Tr([html.Td("Memory Eff:"), html.Td([html.B(f"{row['data_eff']:.1f}%"), html.Span(" ⚠️ Waste", style={'color': '#e74c3c', 'fontWeight': 'bold', 'fontSize': '11px'}) if mem_waste > 20 else html.Span(" ✅ Good", style={'color': 'green', 'fontSize': '11px'})])]),
        ], style={'width': '100%', 'lineHeight': '1.8', 'marginBottom': '15px'}),

        html.B("⚙️ Physical Stats"),
        html.Table([
            html.Tr([html.Td("Bandwidth:"), html.Td(f"{row['actual_bw']:.2f} GB/s")]),
            html.Tr([html.Td("Intensity (AI):"), html.Td(f"{row['ai_real']:.2f} MACs/Byte")]),
            html.Tr([html.Td("Performance (Real):"), html.Td(f"{row['perf']:.2f} TOPS")]),
            html.Tr([html.Td("Performance (Limit):"), html.Td(f"{row['real_ceiling']:.2f} TOPS")]),
            html.Tr([html.Td("Traffic (Real):"), html.Td(format_bytes(row['real_bytes']))]),
            html.Tr([html.Td("Traffic (Ideal):"), html.Td(format_bytes(row['theo_bytes']))]),
        ], style={'width': '100%', 'lineHeight': '1.6', 'fontSize': '13px', 'color': '#333'}),
        
        html.Hr(style={'margin': '10px 0'}),
        html.Details([html.Summary("View Raw JSON", style={'cursor': 'pointer', 'fontSize': '12px'}), html.Pre(json.dumps(row['raw'], indent=2), style={'background': '#eee', 'padding': '10px', 'fontSize': '10px', 'overflowX': 'auto'})]),
        
        get_analysis_guide()
    ]

# I. Export CSV Callback
@app.callback(
    Output("download-csv", "data"),
    Input("export-csv-btn", "n_clicks"),
    [State("processed-df-store", "data"), State("active-filename-store", "data")],
    prevent_initial_call=True
)
def export_csv(n, data, filename):
    if not data: return no_update
    df = pd.DataFrame(data)
    export_cols = ['name', 'type', 'status', 'perf', 'ai_real', 'util', 'bw_eff', 'data_eff', 'actual_bw']
    df_export = df[export_cols].copy()
    csv_name = filename.replace('.json', '.csv') if filename else "metrics_export.csv"
    return dcc.send_data_frame(df_export.to_csv, csv_name)

# J. Save Config (Expanded to save issue spotter settings)
@app.callback(
    Output('config-status', 'children'),
    Input('save-config-btn', 'n_clicks'),
    [State('tops-input', 'value'), State('bw-input', 'value'), 
     State('mem-thresh-slider', 'value'), State('comp-thresh-slider', 'value'),
     State('filter-eff-limit', 'value'), State('filter-waste-limit', 'value'), State('filter-perf-limit', 'value')],
    prevent_initial_call=True
)
def save_cfg(n, t, b, mt, ct, eff, waste, perf):
    cfg = {
        "peak_tops": t, "mem_bw": b, "mem_thresh": mt, "comp_thresh": ct,
        "issue_eff_limit": eff, "issue_waste_limit": waste, "issue_perf_limit": perf
    }
    with open(CONFIG_FILE, 'w') as f: json.dump(cfg, f)
    return f"✅ Saved at {datetime.datetime.now().strftime('%H:%M:%S')}"

if __name__ == '__main__':
    Timer(1.5, lambda: webbrowser.open('http://127.0.0.1:8050')).start()
    app.run(debug=False, port=8050)