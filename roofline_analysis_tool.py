import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import datetime
import os
import sys

# --- 1. 硬體與配置設定 ---
CONFIG_FILE = 'roofline_config.json'
DT_MAP = {"INT8": 1, "FP16": 2, "BF16": 2, "FP32": 4}
SYMBOL_MAP = {"ID_MM": "circle", "ID_CONV": "square", "OTHER": "triangle-up"}
COLOR_MAP = {
    "Compute Bound": "#27ae60",
    "Memory Bound": "#2ecc71",
    "Compute Inefficient": "#e74c3c",
    "Memory Inefficient": "#c0392b",
    "Normal": "#2980b9"
}

DEFAULT_CONFIG = {
    "peak_tops": 32.0,
    "mem_bw": 50.0,
    "mem_thresh": 40,
    "comp_thresh": 40
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            return DEFAULT_CONFIG
    return DEFAULT_CONFIG

def save_config_to_file(data):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# --- 2. 核心計算函數 ---
def calculate_metrics(raw_data, peak_tops, mem_bw, mem_thresh, comp_thresh):
    if not raw_data: return pd.DataFrame()
    processed = []
    # 臨界強度 AI* 計算
    turning_point = peak_tops / (mem_bw / 1000)

    for i, cmd in enumerate(raw_data):
        name = cmd.get('name', f"#{i}")
        dt_size = DT_MAP.get(cmd.get('data_type', "INT8"), 1)
        cmd_type = cmd['type']
        
        if cmd_type == "ID_CONV":
            n, ci, d, h, w = cmd['input_shape']
            co = cmd['output_channels']
            kd, kh, kw = cmd['kernel_size']
            sd, sh, sw = cmd['stride']
            pad_d, pad_h, pad_w = cmd['padding']
            do, ho, wo = (d+2*pad_d-kd)//sd+1, (h+2*pad_h-kh)//sh+1, (w+2*pad_w-kw)//sw+1
            macs = n * co * do * ho * wo * (ci * kd * kh * kw)
            bytes_moved = (n*ci*d*h*w + co*ci*kd*kh*kw + n*co*do*ho*wo) * dt_size
        elif cmd_type == "ID_MM":
            m1, m2 = cmd['input_mat1'], cmd['input_mat2']
            macs = np.prod(m1[:-2]) * m1[-2] * m2[-1] * m1[-1]
            bytes_moved = (np.prod(m1) + np.prod(m2) + (np.prod(m1[:-2])*m1[-2]*m2[-1])) * dt_size
        else:
            macs = np.prod(cmd.get('input_shape', [1]))
            bytes_moved = macs * 2 * dt_size
            
        ai = macs / bytes_moved
        perf = (macs / (cmd['time_us'] / 1e6)) / 1e12
        roof_limit = min(peak_tops, (mem_bw * ai) / 1000)
        util = (perf / roof_limit) * 100
        actual_bw = (bytes_moved / (cmd['time_us'] / 1e6)) / 1e9

        # 診斷狀態判定
        if ai >= turning_point:
            status = "Compute Bound" if util > 70 else ("Compute Inefficient" if util < comp_thresh else "Normal")
        else:
            status = "Memory Bound" if util > 70 else ("Memory Inefficient" if actual_bw < (mem_bw * mem_thresh / 100) else "Normal")

        processed.append({
            "name": name, "type": cmd_type, "ai": ai, "perf": perf, "util": util,
            "status": status, "status_color": COLOR_MAP[status],
            "symbol": SYMBOL_MAP.get(cmd_type, SYMBOL_MAP["OTHER"]),
            "actual_bw": actual_bw, "macs": macs, "bytes": bytes_moved, "raw": cmd
        })
    return pd.DataFrame(processed)

# --- 3. Dash App 佈局 ---
app = Dash(__name__)
current_cfg = load_config()

app.layout = html.Div(style={'display': 'flex', 'height': '100vh', 'fontFamily': 'Segoe UI, Arial'}, children=[
    dcc.Store(id='raw-json-store'),
    dcc.Store(id='processed-df-store'),

    html.Div(style={'flex': '75', 'padding': '20px', 'display': 'flex', 'flexDirection': 'column'}, children=[
        html.Div(style={'background': '#f1f3f4', 'padding': '15px', 'borderRadius': '8px', 'marginBottom': '15px'}, children=[
            html.Div(style={'display': 'flex', 'gap': '20px', 'alignItems': 'center'}, children=[
                html.Div([html.B("Peak TOPS: "), dcc.Input(id='tops-input', type='number', value=current_cfg['peak_tops'], style={'width': '60px'})]),
                html.Div([html.B("Mem BW (GB/s): "), dcc.Input(id='bw-input', type='number', value=current_cfg['mem_bw'], style={'width': '60px'})]),
                html.Button('🔄 Reload JSON', id='reload-btn', n_clicks=0, style={'padding': '5px 12px'}),
                html.Button('💾 Save Config', id='save-config-btn', n_clicks=0, style={'padding': '5px 12px', 'background': '#e8f0fe', 'border': '1px solid #4285f4'}),
                html.Span(id='config-status', style={'fontSize': '12px', 'color': '#4285f4'})
            ]),
            html.Div(style={'display': 'flex', 'gap': '40px', 'marginTop': '15px'}, children=[
                html.Div(style={'flex': '1'}, children=[
                    html.Label("Memory Inefficient Threshold (%)"),
                    dcc.Slider(0, 100, 5, value=current_cfg['mem_thresh'], id='mem-thresh-slider', marks={0:'0', 100:'100'})
                ]),
                html.Div(style={'flex': '1'}, children=[
                    html.Label("Compute Inefficient Threshold (%)"),
                    dcc.Slider(0, 100, 5, value=current_cfg['comp_thresh'], id='comp-thresh-slider', marks={0:'0', 100:'100'})
                ]),
            ])
        ]),
        dcc.Graph(id='roofline-chart', style={'flex-grow': '1'})
    ]),
    
    html.Div(id='detail-panel', style={'flex': '25', 'padding': '25px', 'background': '#fdfdfd', 'borderLeft': '1px solid #eee', 'overflowY': 'auto'}, children=[
        html.H3("Diagnostic Panel"), html.P("Initializing...")
    ])
])

# --- 4. Callbacks ---

# 儲存配置 (僅手動觸發)
@app.callback(
    Output('config-status', 'children'),
    Input('save-config-btn', 'n_clicks'),
    [State('tops-input', 'value'), State('bw-input', 'value'),
     State('mem-thresh-slider', 'value'), State('comp-thresh-slider', 'value')],
    prevent_initial_call=True
)
def save_config_callback(n_clicks, tops, bw, mem_t, comp_t):
    new_cfg = {"peak_tops": tops, "mem_bw": bw, "mem_thresh": mem_t, "comp_thresh": comp_t}
    save_config_to_file(new_cfg)
    return f"✅ Config Saved ({datetime.datetime.now().strftime('%H:%M:%S')})"

def get_base_path():
    if getattr(sys, 'frozen', False):
        # 執行檔模式
        return os.path.dirname(sys.executable)
    # 一般 Python 模式
    return os.path.dirname(os.path.abspath(__file__))

# 【修復關鍵】：資料讀取函數
# 將 prevent_initial_call 設為 'initial_duplicate' 以支援啟動自動執行
@app.callback(
    [Output('raw-json-store', 'data'), Output('detail-panel', 'children', allow_duplicate=True)], 
    [Input('reload-btn', 'n_clicks')],
    prevent_initial_call='initial_duplicate' 
)
def load_file(n_clicks):
    try:
        file_path = os.path.join(get_base_path(), 'commands.json')
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data, [html.H3("Diagnostic Panel"), html.P(f"✅ Auto-loaded {len(data)} items.")]
    except Exception as e:
        return None, [html.H3("Error"), html.P(f"❌ Could not load commands.json: {str(e)}")]

# 繪圖與指標 (自動監聽 raw-json-store)
@app.callback(
    Output('processed-df-store', 'data'), 
    [Input('raw-json-store', 'data'), Input('tops-input', 'value'), Input('bw-input', 'value'), 
     Input('mem-thresh-slider', 'value'), Input('comp-thresh-slider', 'value')]
)
def update_calcs(raw, tops, bw, mt, ct):
    if not raw: return None
    return calculate_metrics(raw, tops, bw, mt, ct).to_dict('records')

@app.callback(
    Output('roofline-chart', 'figure'), 
    [Input('processed-df-store', 'data'), Input('tops-input', 'value'), Input('bw-input', 'value')]
)
def update_chart(data, tops, bw):
    if not data: return go.Figure()
    df_curr = pd.DataFrame(data)
    fig = go.Figure()
    x_roof = np.logspace(-2, 4, 1000)
    y_roof = np.minimum(tops, (bw * x_roof) / 1000)
    fig.add_trace(go.Scatter(x=x_roof, y=y_roof, mode='lines', line=dict(color='red', width=3), name='Theoretical Roofline', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=df_curr['ai'], y=df_curr['perf'], mode='markers', 
                             marker=dict(size=18, color=df_curr['status_color'], symbol=df_curr['symbol'], line=dict(width=1, color='white')), 
                             customdata=df_curr.index.tolist(), hovertext=df_curr['name']))
    fig.update_xaxes(type="log", title="Arithmetic Intensity (MACs/Byte)")
    fig.update_yaxes(type="log", title="Performance (TOPS)", range=[-3.5, np.log10(tops*2)])
    fig.update_layout(template="plotly_white", margin=dict(l=60, r=20, t=10, b=60), clickmode='event+select')
    return fig

@app.callback(
    Output('detail-panel', 'children'), 
    [Input('roofline-chart', 'clickData'), State('processed-df-store', 'data'),
     State('tops-input', 'value'), State('bw-input', 'value')]
)
def show_detail(click, data, tops, bw):
    if not click or not data: return [html.H3("Diagnostic Panel"), html.P("Select a node.")]
    df_curr = pd.DataFrame(data)
    idx = int(click['points'][0].get('customdata'))
    row = df_curr.iloc[idx]
    
    return [
        html.H2(row['name'], style={'color': row['status_color'], 'marginBottom': '0px'}),
        html.Div(row['status'], style={'fontSize': '18px', 'fontWeight': 'bold', 'color': row['status_color'], 'marginBottom': '10px'}),
        html.Hr(),
        html.B("📊 Performance Metrics"),
        html.Table([
            html.Tr([html.Td("Actual Perf:"), html.Td(html.B(f"{row['perf']:.4f} TOPS"))]),
            html.Tr([html.Td("Utilization:"), html.Td(html.B(f"{row['util']:.2f} %"))]),
            html.Tr([html.Td("AI Intensity:"), html.Td(f"{row['ai']:.2f} MACs/B")]),
        ], style={'width': '100%', 'lineHeight': '1.8'}),
        html.Br(),
        html.B("⚙️ Hardware Stats"),
        html.Table([
            html.Tr([html.Td("Actual DMA BW:"), html.Td(html.B(f"{row['actual_bw']:.2f} GB/s"))]),
            html.Tr([html.Td("Total MACs:"), html.Td(f"{row['macs']:.2e}")]),
            html.Tr([html.Td("Total Bytes:"), html.Td(f"{row['bytes']:.2e}")]),
        ], style={'width': '100%', 'lineHeight': '1.8', 'fontSize': '14px', 'color': '#555'}),
        html.Hr(),
        html.Pre(json.dumps(row['raw'], indent=2), style={'background': '#2d3436', 'color': '#eee', 'padding': '10px', 'fontSize': '11px'})
    ]

if __name__ == '__main__':
    import webbrowser
    from threading import Timer

    Timer(2, lambda: webbrowser.open('http://127.0.0.1:8050')).start()
    app.run(debug=False)