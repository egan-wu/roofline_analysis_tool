# op_registry.py
import numpy as np

CONV_SYMBOL  = "square"
MM_SYMBOL    = "circle"
EW_SYMBOL    = "triangle-up"
OTHER_SYMBOL = "diamond"
ERR_SYMBOL   = "star"

def get_custom_metrics(cmd, dt_size):
    ctype = cmd.get('type')
    
    if ctype == "ID_MUL":
        m1, m2 = cmd['input_self'], cmd['input_other']
        macs = np.prod(m1)
        bytes_moved = (np.prod(m1) + np.prod(m2)) * dt_size
        return macs, bytes_moved, EW_SYMBOL

    elif ctype == "ID_MM":
        m1, m2 = cmd['input_mat1'], cmd['input_mat2']
        macs = np.prod(m1[:-2]) * m1[-2] * m2[-1] * m1[-1]
        bytes_moved = (np.prod(m1) + np.prod(m2) + (np.prod(m1[:-2])*m1[-2]*m2[-1])) * dt_size
        return macs, bytes_moved, MM_SYMBOL

    elif ctype == "ID_ADDMM":
        m1, m2 = cmd['input_mat1'], cmd['input_mat2']
        macs = np.prod(m1[:-2]) * m1[-2] * m2[-1] * m1[-1]
        bytes_moved = (np.prod(m1) + np.prod(m2) + (np.prod(m1[:-2])*m1[-2]*m2[-1])) * dt_size
        return macs, bytes_moved, MM_SYMBOL
        
    elif ctype == "ID_CONV":
        n, ci, d, h, w = cmd['input_shape']
        co, (kd, kh, kw), (sd, sh, sw), (p_d, ph, pw) = \
            cmd['output_channels'], cmd['kernel_size'], cmd['stride'], cmd['padding']
        
        do, ho, wo = (d+2*p_d-kd)//sd+1, (h+2*ph-kh)//sh+1, (w+2*pw-kw)//sw+1
        macs = n * co * do * ho * wo * (ci * kd * kh * kw)
        bytes_moved = (n*ci*d*h*w + co*ci*kd*kh*kw + n*co*do*ho*wo) * dt_size
        return macs, bytes_moved, CONV_SYMBOL

    # Let this be the last resort
    elif 'macs' in cmd and 'bytes' in cmd:
        return cmd['macs'], cmd['bytes'], OTHER_SYMBOL

    return 0, 0, ERR_SYMBOL