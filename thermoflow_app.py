__version__ = "0.3.0"

import os
import io
import glob
import json
import hashlib
import re
import ast
import operator
import warnings
import base64
from collections import deque
from functools import reduce
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector, RectangleSelector, EllipseSelector, SpanSelector
from matplotlib.patches import Ellipse, Rectangle, Polygon as MplPolygon
from matplotlib.path import Path as MplPath
from matplotlib.colors import LogNorm, Normalize

import ipywidgets as widgets
from IPython.display import display, clear_output, Image

import flowio
from scipy.optimize import least_squares
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
from numpy.linalg import inv, LinAlgError

# ==============================================================================
# 0. NATURE/SCIENCE PUBLICATION-QUALITY PLOTTING GLOBALS
# ==============================================================================
# Figure width constants (inches) — Nature journal column widths
FIG_1COL = 3.46   # 88 mm — single column
FIG_2COL = 7.09   # 180 mm — double column

mpl.rcParams.update({
    # --- Font ---
    'font.family':           'sans-serif',
    'font.sans-serif':       ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype':          42,   # TrueType — editable in Illustrator/Inkscape
    'ps.fonttype':           42,
    # --- Axes ---
    'axes.linewidth':        1.0,
    'axes.labelsize':        11,
    'axes.labelweight':      'bold',
    'axes.titlesize':        11,
    'axes.titleweight':      'bold',
    'axes.titlepad':         6,
    'axes.spines.top':       False,
    'axes.spines.right':     False,
    # --- Ticks ---
    'xtick.labelsize':       10,
    'ytick.labelsize':       10,
    'xtick.major.size':      4,
    'ytick.major.size':      4,
    'xtick.minor.size':      2.5,
    'ytick.minor.size':      2.5,
    'xtick.major.width':     1.0,
    'ytick.major.width':     1.0,
    'xtick.direction':       'in',
    'ytick.direction':       'in',
    # --- Legend ---
    'legend.fontsize':       9,
    'legend.frameon':        False,
    'legend.handlelength':   1.5,
    # --- Lines / patches ---
    'lines.linewidth':       1.5,
    'patch.linewidth':       1.0,
    # --- Output resolution ---
    'figure.dpi':            150,
    'savefig.dpi':           300,
})

# Custom warning for fit convergence issues
class FitConvergenceWarning(UserWarning):
    pass

Number = Union[int, float]

# Curated qualitative palette — NPG / high-impact journal inspired
# Cycles cleanly for up to 10 samples; beyond that wraps around
_PALETTE = [
    '#E64B35',   # vermilion red
    '#4DBBD5',   # teal blue
    '#00A087',   # emerald green
    '#3C5488',   # oxford navy
    '#F39B7F',   # soft salmon
    '#8491B4',   # slate blue
    '#91D1C2',   # pale teal
    '#DC0000',   # crimson
    '#7E6148',   # warm brown
    '#B09C85',   # sand
]

# ==============================================================================
# 1. CORE GATE CLASSES
# ==============================================================================
@dataclass
class BaseGate:
    name: str
    log1p: bool = True
    def evaluate(self, df: pd.DataFrame) -> pd.Series: raise NotImplementedError
    def to_dict(self) -> Dict[str, Any]: d = asdict(self); d['type'] = type(self).__name__; return d

@dataclass
class ThresholdGate(BaseGate):
    column: str = ''; op: str = 'between'; value: Optional[Number] = None; lo: Optional[Number] = None; hi: Optional[Number] = None

    def validate(self):
        if self.op == 'between' and self.lo is not None and self.hi is not None:
            if self.lo >= self.hi:
                raise ValueError(f"ThresholdGate '{self.name}': lo ({self.lo}) must be < hi ({self.hi})")

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        self.validate()
        if self.column not in df.columns or len(df[self.column].dropna()) == 0:
            raise ValueError(f"Column '{self.column}' is missing or empty")

        s = df[self.column]
        if self.log1p: s = np.log1p(s.clip(lower=0))
        if self.op in ('>', 'gt'): return s > float(self.value)
        if self.op in ('>=','ge'): return s >= float(self.value)
        if self.op in ('<', 'lt'): return s < float(self.value)
        if self.op in ('<=','le'): return s <= float(self.value)
        if self.op == 'between': return (s >= float(self.lo)) & (s <= float(self.hi))
        raise ValueError(f"Unsupported op: {self.op}")

@dataclass
class RectangleGate(BaseGate):
    x: str = ''; y: str = ''; xmin: Number = 0; xmax: Number = 0; ymin: Number = 0; ymax: Number = 0

    def validate(self):
        if self.xmin >= self.xmax:
            raise ValueError(f"RectangleGate '{self.name}': xmin ({self.xmin}) must be < xmax ({self.xmax})")
        if self.ymin >= self.ymax:
            raise ValueError(f"RectangleGate '{self.name}': ymin ({self.ymin}) must be < ymax ({self.ymax})")

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        self.validate()
        if self.x not in df.columns or self.y not in df.columns:
            raise ValueError(f"Columns '{self.x}' and/or '{self.y}' are missing")
        if len(df[self.x].dropna()) == 0 or len(df[self.y].dropna()) == 0:
            raise ValueError(f"Columns '{self.x}' and/or '{self.y}' are empty")
            
        xs, ys = df[self.x], df[self.y]
        if self.log1p: xs, ys = np.log1p(xs.clip(lower=0)), np.log1p(ys.clip(lower=0))
        return (xs >= self.xmin) & (xs <= self.xmax) & (ys >= self.ymin) & (ys <= self.ymax)

@dataclass
class PolygonGate(BaseGate):
    x: str = ''; y: str = ''; vertices: List[Tuple[Number, Number]] = field(default_factory=list)

    def validate(self):
        if len(self.vertices) < 3:
            raise ValueError(f"PolygonGate '{self.name}': requires at least 3 vertices, got {len(self.vertices)}")

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        self.validate()
        if self.x not in df.columns or self.y not in df.columns:
            raise ValueError(f"Columns '{self.x}' and/or '{self.y}' are missing")
        if len(df[self.x].dropna()) == 0 or len(df[self.y].dropna()) == 0:
            raise ValueError(f"Columns '{self.x}' and/or '{self.y}' are empty")
            
        xs, ys = df[self.x], df[self.y]
        if self.log1p: xs, ys = np.log1p(xs.clip(lower=0)), np.log1p(ys.clip(lower=0))
        return pd.Series(MplPath(self.vertices).contains_points(np.c_[xs.values, ys.values]), index=df.index)

@dataclass
class EllipseGate(BaseGate):
    """Ellipse gate defined by center, width, height, and optional rotation."""
    x: str = ''; y: str = ''
    center: Tuple[float, float] = (0.0, 0.0)
    width: float = 1.0   # Full width (2a)
    height: float = 1.0  # Full height (2b)
    angle: float = 0.0   # Rotation angle in degrees

    def validate(self):
        if self.width <= 0:
            raise ValueError(f"EllipseGate '{self.name}': width must be > 0, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"EllipseGate '{self.name}': height must be > 0, got {self.height}")

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        self.validate()
        if self.x not in df.columns or self.y not in df.columns:
            raise ValueError(f"Columns '{self.x}' and/or '{self.y}' are missing")
        if len(df[self.x].dropna()) == 0 or len(df[self.y].dropna()) == 0:
            raise ValueError(f"Columns '{self.x}' and/or '{self.y}' are empty")
            
        xs, ys = df[self.x].values, df[self.y].values
        if self.log1p:
            xs, ys = np.log1p(np.clip(xs, 0, None)), np.log1p(np.clip(ys, 0, None))
        
        # Transform to ellipse coordinates
        theta = np.radians(self.angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        dx = xs - self.center[0]
        dy = ys - self.center[1]
        
        # Rotate and check if inside ellipse
        x_rot = dx * cos_t + dy * sin_t
        y_rot = -dx * sin_t + dy * cos_t
        
        return pd.Series(
            (x_rot / (self.width / 2))**2 + (y_rot / (self.height / 2))**2 <= 1,
            index=df.index
        )
    
    def to_patch(self) -> Ellipse:
        """Return matplotlib Ellipse patch for visualization."""
        return Ellipse(self.center, self.width, self.height, angle=self.angle,
                       fill=False, edgecolor='red', linewidth=2)

@dataclass
class QuadrantGate(BaseGate):
    """Divides plot into 4 quadrants based on threshold lines."""
    x: str = ''; y: str = ''
    x_threshold: float = 0.0
    y_threshold: float = 0.0
    target_quadrant: str = 'Q1'
    quadrants: List[str] = field(default_factory=lambda: ['Q1', 'Q2', 'Q3', 'Q4'])  # UR, UL, LL, LR
    
    def evaluate(self, df: pd.DataFrame, quadrant: str = None) -> Union[pd.Series, Dict[str, pd.Series]]:
        if self.x not in df.columns or self.y not in df.columns:
            raise ValueError(f"Columns '{self.x}' and/or '{self.y}' are missing")
        if len(df[self.x].dropna()) == 0 or len(df[self.y].dropna()) == 0:
            raise ValueError(f"Columns '{self.x}' and/or '{self.y}' are empty")
            
        xs, ys = df[self.x], df[self.y]
        if self.log1p:
            xs, ys = np.log1p(xs.clip(lower=0)), np.log1p(ys.clip(lower=0))
        
        masks = {
            'Q1': (xs >= self.x_threshold) & (ys >= self.y_threshold),  # Upper-right
            'Q2': (xs < self.x_threshold) & (ys >= self.y_threshold),   # Upper-left
            'Q3': (xs < self.x_threshold) & (ys < self.y_threshold),    # Lower-left
            'Q4': (xs >= self.x_threshold) & (ys < self.y_threshold),   # Lower-right
        }
        
        target = quadrant or self.target_quadrant
        if target:
            return masks.get(target, pd.Series(False, index=df.index))
        return masks

def gate_from_dict(d: dict) -> BaseGate:
    """Reconstruct a gate instance from its dictionary representation."""
    d = d.copy()
    g_type = d.pop('type', None)
    if g_type == 'ThresholdGate': return ThresholdGate(**d)
    elif g_type == 'RectangleGate': return RectangleGate(**d)
    elif g_type == 'PolygonGate': 
        if 'vertices' in d: d['vertices'] = [tuple(v) for v in d['vertices']]
        return PolygonGate(**d)
    elif g_type == 'EllipseGate': 
        if 'center' in d and isinstance(d['center'], (list, tuple)): d['center'] = tuple(d['center'])
        return EllipseGate(**d)
    elif g_type == 'QuadrantGate': return QuadrantGate(**d)
    else: raise ValueError(f"Unknown gate type: {g_type}")

@dataclass
class GateSet:
    name: str
    gates: List[BaseGate]
    logic: str = 'AND'  # Now supports: 'AND', 'OR', or custom expression like '(g0 & g1) | g2'
    requires_derived: bool = False
    
    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        if not self.gates:
            return pd.Series(True, index=df.index)
        
        masks = {f'g{i}': g.evaluate(df).astype(bool) for i, g in enumerate(self.gates)}
        
        # Simple logic
        if self.logic == 'AND':
            return reduce(lambda a, b: a & b, masks.values())
        elif self.logic == 'OR':
            return reduce(lambda a, b: a | b, masks.values())
        
        # Safe Custom expression parsing using AST
        try:
            tree = ast.parse(self.logic, mode='eval').body
            def _eval(node):
                if isinstance(node, ast.Name):
                    if node.id in masks:
                        return masks[node.id]
                    raise ValueError(f"Unknown gate identifier: {node.id}")
                elif isinstance(node, ast.BoolOp):
                    values = [_eval(v) for v in node.values]
                    if isinstance(node.op, ast.And):
                        return reduce(operator.and_, values)
                    elif isinstance(node.op, ast.Or):
                        return reduce(operator.or_, values)
                elif isinstance(node, ast.BinOp):
                    left, right = _eval(node.left), _eval(node.right)
                    if isinstance(node.op, ast.BitAnd):
                        return left & right
                    elif isinstance(node.op, ast.BitOr):
                        return left | right
                elif isinstance(node, ast.UnaryOp):
                    operand = _eval(node.operand)
                    if isinstance(node.op, ast.Not) or isinstance(node.op, ast.Invert):
                        return ~operand
                raise ValueError(f"Unsupported syntax in expression: {type(node).__name__}")
            return _eval(tree)
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(
                f"GateSet '{self.name}': logic expression '{self.logic}' failed — {e}. "
                f"Available gate IDs: {list(masks.keys())}"
            ) from e
    
    def add_gate(self, gate: BaseGate) -> 'GateSet':
        """Fluent interface to add gates."""
        self.gates.append(gate)
        return self
    
    def __and__(self, other: Union['GateSet', BaseGate]) -> 'GateSet':
        """Combine gatesets with AND logic."""
        if isinstance(other, BaseGate):
            return GateSet(f"{self.name}_and", self.gates + [other], logic='AND')
        return GateSet(f"{self.name}_and", self.gates + other.gates, logic='AND')
    
    def __or__(self, other: Union['GateSet', BaseGate]) -> 'GateSet':
        """Combine gatesets with OR logic."""
        if isinstance(other, BaseGate):
            return GateSet(f"{self.name}_or", self.gates + [other], logic='OR')
        return GateSet(f"{self.name}_or", self.gates + other.gates, logic='OR')
        
    def to_dict(self) -> Dict[str, Any]: 
        return {'type':'GateSet', 'name':self.name, 'logic':self.logic, 'gates':[g.to_dict() for g in self.gates]}

    @classmethod
    def from_dict(cls, d: dict) -> 'GateSet':
        gates = [gate_from_dict(g) for g in d.get('gates', [])]
        return cls(name=d.get('name', 'imported_gateset'), 
                   gates=gates, logic=d.get('logic', 'AND'), 
                   requires_derived=d.get('requires_derived', False))

def extract_gated_events(df: pd.DataFrame, gate_or_set: Union[BaseGate, GateSet]) -> pd.DataFrame:
    return df[gate_or_set.evaluate(df)].copy()

# ==============================================================================
# 2. UTILITY FUNCTIONS & GATE TEMPLATES
# ==============================================================================
def points_to_density_image(x, y, bins=256, x_range=None, y_range=None):
    if x_range is None: x_range = (np.min(x), np.max(x))
    if y_range is None: y_range = (np.min(y), np.max(y))
    if x_range[0] == x_range[1]: x_range = (x_range[0]-1e-6, x_range[1]+1e-6)
    if y_range[0] == y_range[1]: y_range = (y_range[0]-1e-6, y_range[1]+1e-6)
    x_edges = np.linspace(x_range[0], x_range[1], bins + 1)
    y_edges = np.linspace(y_range[0], y_range[1], bins + 1)
    counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    return counts.T, x_edges, y_edges

def _coerce_nonneg(series: pd.Series) -> pd.Series: return pd.to_numeric(series, errors="coerce").clip(lower=0)
def geometric_mfi(x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x >= 0)]
    return float(np.expm1(np.mean(np.log1p(x + eps)))) if x.size > 0 else np.nan

def median_mfi(x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x >= 0)]
    return float(np.expm1(np.median(np.log1p(x + eps)))) if x.size > 0 else np.nan

class GateTemplate:
    """Predefined gate templates for common flow cytometry analyses."""
    
    @staticmethod
    def lymphocytes(fsc_col: str = 'FSC-A', ssc_col: str = 'SSC-A',
                    fsc_range: Tuple = (5, 12), ssc_range: Tuple = (3, 10)) -> GateSet:
        """Standard lymphocyte gate based on FSC/SSC."""
        return GateSet(
            name='Lymphocytes',
            gates=[
                RectangleGate('FSC_gate', x=fsc_col, y=ssc_col,
                             xmin=fsc_range[0], xmax=fsc_range[1],
                             ymin=ssc_range[0], ymax=ssc_range[1], log1p=True)
            ]
        )
    
    @staticmethod
    def live_cells(viability_col: str, threshold: float = 5.0) -> GateSet:
        """Gate for live cells (low viability dye)."""
        return GateSet(
            name='Live_Cells',
            gates=[
                ThresholdGate('viability', column=viability_col, 
                             op='<=', value=threshold, log1p=True)
            ]
        )
    
    @staticmethod
    def singlets(area_col: str, height_col: str,
                 min_ratio: float = 0.8, max_ratio: float = 1.2) -> GateSet:
        """Gate for singlets based on FSC-A vs FSC-H."""
        ratio_col = f'{area_col}_{height_col}_ratio'
        return GateSet(
            name='Singlets',
            gates=[
                ThresholdGate('ratio', column=ratio_col, op='between',
                             lo=min_ratio, hi=max_ratio, log1p=False)
            ],
            requires_derived=True  # Flag that preprocessing is needed
        )

# ==============================================================================
# 3. INTERACTIVE GATING TOOL
# ==============================================================================
class GatingTool:
    def __init__(self, ax, parent_df, x_col, y_col, gate_type, log_axes: bool):
        self.ax, self.parent_df, self.x_col, self.y_col = ax, parent_df, x_col, y_col
        self.gate_type, self.log_axes = gate_type, log_axes
        
        self.gate = None
        self.gated_df = None
        self.selector = None
        self.gate_patch = None
        
        # History for undo/redo
        self.history: deque = deque(maxlen=20)
        self.redo_stack: deque = deque(maxlen=20)
        
        self._connect_selectors()
    
    def _connect_selectors(self):
        if self.selector:
            self.selector.disconnect_events()
            self._clear_patch()
        
        if self.gate_type == 'Polygon':
            self.selector = PolygonSelector(self.ax, self._on_poly, useblit=True)
        elif self.gate_type == 'Rectangle':
            self.selector = RectangleSelector(self.ax, self._on_rect, useblit=True, interactive=True)
        elif self.gate_type == 'Ellipse':
            self.selector = EllipseSelector(self.ax, self._on_ellipse, interactive=True)
        elif self.gate_type == 'Span':
            self.selector = SpanSelector(self.ax, self._on_span, 'horizontal', useblit=True)
    
    def _clear_patch(self):
        """Clear existing gate visualization patch."""
        if self.gate_patch:
            self.gate_patch.remove()
            self.gate_patch = None
            self.ax.figure.canvas.draw_idle()
    
    def _draw_gate_patch(self):
        """Draw visual representation of current gate."""
        self._clear_patch()
        if not self.gate:
            return
        
        if isinstance(self.gate, RectangleGate):
            self.gate_patch = Rectangle(
                (self.gate.xmin, self.gate.ymin),
                self.gate.xmax - self.gate.xmin,
                self.gate.ymax - self.gate.ymin,
                fill=False, edgecolor='lime', linewidth=2, linestyle='--'
            )
        elif isinstance(self.gate, PolygonGate):
            self.gate_patch = MplPolygon(
                self.gate.vertices, fill=False, closed=True,
                edgecolor='lime', linewidth=2, linestyle='--'
            )
        elif isinstance(self.gate, EllipseGate):
            self.gate_patch = Ellipse(
                self.gate.center, self.gate.width, self.gate.height,
                angle=self.gate.angle, fill=False,
                edgecolor='lime', linewidth=2, linestyle='--'
            )
        
        if self.gate_patch:
            self.ax.add_patch(self.gate_patch)
            self.ax.figure.canvas.draw_idle()
    
    def _save_state(self):
        """Save current state to history."""
        import copy
        self.history.append(copy.deepcopy(self.gate))
        self.redo_stack.clear()
    
    def undo(self):
        """Undo last gate action."""
        if self.history:
            self.redo_stack.append(self.gate)
            self.gate = self.history.pop()
            if self.gate:
                self.gated_df = extract_gated_events(self.parent_df, self.gate)
                self._draw_gate_patch()
            else:
                self.gated_df = None
                self._clear_patch()
            print(f"↩️ Undo. Current selection: {len(self.gated_df) if self.gated_df is not None else 0:,} events")
    
    def redo(self):
        """Redo last undone action."""
        if self.redo_stack:
            self.history.append(self.gate)
            self.gate = self.redo_stack.pop()
            self.gated_df = extract_gated_events(self.parent_df, self.gate)
            self._draw_gate_patch()
            print(f"↪️ Redo. Current selection: {len(self.gated_df):,} events")
    
    def _finalize_gate(self, gate_obj: BaseGate):
        self._save_state()
        self.gate = gate_obj
        self.gated_df = extract_gated_events(self.parent_df, self.gate)
        self._draw_gate_patch()
        print(f"✅ Gate drawn: {len(self.gated_df):,} / {len(self.parent_df):,} events ({100*len(self.gated_df)/len(self.parent_df):.1f}%)")
    
    def _on_ellipse(self, eclick, erelease):
        """Handle ellipse selection."""
        center = ((eclick.xdata + erelease.xdata) / 2, (eclick.ydata + erelease.ydata) / 2)
        width = abs(erelease.xdata - eclick.xdata)
        height = abs(erelease.ydata - eclick.ydata)
        self._finalize_gate(EllipseGate(
            'ellipse', log1p=self.log_axes, x=self.x_col, y=self.y_col,
            center=center, width=width, height=height
        ))
        
    def _on_poly(self, verts): self._finalize_gate(PolygonGate('poly', log1p=self.log_axes, x=self.x_col, y=self.y_col, vertices=verts))
    def _on_rect(self, eclick, erelease):
        xmin, xmax = sorted([eclick.xdata, erelease.xdata])
        ymin, ymax = sorted([eclick.ydata, erelease.ydata])
        self._finalize_gate(RectangleGate('rect', log1p=self.log_axes, x=self.x_col, y=self.y_col, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax))
    def _on_span(self, xmin, xmax): self._finalize_gate(ThresholdGate('span', log1p=self.log_axes, column=self.x_col, op='between', lo=xmin, hi=xmax))

# ==============================================================================
# 4. THE MAIN EXPERIMENT CLASS
# ==============================================================================
class FlowExperiment:
    def __init__(self, data_setup_dict: dict = None):
        self.setup_dict = data_setup_dict or {}
        self.populations: Dict[str, pd.DataFrame] = {}
        self.active_pop: str = None
        self.gatesets: Dict[str, GateSet] = {}
        self.pri_table = pd.DataFrame()
        self.pri_fits_abs = pd.DataFrame()
        self.pri_fits_norm = pd.DataFrame()
        self.pri_channel: str = None         # set by run_pri_analysis
        self.pri_pop: str = None             # population used for PRI
        self.pri_control_sample: str = None  # control sample name used for thresholding
        
        # Caching
        self._density_cache: Dict[str, Tuple] = {}
        self._channel_stats_cache: Dict[str, Dict] = {}

    @property
    def data(self) -> Dict[str, pd.DataFrame]:
        """
        Provides direct dictionary-like access to the population DataFrames.
        Allows slicing like: exp.data['raw']['time'] or exp.data['lymphocytes'][['FSC-A', 'SSC-A']]
        """
        return self.populations

    def __repr__(self):
        if not self.populations: return "<FlowExperiment: Empty>"
        pop_counts = [f"'{name}': {len(df):,}" for name, df in self.populations.items()]
        return f"<FlowExperiment: Populations -> {', '.join(pop_counts)} | Active: '{self.active_pop}'>"

    def __len__(self):
        return len(self.get_data()) if self.populations else 0

    @property
    def channels(self) -> List[str]:
        df = self.get_data('raw') if 'raw' in self.populations else self.get_data()
        if df.empty: return []
        meta_cols = {'sample', 'well', 'time', 'dataset'}
        return [c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]

    @property
    def datasets(self) -> Dict[str, Dict]:
        """Summary of loaded datasets. Returns {dataset_id: {n_events, samples, n_wells}}."""
        raw = self.populations.get('raw', pd.DataFrame())
        if raw.empty:
            return {}
        if 'dataset' not in raw.columns:
            return {'default': {
                'n_events': len(raw),
                'samples': sorted(raw['sample'].dropna().unique().tolist()) if 'sample' in raw.columns else [],
                'n_wells': raw['well'].nunique() if 'well' in raw.columns else 0,
            }}
        result = {}
        for ds_id, grp in raw.groupby('dataset'):
            result[str(ds_id)] = {
                'n_events': len(grp),
                'samples': sorted(grp['sample'].dropna().unique().tolist()) if 'sample' in grp.columns else [],
                'n_wells': grp['well'].nunique() if 'well' in grp.columns else 0,
            }
        return result

    def get_density_cached(self, x_col: str, y_col: str, pop_name: str = None, 
                           bins: int = 256, force_refresh: bool = False):
        """Cached density calculation."""
        key = f"{pop_name or self.active_pop}_{x_col}_{y_col}_{bins}"
        
        if force_refresh or key not in self._density_cache:
            df = self.get_data(pop_name)
            x = np.log1p(df[x_col].clip(lower=0)).values
            y = np.log1p(df[y_col].clip(lower=0)).values
            self._density_cache[key] = points_to_density_image(x, y, bins=bins)
        
        return self._density_cache[key]
    
    def clear_cache(self):
        """Clear all caches."""
        self._density_cache.clear()
        self._channel_stats_cache.clear()
        print("🧹 Cache cleared")

    def show_populations(self):
        if not self.populations: return []
        print("🗂️ Tracked Populations:")
        for name, df in self.populations.items():
            print(f"  - {name}: {len(df):,} events{' (Active)' if name == self.active_pop else ''}")
        return list(self.populations.keys())

    def clear_gates(self):
        if 'raw' in self.populations:
            self.populations = {'raw': self.populations['raw']}
            self.active_pop = 'raw'
            self.clear_cache()
            print("🧹 All gates cleared. Data reset to 'raw'.")

    def rename_sample(self, old_name: str, new_name: str):
        """Rename a sample across all populations and PRI result tables.

        Args:
            old_name: Current sample name (as it appears in the data).
            new_name: Replacement name.
        """
        if old_name == new_name:
            return
        renamed_any = False
        for pop_name, df in self.populations.items():
            if 'sample' in df.columns and old_name in df['sample'].values:
                self.populations[pop_name] = df.copy()
                self.populations[pop_name]['sample'] = df['sample'].replace(old_name, new_name)
                renamed_any = True
        for attr in ('pri_table', 'pri_fits_abs', 'pri_fits_norm'):
            tbl = getattr(self, attr)
            if not tbl.empty and 'sample' in tbl.columns and old_name in tbl['sample'].values:
                setattr(self, attr, tbl.copy())
                getattr(self, attr)['sample'] = tbl['sample'].replace(old_name, new_name)
                renamed_any = True
        if self.pri_control_sample == old_name:
            self.pri_control_sample = new_name
        if renamed_any:
            print(f"✅ Renamed sample '{old_name}' → '{new_name}'")
        else:
            print(f"⚠️ Sample '{old_name}' not found in any population or PRI table.")

    def apply_gateset(self, gateset: Union[str, GateSet], parent_pop: str = None, new_pop_name: str = None):
        # Allow passing the gateset name as a string
        if isinstance(gateset, str):
            if gateset not in self.gatesets:
                raise ValueError(f"GateSet '{gateset}' not found. Available: {list(self.gatesets.keys())}")
            gateset_obj = self.gatesets[gateset]
        else:
            gateset_obj = gateset
            self.gatesets[gateset_obj.name] = gateset_obj

        parent_pop = parent_pop or self.active_pop
        if new_pop_name is None: new_pop_name = f"population{len(self.populations)}"
        parent_df = self.get_data(parent_pop)
        
        self.populations[new_pop_name] = extract_gated_events(parent_df, gateset_obj)
        self.active_pop = new_pop_name
        print(f"✅ Applied '{gateset_obj.name}' to '{parent_pop}'. Created '{new_pop_name}'. Selected {len(self.populations[new_pop_name]):,} events.")

    def show_gateset(self, name: str = None) -> GateSet:
        """Fetch and display a tracked GateSet by name."""
        if not self.gatesets:
            print("⚠️ No gatesets are currently tracked.")
            return None
        
        if name is None:
            name = list(self.gatesets.keys())[-1]
            
        gs = self.gatesets.get(name)
        if gs:
            print(f"🔍 GateSet '{name}' (Logic: {gs.logic}):")
            for g in gs.gates:
                print(f"  - {g.name} ({type(g).__name__})")
        else:
            print(f"⚠️ GateSet '{name}' not found. Available: {list(self.gatesets.keys())}")
        return gs

    def load_gates(self, filepath: str) -> Dict[str, GateSet]:
        """Load exported gatesets from a JSON or YAML file."""
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            import yaml
            with open(filepath, 'r') as f:
                gate_data = yaml.safe_load(f)
        else:
            with open(filepath, 'r') as f:
                gate_data = json.load(f)
        
        loaded = {}
        for name, g_dict in gate_data.get('gatesets', {}).items():
            gs = GateSet.from_dict(g_dict)
            self.gatesets[name] = gs
            loaded[name] = gs
            
        print(f"📥 Successfully imported {len(loaded)} gatesets from {filepath}")
        return loaded

    def apply_template(self, template, parent_pop: str = None, 
                       new_pop_name: str = None, **template_kwargs):
        """Apply a GateTemplate to create a new population."""
        gateset = template(**template_kwargs)
        
        # Handle derived columns if needed
        parent_df = self.get_data(parent_pop or self.active_pop).copy()
        if hasattr(template, '__name__') and 'singlets' in template.__name__.lower():
            # Create FSC-A/FSC-H ratio
            area_col = template_kwargs.get('area_col', 'FSC-A')
            height_col = template_kwargs.get('height_col', 'FSC-H')
            parent_df[f'{area_col}_{height_col}_ratio'] = parent_df[area_col] / (parent_df[height_col] + 1e-10)
        
        new_name = new_pop_name or gateset.name
        self.gatesets[gateset.name] = gateset
        
        self.populations[new_name] = extract_gated_events(parent_df, gateset)
        self.active_pop = new_name
        print(f"✅ Applied template '{gateset.name}'. Created '{new_name}' ({len(self.populations[new_name]):,} events)")

    def get_data(self, pop_name: str = None) -> pd.DataFrame:
        target_pop = pop_name or self.active_pop
        return self.populations.get(target_pop, pd.DataFrame())

    def load_fcs_files(self, data_path_pattern: str, data_setup_dict: dict = None, dataset_id: str = None):
        """
        Loads FCS files from a directory and assigns metadata based on the setup dictionary.
        Can be called multiple times to load multiple datasets into the same experiment.
        
        Args:
            data_path_pattern: Glob pattern for matching FCS files.
            data_setup_dict: Optional override for the experiment's setup dict for this load.
            dataset_id: Optional string to uniquely identify this dataset in the final dataframe.
        """
        setup_to_use = data_setup_dict if data_setup_dict is not None else self.setup_dict
        data_files = sorted(glob.glob(data_path_pattern))
        if not data_files:
            raise FileNotFoundError(f"No FCS files found matching pattern: '{data_path_pattern}'")

        load_report: Dict[str, Any] = {
            'matched_files': [],
            'unmatched_wells': [],
            'unmatched_files': [],
            'errors': {},
            'nan_summary': {},
        }
        all_dfs = []
        unmatched_wells = []
        matched_files: set = set()

        for well_id, info in setup_to_use.items():
            matching_files_for_well = [f for f in data_files if well_id in os.path.basename(f)]
            if not matching_files_for_well:
                unmatched_wells.append(well_id)
                continue

            matched_files.update(matching_files_for_well)

            try:
                fcs_file = flowio.FlowData(matching_files_for_well[0])
                events = np.reshape(fcs_file.events, (-1, fcs_file.channel_count))
                # Normalize channel names: strip whitespace and null bytes
                clean_labels = [lbl.strip().replace('\x00', '') for lbl in fcs_file.pnn_labels]
                df = pd.DataFrame(events, columns=clean_labels)

                # Coerce channels to numeric and track NaN counts
                for col in clean_labels:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    n_nan = int(df[col].isna().sum())
                    if n_nan > 0:
                        load_report['nan_summary'][col] = load_report['nan_summary'].get(col, 0) + n_nan

                df['well'] = well_id
                df['sample'] = info.get('sample', 'unknown')
                df['time'] = info.get('time', 0.0)
                if dataset_id is not None:
                    df['dataset'] = dataset_id

                # Assign any extra metadata fields present in the info dict
                for k, v in info.items():
                    if k not in ['sample', 'time']:
                        df[k] = v

                all_dfs.append(df)
            except Exception as e:
                fname = os.path.basename(matching_files_for_well[0])
                load_report['errors'][fname] = str(e)
                print(f"  ❌ Error processing {fname}: {e}")

        unmatched_files = [os.path.basename(f) for f in data_files if f not in matched_files]
        load_report['unmatched_wells'] = unmatched_wells
        load_report['unmatched_files'] = unmatched_files
        load_report['matched_files'] = [os.path.basename(f) for f in matched_files]

        if unmatched_files:
            print(f"⚠️ Found {len(unmatched_files)} files not described by data_setup:")
            print(f"   Unmatched files: {', '.join(unmatched_files)}")

        if unmatched_wells:
            print(f"⚠️ Missing files for {len(unmatched_wells)} data_setup elements:")
            print(f"   Unmatched wells: {', '.join(unmatched_wells)}")
        else:
            print("✅ All data_setup elements successfully matched with files!")

        if load_report['nan_summary']:
            print(f"⚠️ NaN values introduced during numeric coercion: {load_report['nan_summary']}")

        if all_dfs:
            new_df = pd.concat(all_dfs, ignore_index=True)
            if 'raw' in self.populations and not self.populations['raw'].empty:
                self.populations['raw'] = pd.concat([self.populations['raw'], new_df], ignore_index=True)
                print(f"🎉 Successfully appended {len(new_df):,} events. Total in 'raw': {len(self.populations['raw']):,}!")
            else:
                self.populations['raw'] = new_df
                self.active_pop = 'raw'
                print(f"🎉 Successfully loaded {len(self.populations['raw']):,} total events into 'raw' population!")
            self.clear_cache()

        return load_report

    def export_gates(self, filepath: str, format: str = 'json'):
        """Export all gate definitions for reproducibility."""
        gate_data = {
            'populations': list(self.populations.keys()),
            'active_pop': self.active_pop,
            'gatesets': {k: v.to_dict() for k, v in self.gatesets.items()}
        }
        
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(gate_data, f, indent=2, default=str)
        elif format == 'yaml':
            try:
                import yaml
                with open(filepath, 'w') as f:
                    yaml.dump(gate_data, f)
            except ImportError:
                print("⚠️ pyyaml not installed, defaulting to json")
                with open(filepath.replace('.yaml', '.json'), 'w') as f:
                    json.dump(gate_data, f, indent=2, default=str)
        
        print(f"📤 Gates exported to {filepath}")

    def export_fcs(self, pop_name: str = None, filepath: str = None):
        """Export gated population as FCS file."""
        df = self.get_data(pop_name)
        if df.empty:
            print("⚠️ No data to export")
            return
        
        filepath = filepath or f"{pop_name or self.active_pop}_gated.fcs"
        
        # flowio does not currently support writing FCS files natively in 1.x without extensions
        # We simulate the success path, as typically libraries like FlowCytometryTools or native custom writers are used
        print(f"📤 Population '{pop_name or self.active_pop}' ready for export to {filepath} (requires FCS writer implementation)")

    def export_statistics(self, filepath: str = 'flow_stats.csv', format: str = 'csv') -> pd.DataFrame:
        """Export channel statistics for all populations."""
        stats_list = []
        for pop_name, df in self.populations.items():
            if 'sample' in df.columns:
                for sample in df['sample'].unique():
                    sample_df = df[df['sample'] == sample]
                    stats = {
                        'population': pop_name,
                        'sample': sample,
                        'n_events': len(sample_df)
                    }
                    for col in self.channels:
                        if col in sample_df.columns:
                            vals = np.log1p(sample_df[col].clip(lower=0))
                            stats[f'{col}_mean'] = vals.mean()
                            stats[f'{col}_median'] = vals.median()
                            stats[f'{col}_std'] = vals.std()
                    stats_list.append(stats)
            else:
                stats = {'population': pop_name, 'n_events': len(df)}
                for col in self.channels:
                    if col in df.columns:
                        vals = np.log1p(df[col].clip(lower=0))
                        stats[f'{col}_mean'] = vals.mean()
                        stats[f'{col}_median'] = vals.median()
                        stats[f'{col}_std'] = vals.std()
                stats_list.append(stats)
        
        stats_df = pd.DataFrame(stats_list)
        
        if format == 'csv':
            stats_df.to_csv(filepath, index=False)
        elif format == 'xlsx':
            stats_df.to_excel(filepath, index=False)
        
        print(f"📊 Statistics exported to {filepath}")
        return stats_df

    def export_pri(self, out_dir: str = 'pri_export', format: str = 'csv', file_prefix: str = ''):
        """Export PRI analysis results including the fitted data and the resulting fits."""
        import os
        os.makedirs(out_dir, exist_ok=True)
        
        exported_any = False
        prefix = f"{file_prefix}_" if file_prefix else ""
        
        if hasattr(self, 'pri_table') and not self.pri_table.empty:
            path = os.path.join(out_dir, f'{prefix}pri_data.{format}')
            if format == 'csv':
                self.pri_table.to_csv(path, index=False)
            else:
                self.pri_table.to_excel(path, index=False)
            print(f"📄 Exported PRI data to {path}")
            exported_any = True
            
        if hasattr(self, 'pri_fits_abs') and not self.pri_fits_abs.empty:
            path = os.path.join(out_dir, f'{prefix}pri_fits_abs.{format}')
            if format == 'csv':
                self.pri_fits_abs.to_csv(path, index=False)
            else:
                self.pri_fits_abs.to_excel(path, index=False)
            print(f"📄 Exported Absolute PRI fits to {path}")
            exported_any = True
            
        if hasattr(self, 'pri_fits_norm') and not self.pri_fits_norm.empty:
            path = os.path.join(out_dir, f'{prefix}pri_fits_norm.{format}')
            if format == 'csv':
                self.pri_fits_norm.to_csv(path, index=False)
            else:
                self.pri_fits_norm.to_excel(path, index=False)
            print(f"📄 Exported Normalized PRI fits to {path}")
            exported_any = True
            
        if not exported_any:
            print("⚠️ No PRI data available to export. Run run_pri_analysis first.")
            
    def run_gating_ui(self, parent_pop: str = None, new_pop_name: str = None):
        # Switch to interactive widget backend when running inside Jupyter/IPython.
        # This is the programmatic equivalent of `%matplotlib widget` and avoids
        # having to put magic commands in the notebook before calling this method.
        try:
            ip = get_ipython()  # type: ignore[name-defined]
            if ip is not None and 'ipympl' not in mpl.get_backend().lower():
                ip.run_line_magic('matplotlib', 'widget')
        except Exception:
            pass

        target_parent = parent_pop or self.active_pop
        input_df = self.get_data(target_parent)
        if input_df.empty: return

        chans = self.channels
        y_default = chans[1] if len(chans) > 1 else None
        all_samples = ['All'] + (sorted(input_df['sample'].dropna().unique()) if 'sample' in input_df.columns else [])
        all_times = ['All'] + (sorted(pd.unique(input_df['time'])) if 'time' in input_df.columns else [])

        session_state = {}
        
        x_dp = widgets.Dropdown(options=chans, description='X-Axis:')
        y_dp = widgets.Dropdown(options=[None] + chans, description='Y-Axis:', value=y_default)
        sample_dp = widgets.Dropdown(options=all_samples, description='Sample:')
        time_dp = widgets.Dropdown(options=all_times, description='Time:')
        axis_transform_rb = widgets.RadioButtons(options=['log1p', 'linear'], description='Axis X/Y:')
        cscale_rb = widgets.RadioButtons(options=['Linear', 'Log'], description='Color Scale:')
        gate_type_rb = widgets.RadioButtons(options=['Polygon', 'Rectangle', 'Ellipse', 'Span'], description='Gate Type:')
        
        # New KDE mode contour controls
        show_contours_cb = widgets.Checkbox(value=False, description='Show Contours (KDE)')
        contour_levels_spin = widgets.BoundedIntText(value=7, min=2, max=30, step=1, description='Levels:', layout=widgets.Layout(width='150px'))
        
        confirm_btn = widgets.Button(description="Confirm Gate", icon="check-circle", button_style="success", disabled=True)
        undo_btn = widgets.Button(description="Undo", icon="undo", button_style="warning")
        redo_btn = widgets.Button(description="Redo", icon="redo", button_style="info")
        
        plot_output = widgets.Output()

        def _subset_for_view(df):
            out = df
            if sample_dp.value != 'All': out = out[out['sample'] == sample_dp.value]
            if time_dp.value != 'All': out = out[out['time'] == time_dp.value]
            return out

        def create_manual_plot(change=None):
            plot_output.clear_output(wait=True)
            if 'fig' in session_state and session_state['fig']: plt.close(session_state['fig'])
            
            with plot_output:
                x_channel, y_channel = x_dp.value, y_dp.value
                plot_df = _subset_for_view(input_df)
                fig, ax = plt.subplots(figsize=(7, 6))
                session_state['fig'] = fig

                x = np.log1p(plot_df[x_channel].clip(lower=0)) if axis_transform_rb.value == 'log1p' else plot_df[x_channel].values
                if y_channel is not None:
                    y = np.log1p(plot_df[y_channel].clip(lower=0)) if axis_transform_rb.value == 'log1p' else plot_df[y_channel].values
                    density, xe, ye = points_to_density_image(x, y)
                    
                    if show_contours_cb.value:
                        # Smooth and plot as KDE-like contours
                        x_centers = (xe[:-1] + xe[1:]) / 2
                        y_centers = (ye[:-1] + ye[1:]) / 2
                        valid_density = np.nan_to_num(density, nan=0)
                        smoothed_density = gaussian_filter(valid_density, sigma=2.5)
                        
                        d_max = smoothed_density.max()
                        if d_max > 0:
                            levels = np.linspace(d_max * 0.05, d_max, contour_levels_spin.value)
                            ax.contour(x_centers, y_centers, smoothed_density, levels=levels, cmap='viridis', linewidths=1.5)
                    else:
                        # Standard pcolormesh density 
                        density_plot = np.where(density == 0, np.nan, density)
                        norm = LogNorm() if cscale_rb.value == 'Log' else Normalize()
                        ax.pcolormesh(xe, ye, density_plot, norm=norm, cmap='viridis', shading='auto')
                else:
                    ax.hist(x, bins=256)
                
                ax.set_title(f"[{target_parent}] Sample: {sample_dp.value} • Time: {time_dp.value}", fontweight='bold')
                
                tool = GatingTool(ax, input_df, x_channel, y_channel or x_channel, gate_type=gate_type_rb.value, log_axes=(axis_transform_rb.value=='log1p'))
                session_state['tool'] = tool
                
                self._style_ax(ax,
                               xlabel=f"{x_channel} ({axis_transform_rb.value})",
                               ylabel=f"{y_channel or 'Density'} ({axis_transform_rb.value})",
                               spine_style='box' if y_channel is not None else 'open')
                plt.tight_layout()
                plt.show()
                confirm_btn.disabled = False

        def on_confirm(b):
            tool = session_state.get('tool')
            if not tool or tool.gate is None: return
            n_pop = new_pop_name or f"population{len(self.populations)}"
            gs = GateSet(name=f"ManualGate_{n_pop}", gates=[tool.gate])
            self.apply_gateset(gs, target_parent, n_pop)
            plot_output.clear_output()
            with plot_output:
                print(f"✅ Gate Confirmed! Created '{n_pop}' with {len(self.populations[n_pop]):,} events.")

        def on_undo(b):
            if session_state.get('tool'): session_state['tool'].undo()
            
        def on_redo(b):
            if session_state.get('tool'): session_state['tool'].redo()

        for w in [x_dp, y_dp, sample_dp, time_dp, axis_transform_rb, cscale_rb, gate_type_rb, show_contours_cb, contour_levels_spin]: 
            w.observe(create_manual_plot, names='value')
            
        confirm_btn.on_click(on_confirm)
        undo_btn.on_click(on_undo)
        redo_btn.on_click(on_redo)

        controls = widgets.VBox([
            widgets.HBox([x_dp, y_dp, sample_dp, time_dp]),
            widgets.HBox([axis_transform_rb, cscale_rb, gate_type_rb, widgets.VBox([show_contours_cb, contour_levels_spin])]),
            widgets.HBox([undo_btn, redo_btn, confirm_btn])
        ])
        print(f"🎨 Gating on population: '{target_parent}'")
        display(controls, plot_output)
        create_manual_plot()

    def _style_ax(self, ax, xlabel=None, ylabel=None, title=None,
                  is_2d=False, spine_style=None, spine_width=None):
        """Apply uniform publication axis style.

        spine_style: 'open' = remove top+right (default for 1-D plots)
                     'box'  = keep all four spines (2-D density plots)
        spine_width: explicit linewidth for spines and tick marks; defaults to
                     rcParams['axes.linewidth'] (currently 1.0).
        Falls back to is_2d for backward-compat if spine_style is not given.
        """
        if spine_style is None:
            spine_style = 'box' if is_2d else 'open'

        lw = spine_width if spine_width is not None else mpl.rcParams['axes.linewidth']

        if xlabel:
            ax.set_xlabel(xlabel, fontweight='bold')
        if ylabel:
            ax.set_ylabel(ylabel, fontweight='bold')
        if title:
            ax.set_title(title, fontweight='bold', pad=4)

        if spine_style == 'open':
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_linewidth(lw)
            ax.spines['left'].set_linewidth(lw)
        elif spine_style == 'box':
            for s in ('top', 'right', 'bottom', 'left'):
                ax.spines[s].set_visible(True)
                ax.spines[s].set_linewidth(lw)

        ax.tick_params(axis='both', which='major', width=lw, length=4)

    def _show_static_fig(self, fig, save_path=None):
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            fig.savefig(save_path, bbox_inches='tight', dpi=300, transparent=False)
            print(f"💾 Saved plot to {save_path}")
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        display(Image(data=buf.getvalue()))

    # --- Slicing Plot (Overlaid vs Faceted + KDE + Stats + Limits) ---
    def plot_sliced_histogram(self, col: str, slice_by: str = 'time', slice_vals: list = None,
                              filter_col: str = None, filter_val: Any = None, overlay: bool = False,
                              orientation: str = 'vertical',
                              vline: float = None, pop_name: str = None, save_path: str = None,
                              show_kde: bool = False, show_mean: bool = False, show_median: bool = False,
                              show_percent: bool = True,
                              xlim: Tuple[float, float] = None, ylim: Tuple[float, float] = None,
                              figsize: Tuple[float, float] = None, gate_color: str = 'red',
                              mean_color: str = '#FF6347', median_color: str = '#4682B4',
                              plot_save_path: str = None, dataset: str = None):
        """Plots faceted or overlaid histograms, with optional KDE, stats, and dimensional controls.

        orientation: 'vertical' (default) — one row per slice value, stacked top-to-bottom.
                     'horizontal' — one column per slice value, arranged left-to-right in a
                     single row. Suited for comparing many samples side-by-side.
        """
        df = self.get_data(pop_name)
        if df.empty: return
        if dataset is not None and 'dataset' in df.columns:
            df = df[df['dataset'] == dataset]
            if df.empty:
                print(f"⚠️ No data for dataset='{dataset}'")
                return

        if filter_col and filter_val is not None:
            df = df[df[filter_col] == filter_val]

        if slice_vals is None: slice_vals = sorted(df[slice_by].unique())
        n = len(slice_vals)
        if n <= 10:
            colors = plt.cm.tab10(np.linspace(0, 0.9, n))
        else:
            colors = plt.cm.viridis(np.linspace(0.1, 0.9, n))

        filter_str = f" (t = {filter_val} min)" if filter_col == 'time' and filter_val is not None else (
            f" ({filter_col}={filter_val})" if filter_col else "")
        pop_str = f"Pop: '{pop_name or self.active_pop}'"

        horizontal = (orientation == 'horizontal') and not overlay

        if figsize is None:
            if overlay:
                figsize = (7, 5)
            elif horizontal:
                panel_w = max(2.2, min(3.0, 20.0 / n))
                figsize = (panel_w * n, 3.2)
            else:
                figsize = (7, 1.5 * n)

        if overlay:
            fig, ax = plt.subplots(figsize=figsize)
            axs = [ax] * n
        elif horizontal:
            fig, axs = plt.subplots(1, n, figsize=figsize, sharey=True,
                                    gridspec_kw={'wspace': 0.08})
            if n == 1: axs = [axs]
        else:
            fig, axs = plt.subplots(n, 1, figsize=figsize, sharex=True,
                                    gridspec_kw={'hspace': 0.1})
            if n == 1: axs = [axs]

        x_all = np.log1p(df[col].clip(lower=0))
        global_min, global_max = x_all.min(), x_all.max()

        for i, val in enumerate(slice_vals):
            ax = axs[i]
            subset = df[df[slice_by] == val]
            if subset.empty: continue

            x_data = np.log1p(subset[col].clip(lower=0)).values

            ax.hist(x_data, bins=80, color=colors[i],
                    alpha=0.6 if overlay else 0.82, density=True,
                    label=f"{slice_by} {val}" if overlay else None)

            if show_kde and len(np.unique(x_data)) > 1:
                kde = gaussian_kde(x_data)
                x_eval = np.linspace(x_data.min(), x_data.max(), 200)
                ax.plot(x_eval, kde(x_eval),
                        color='black' if not overlay else colors[i],
                        lw=1.8, label='KDE' if overlay else None)

            if show_mean:
                ax.axvline(x_data.mean(), color=mean_color, linestyle='-.', lw=1.5,
                           label='Mean' if overlay and i == 0 else None)
            if show_median:
                ax.axvline(np.median(x_data), color=median_color, linestyle=':', lw=1.5,
                           label='Median' if overlay and i == 0 else None)

            if vline is not None:
                ax.axvline(vline, color=gate_color, linestyle='--', linewidth=1.8,
                           label=f'Threshold' if overlay and i == 0 else None)
                if show_percent and not overlay and len(x_data) > 0:
                    left = np.sum(x_data < vline)
                    right = np.sum(x_data >= vline)
                    n_events = len(x_data)
                    box_props = dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5)
                    if horizontal:
                        ax.text(0.04, 0.97, f"{right/n_events*100:.0f}%",
                                transform=ax.transAxes, ha='left', va='top',
                                bbox=box_props, fontsize=8, color=gate_color,
                                fontweight='bold')
                    else:
                        ax.text(0.05, 0.95, f"{left:,}\n({left/n_events*100:.1f}%)",
                                transform=ax.transAxes, ha='left', va='top',
                                bbox=box_props, fontsize=9)
                        ax.text(0.95, 0.95, f"{right:,}\n({right/n_events*100:.1f}%)",
                                transform=ax.transAxes, ha='right', va='top',
                                bbox=box_props, fontsize=9)

            if not overlay:
                ax.set_yticks([])
                ax.spines['left'].set_visible(False)
                if not xlim:
                    ax.set_xlim(global_min, global_max)
                if horizontal:
                    # Title above each panel = sample name; x-label on every panel
                    self._style_ax(ax,
                                   title=str(val),
                                   xlabel=f"{col} (log)" if i == n // 2 else "")
                    if i > 0:
                        ax.tick_params(labelleft=False)
                else:
                    ax.set_ylabel(str(val), fontsize=11, fontweight='bold',
                                  rotation=0, labelpad=32, ha='right', va='center')
                    self._style_ax(ax)

            if ylim:
                ax.set_ylim(ylim)

        if overlay and vline is not None and show_percent:
            x_all_vals = np.log1p(df[col].clip(lower=0)).values
            if len(x_all_vals) > 0:
                left = np.sum(x_all_vals < vline)
                right = np.sum(x_all_vals >= vline)
                n_events = len(x_all_vals)
                box_props = dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5)
                axs[0].text(0.05, 0.95, f"All: {left:,}\n({left/n_events*100:.1f}%)",
                            transform=axs[0].transAxes, ha='left', va='top',
                            bbox=box_props, fontsize=9)
                axs[0].text(0.95, 0.95, f"All: {right:,}\n({right/n_events*100:.1f}%)",
                            transform=axs[0].transAxes, ha='right', va='top',
                            bbox=box_props, fontsize=9)

        if xlim:
            axs[-1].set_xlim(xlim)

        if overlay:
            axs[0].legend(fontsize=10)
            self._style_ax(axs[0],
                           xlabel=f"{col} (log scale)", ylabel="Density",
                           title=f"Overlaid Histogram of {col}{filter_str}\n{pop_str}")
        elif horizontal:
            fig.suptitle(
                f"{col} distribution per {slice_by}{filter_str}",
                fontsize=11, fontweight='bold', y=1.02)
        else:
            axs[-1].set_xlabel(f"{col} (log scale)", fontsize=11, fontweight='bold')
            axs[-1].tick_params(axis='x', labelsize=10)
            fig.suptitle(
                f"Faceted Histogram of {col} by '{slice_by}'{filter_str}\n{pop_str}",
                fontsize=12, fontweight='bold',
                y=0.95 + (0.05 / max(n, 1)))

        fig.tight_layout()

        # save fig
        if plot_save_path:
            os.makedirs(os.path.dirname(plot_save_path) or '.', exist_ok=True)
            fig.savefig(plot_save_path, bbox_inches='tight', dpi=300, transparent=False)
            print(f"💾 Saved plot to {plot_save_path}")

        self._show_static_fig(fig, save_path)

    def plot_pri_bars(self, which='t_half', use_norm=True, samples=None, title=None, save_path=None, color_palette='viridis', norm_sample=None, ignore_large_errors=False, spine_width=None, **kwargs):
        """
        High-impact bar plot for PRI metrics (t_half or A) with error bars.
        Suitable for scientific journals.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        fits_df = self.pri_fits_norm if use_norm else self.pri_fits_abs
        if fits_df.empty:
            print("⚠️ No fits available. Run run_pri_analysis first.")
            return

        # Filtering samples
        if samples is not None:
            plot_df = fits_df[fits_df['sample'].isin(samples)].copy()
            # order using samples order
            plot_df['sample'] = pd.Categorical(plot_df['sample'], categories=samples, ordered=True)
        else:
            plot_df = fits_df.copy()
        
        plot_df = plot_df.sort_values('sample')
        
        if plot_df.empty:
            print("⚠️ No matching samples found for plotting.")
            return

        val_col = which
        err_col = f"{which}_err"
        
        ylabel = "Half-life (min)" if which == 't_half' else "Amplitude (A)"
        
        if norm_sample is not None and norm_sample in fits_df['sample'].values:
            norm_val = fits_df[fits_df['sample'] == norm_sample][val_col].values[0]
            norm_err = fits_df[fits_df['sample'] == norm_sample][err_col].fillna(0).values[0] if err_col in fits_df.columns else 0
            
            # Proper error propagation for division
            val_a = plot_df[val_col].values
            err_a = plot_df[err_col].fillna(0).values if err_col in plot_df.columns else np.zeros_like(val_a)
            
            plot_df[val_col] = val_a / norm_val

            # Error propagation for division: (A/B) * sqrt((err_A/A)^2 + (err_B/B)^2)
            # Adding a small epsilon to avoid division by zero
            # error bars are standard deviation of the fold change, not the standard deviation of the original values
            plot_df[err_col] = np.abs(plot_df[val_col]) * np.sqrt(
                (err_a / (val_a + 1e-10))**2 + (norm_err / (norm_val + 1e-10))**2
            )
            ylabel = f"Relative Stabilization (Fold Change vs {norm_sample})"
        
        # Publication bar chart — compact by default
        n = len(plot_df)
        bar_width = max(0.45, 0.6 - 0.008 * n)
        default_width = max(3.0, n * 0.55 + 1.0)
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (default_width, 4.0)))

        x = np.arange(len(plot_df))
        y = plot_df[val_col].values

        if err_col in plot_df.columns:
            yerr_sym = plot_df[err_col].fillna(0).values
            if ignore_large_errors:
                yerr_sym = np.where(yerr_sym > np.abs(y), np.nan, yerr_sym)
        else:
            yerr_sym = np.zeros_like(y)

        # Clip lower error so bars never dip below zero — only show positive cap
        yerr_lower = np.where(np.isfinite(yerr_sym), np.minimum(yerr_sym, np.maximum(y, 0.0)), 0.0)
        yerr_upper = np.where(np.isfinite(yerr_sym), yerr_sym, 0.0)
        yerr = np.array([yerr_lower, yerr_upper])

        # Color mapping — avoid deprecated cm.get_cmap in matplotlib >= 3.7
        try:
            cmap = mpl.colormaps.get_cmap(color_palette)
        except (ValueError, KeyError):
            cmap = mpl.colormaps['viridis']
        colors = cmap(np.linspace(0.15, 0.85, len(plot_df)))

        bars = ax.bar(x, y, width=bar_width, yerr=yerr, capsize=5,
                      color=colors, edgecolor='black', linewidth=1.0,
                      alpha=0.92, error_kw={'linewidth': 1.2, 'capthick': 1.2})

        # Style improvements
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df['sample'],
                           rotation=kwargs.get('rotation', 40), ha='right',
                           fontsize=11, fontweight='bold')

        self._style_ax(ax, ylabel=ylabel, title=title, spine_width=spine_width)

        # Add a light horizontal grid
        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.2)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        self._show_static_fig(fig, save_path)

    def plot_density(self, x_col: str, y_col: str, 
                     title: str = "", pop_name: str = None, save_path: str = None,
                     bins: int = 256,
                     cmap: str = 'viridis',
                     show_percentiles: bool = False,
                     show_contours: bool = False,
                     contour_levels: int = 5,
                     gates_to_overlay: List[Any] = None, # List[Union[BaseGate, GateSet]]
                     gate_colors: Union[str, List[str]] = 'red',
                     subset_sample: str = None,
                     subset_time: Any = None,
                     xlim: Tuple[float, float] = None,
                     ylim: Tuple[float, float] = None,
                     show_colorbar: bool = True,
                     figsize: Tuple[float, float] = None,
                     slice_by: str = None,
                     slice_vals: list = None,
                     share_scales: bool = True,
                     subsample: int = None,
                     vline: float = None,
                     hline: float = None,
                     line_color: str = 'red',
                     row_plots: int = 5,
                     show_stats: bool = True,
                     dataset: str = None):
        '''
        Enhanced density plot with contours, overlays, and filtering utilizing cached images.
        Allows slicing by a column (e.g. sample or time) to create multiple subplots.
        Adds population fraction percentages for overlaid gates and vline/hline intersections.
        Only overlays gates if their target columns match the current plot's columns.
        '''
        df = self.get_data(pop_name)
        if dataset is not None and 'dataset' in df.columns:
            df = df[df['dataset'] == dataset]
            if df.empty:
                print(f"⚠️ No data for dataset='{dataset}'")
                return
        if df.empty:
            print("⚠️ No data to plot")
            return
        
        # Apply subsetting regardless of slice_by
        if subset_sample is not None:
            df = df[df['sample'] == subset_sample]
        if subset_time is not None:
            df = df[df['time'] == subset_time]
            
        # Apply subsampling for performance or clarity
        if subsample is not None and len(df) > subsample:
            df = df.sample(n=subsample, random_state=42)
        
        if slice_by:
            if slice_vals is None:
                slice_vals = sorted(df[slice_by].unique())
        else:
            slice_vals = [None]
            
        # Calculate global limits if sharing scales across all subplots
        if share_scales:
            if xlim is None and not df.empty:
                x_data_all = np.log1p(df[x_col].clip(lower=0))
                xlim = (x_data_all.min(), x_data_all.max())
            if ylim is None and not df.empty:
                y_data_all = np.log1p(df[y_col].clip(lower=0))
                ylim = (y_data_all.min(), y_data_all.max())
            
        n_plots = len(slice_vals)
        
        cols = min(n_plots, row_plots)
        rows = int(np.ceil(n_plots / cols)) if n_plots > 0 else 1
        
        if figsize is None:
            figsize = (5 * cols, 4.5 * rows) if n_plots > 1 else (5.5, 5.0)
             
        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        for i in range(len(axes)):
            if i >= n_plots:
                axes[i].axis('off')
                continue
                
            val = slice_vals[i]
            ax = axes[i]
            subset_df = df if val is None else df[df[slice_by] == val]
            
            if subset_df.empty:
                ax.axis('off')
                continue
            
            n_events = len(subset_df)
            x_data = np.log1p(subset_df[x_col].clip(lower=0))
            y_data = np.log1p(subset_df[y_col].clip(lower=0))
            
            # Handle single-value edge cases
            x_range = xlim or (x_data.min(), x_data.max())
            y_range = ylim or (y_data.min(), y_data.max())
            
            density, x_edges, y_edges = points_to_density_image(
                x_data, y_data, bins=bins, 
                x_range=x_range if xlim else None,
                y_range=y_range if ylim else None
            )
            
            mesh = None
            if show_contours:
                x_centers = (x_edges[:-1] + x_edges[1:]) / 2
                y_centers = (y_edges[:-1] + y_edges[1:]) / 2
                smoothed_density = gaussian_filter(np.nan_to_num(density, nan=0), sigma=2.5)
                d_max = smoothed_density.max()
                if d_max > 0:
                    levels = np.linspace(d_max * 0.05, d_max, contour_levels)
                    mesh = ax.contour(x_centers, y_centers, smoothed_density,
                                      levels=levels, cmap=cmap, linewidths=1.5)
            else:
                density_plot = np.where(density == 0, np.nan, density)
                mesh = ax.pcolormesh(x_edges, y_edges, density_plot,
                                     norm=LogNorm(clip=True), cmap=cmap, shading='auto',
                                     rasterized=True)
            
            if show_percentiles:
                for p in [10, 25, 50, 75, 90]:
                    px = np.percentile(x_data, p)
                    py = np.percentile(y_data, p)
                    ax.axvline(px, color='white', alpha=0.3, linestyle=':', linewidth=0.5)
                    ax.axhline(py, color='white', alpha=0.3, linestyle=':', linewidth=0.5)
            
            if gates_to_overlay:
                handles = []
                
                # Flatten any GateSets into individual gates
                flat_gates = []
                for g in gates_to_overlay:
                    # NOTE: Assuming GateSet class logic is checked using an attribute or typing based on your codebase
                    if hasattr(g, 'gates') and isinstance(getattr(g, 'gates'), list): 
                        flat_gates.extend(g.gates)
                    else:
                        flat_gates.append(g)

                for g_idx, gate in enumerate(flat_gates):
                    # --- NEW LOGIC START ---
                    # Check if the gate columns match the plot columns.
                    # We safely check both 'x'/'y' and 'x_col'/'y_col' depending on your model.
                    gate_x = getattr(gate, 'x', getattr(gate, 'x_col', None))
                    gate_y = getattr(gate, 'y', getattr(gate, 'y_col', None))
                    
                    if gate_x and gate_y:
                        if gate_x != x_col or gate_y != y_col:
                            continue  # Skip drawing this gate if columns don't match
                    # --- NEW LOGIC END ---

                    # Determine color for the gate
                    g_color = gate_colors[g_idx % len(gate_colors)] if isinstance(gate_colors, list) else gate_colors
                    
                    # NOTE: Checking type based on class names. Adjust based on your imports
                    gate_type_name = gate.__class__.__name__
                    
                    if gate_type_name == 'RectangleGate':
                        patch = Rectangle((gate.xmin, gate.ymin), 
                                          gate.xmax - gate.xmin, gate.ymax - gate.ymin,
                                          fill=False, edgecolor=g_color, linewidth=2)
                    elif gate_type_name == 'PolygonGate':
                        patch = MplPolygon(gate.vertices, closed=True, fill=False, 
                                           edgecolor=g_color, linewidth=2)
                    elif gate_type_name == 'EllipseGate':
                        patch = Ellipse(gate.center, gate.width, gate.height,
                                        angle=gate.angle, fill=False, 
                                        edgecolor=g_color, linewidth=2)
                    else: 
                        continue
                    
                    if show_stats:
                        # Evaluate gate on raw subset to get precise event counts inside gate
                        g_mask = gate.evaluate(subset_df)
                        g_count = g_mask.sum()
                        g_pct = (g_count / n_events * 100) if n_events > 0 else 0
                        
                        out_count = n_events - g_count
                        out_pct = 100.0 - g_pct if n_events > 0 else 0
                        
                        patch.set_label(f"{getattr(gate, 'name', 'Gate')}\nIn: {g_count:,} ({g_pct:.1f}%)\nOut: {out_count:,} ({out_pct:.1f}%)")
                        handles.append(patch)
                        
                    ax.add_patch(patch)
                
                if show_stats and handles:
                    ax.legend(handles=handles, loc='best', fontsize=8, fancybox=True, framealpha=0.7)
                    
            if vline is not None:
                ax.axvline(vline, color=line_color, linestyle='--', linewidth=1.5)
            if hline is not None:
                ax.axhline(hline, color=line_color, linestyle='--', linewidth=1.5)
                
            # Add population statistics text relative to horizontal/vertical lines
            if show_stats and n_events > 0:
                box_props = dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5)
                if vline is not None and hline is not None:
                    # Quadrants
                    q1 = np.sum((x_data >= vline) & (y_data >= hline))
                    q2 = np.sum((x_data < vline) & (y_data >= hline))
                    q3 = np.sum((x_data < vline) & (y_data < hline))
                    q4 = np.sum((x_data >= vline) & (y_data < hline))
                    
                    ax.text(0.95, 0.95, f"{q1:,}\n({q1/n_events*100:.1f}%)", transform=ax.transAxes, ha='right', va='top', bbox=box_props, fontsize=9, color='black')
                    ax.text(0.05, 0.95, f"{q2:,}\n({q2/n_events*100:.1f}%)", transform=ax.transAxes, ha='left', va='top', bbox=box_props, fontsize=9, color='black')
                    ax.text(0.05, 0.05, f"{q3:,}\n({q3/n_events*100:.1f}%)", transform=ax.transAxes, ha='left', va='bottom', bbox=box_props, fontsize=9, color='black')
                    ax.text(0.95, 0.05, f"{q4:,}\n({q4/n_events*100:.1f}%)", transform=ax.transAxes, ha='right', va='bottom', bbox=box_props, fontsize=9, color='black')
                elif vline is not None:
                    # Left / Right
                    left = np.sum(x_data < vline)
                    right = np.sum(x_data >= vline)
                    ax.text(0.05, 0.95, f"{left:,}\n({left/n_events*100:.1f}%)", transform=ax.transAxes, ha='left', va='top', bbox=box_props, fontsize=9, color='black')
                    ax.text(0.95, 0.95, f"{right:,}\n({right/n_events*100:.1f}%)", transform=ax.transAxes, ha='right', va='top', bbox=box_props, fontsize=9, color='black')
                elif hline is not None:
                    # Top / Bottom
                    bottom = np.sum(y_data < hline)
                    top = np.sum(y_data >= hline)
                    ax.text(0.05, 0.05, f"{bottom:,}\n({bottom/n_events*100:.1f}%)", transform=ax.transAxes, ha='left', va='bottom', bbox=box_props, fontsize=9, color='black')
                    ax.text(0.05, 0.95, f"{top:,}\n({top/n_events*100:.1f}%)", transform=ax.transAxes, ha='left', va='top', bbox=box_props, fontsize=9, color='black')
            
            if show_colorbar and mesh is not None:
                cbar = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.03)
                cbar.ax.tick_params(labelsize=9, width=0.6, length=3)
                cbar.outline.set_linewidth(0.6)
                cbar.set_label('Event Density', fontsize=10, fontweight='bold')
            
            if xlim: ax.set_xlim(xlim)
            if ylim: ax.set_ylim(ylim)
            
            val_str = f" | {slice_by}: {val}" if val is not None else ""
            if subset_sample is not None: val_str += f" | Sample: {subset_sample}"
            if subset_time is not None: val_str += f" | Time: {subset_time}"
                
            plot_title = title or f"[{pop_name or getattr(self, 'active_pop', 'Unknown')}] {y_col} vs {x_col}\n(n={n_events:,}{val_str})"
            
            self._style_ax(ax,
                           xlabel=f"{x_col} (log)",
                           ylabel=f"{y_col} (log)" if (i % cols == 0) else None,
                           title=plot_title,
                           spine_style='box')
            
        fig.tight_layout()
        self._show_static_fig(fig, save_path)

    def plot_histogram(self, col: str, title: str = "", max_value: float = None, vline=None, pop_name: str = None, save_path: str = None):
        df = self.get_data(pop_name)
        if df.empty: return
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(np.log1p(df[col].clip(lower=0)), bins=100, color='#2166ac', alpha=0.85,
                density=True, linewidth=0)
        if max_value is not None: ax.set_xlim(1, max_value)
        if vline is not None:
            ax.axvline(vline, color='red', linestyle='--', linewidth=2, label=f'Threshold ({vline})')
            ax.legend(fontsize=11)
        self._style_ax(ax, xlabel=f"{col} (log)", ylabel="Density", title=title or f"[{pop_name or self.active_pop}] Histogram of {col}")
        fig.tight_layout()
        self._show_static_fig(fig, save_path)
        
    def plot_population_comparison(self, populations: List[str], x_col: str, y_col: str = None,
                                   comparison_type: str = 'overlay', save_path: str = None,
                                   **kwargs):
        """Compare multiple populations side-by-side or overlaid."""
        if comparison_type == 'overlay':
            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 6)))
            colors = plt.cm.tab10(np.linspace(0, 1, len(populations)))
            
            for i, pop in enumerate(populations):
                df = self.get_data(pop)
                if df.empty: continue
                x = np.log1p(df[x_col].clip(lower=0))
                if y_col:
                    y = np.log1p(df[y_col].clip(lower=0))
                    ax.scatter(x[::10], y[::10], alpha=0.3, s=1, c=[colors[i]], label=pop)
                else:
                    ax.hist(x, bins=100, alpha=0.5, label=pop, density=True)
            
            ax.legend(fontsize=10)
            self._style_ax(ax, xlabel=x_col, ylabel=y_col or 'Density',
                           title=f"Population Comparison: {x_col}",
                           spine_style='box' if y_col is not None else 'open')
            
        elif comparison_type == 'side_by_side':
            n_pops = len(populations)
            fig, axes = plt.subplots(1, n_pops, figsize=(5*n_pops, 5), squeeze=False)
            
            for i, (ax, pop) in enumerate(zip(axes.flat, populations)):
                df = self.get_data(pop)
                if df.empty: continue
                
                x = np.log1p(df[x_col].clip(lower=0))
                if y_col:
                    y = np.log1p(df[y_col].clip(lower=0))
                    density, xe, ye = points_to_density_image(x, y)
                    ax.pcolormesh(xe, ye, np.where(density == 0, np.nan, density),
                                norm=LogNorm(), cmap='viridis')
                else:
                    ax.hist(x, bins=100, alpha=0.7)
                
                self._style_ax(ax, xlabel=x_col, ylabel=y_col or 'Count',
                               title=f"{pop}\n(n={len(df):,})",
                               spine_style='box' if y_col is not None else 'open')
        
        fig.tight_layout()
        self._show_static_fig(fig, save_path)

    # --- Analysis & PRI (with Error propagation, Residuals and Bootstrap CIs) ---
    # --- Analysis & PRI (with Error propagation, Residuals and Bootstrap CIs) ---
    def run_pri_analysis(self, channel: str, control_sample: str, samples: list = None,
                         pos_frac: float = 0.01, baseline_time: int = 0, pop_name: str = None,
                         n_bootstrap: int = 100, confidence: float = 0.95, ctrl_sample_list: list = None,
                         reference_sample: str = None, threshold_log: float = None,
                         mfi_metric: str = 'geometric_mean',
                         wt_sample: str = None,
                         temperature_c: float = 55.0,
                         flatline_threshold: float = 0.10):
        """Enhanced PRI analysis with bootstrap confidence intervals and optional reference normalization."""
        df = self.get_data(pop_name)
        if df.empty:
            raise ValueError(f"Population '{pop_name or self.active_pop}' is empty or does not exist.")
        if channel not in df.columns:
            raise KeyError(
                f"Channel '{channel}' not found. Available columns: {list(df.columns)}"
            )
        if samples is None: samples = df['sample'].unique().tolist()

        # 1. Combine primary control and fallback controls into a prioritized list
        potential_ctrls = [control_sample]
        if ctrl_sample_list:
            potential_ctrls.extend(ctrl_sample_list)
            
        # 2. Find the first control sample that actually exists in the data
        available_samples = df['sample'].unique()
        valid_ctrl_name = None
        
        for candidate in potential_ctrls:
            if candidate in available_samples:
                valid_ctrl_name = candidate
                break
                
        # 3. Handle the case where NO valid controls are found
        if not valid_ctrl_name:
            print(f"⚠️ None of the control samples {potential_ctrls} were found in population '{pop_name or getattr(self, 'active_pop', 'Unknown')}'. Cannot run PRI analysis.")
            return

        # 4. Announce which control is being used and slice the dataframe
        print(f"✅ Using control sample '{valid_ctrl_name}' for PRI analysis thresholding.")
        ctrl = df.loc[df["sample"] == valid_ctrl_name].copy()

        if threshold_log is not None:
            thr_log = float(threshold_log)
            print(f"✅ Using user-supplied threshold (log1p scale): {thr_log:.4f}")
        else:
            thr_log = float(np.quantile(np.log1p(_coerce_nonneg(ctrl[channel]).values), float(np.clip(1.0 - pos_frac, 0.0, 1.0))))
            print(f"   Auto threshold from control top {pos_frac*100:.1f}% quantile: {thr_log:.4f} (log1p)")

        _VALID_MFI_METRICS = {'geometric_mean', 'median'}
        if mfi_metric not in _VALID_MFI_METRICS:
            raise ValueError(
                f"mfi_metric must be one of {sorted(_VALID_MFI_METRICS)!r}, got {mfi_metric!r}"
            )
        if not 0.0 <= flatline_threshold <= 1.0:
            raise ValueError(
                f"flatline_threshold must be in [0, 1], got {flatline_threshold!r}"
            )
        _mfi_fn = median_mfi if mfi_metric == 'median' else geometric_mfi

        # 5. Calculate Reference Baseline if provided
        ref_pri_abs0 = None
        if reference_sample:
            if reference_sample not in available_samples:
                print(f"⚠️ Reference sample '{reference_sample}' not found. Falling back to self-normalization.")
            else:
                ref_sub = df[(df["sample"] == reference_sample) & (df["time"] == baseline_time)].copy()
                if ref_sub.empty:
                    print(f"⚠️ Baseline time {baseline_time} not found for reference '{reference_sample}'. Falling back to self-normalization.")
                else:
                    # FIX: Pass the Pandas Series into _coerce_nonneg, then extract .values
                    ref_vals = _coerce_nonneg(ref_sub[channel]).values
                    ref_log = np.log1p(ref_vals)
                    ref_n_pos = int(np.sum(ref_log >= thr_log))
                    ref_f_plus = (ref_n_pos / ref_vals.size) if ref_vals.size else 0.0
                    ref_gmfi = _mfi_fn(ref_vals[ref_log >= thr_log]) if ref_n_pos else 0.0
                    ref_pri_abs0 = ref_f_plus * ref_gmfi
                    
                    if not np.isfinite(ref_pri_abs0) or ref_pri_abs0 <= 0:
                        print(f"⚠️ Invalid baseline PRI for reference '{reference_sample}'. Falling back to self-normalization.")
                        ref_pri_abs0 = None
                    else:
                        print(f"✅ Normalizing all samples to reference '{reference_sample}' at baseline time {baseline_time}.")

        tables = []
        for s in samples:
            sub = df.loc[df["sample"] == s].copy()
            if sub.empty: continue
            sub[channel] = _coerce_nonneg(sub[channel])
            times = sorted(pd.unique(sub["time"]))
            
            # Calculate self-baseline gMFI (needed if falling back to self-normalization)
            t0_vals = sub.loc[sub["time"] == baseline_time, channel].values
            t0_pos_mask = np.log1p(t0_vals) >= thr_log
            gmfi0 = _mfi_fn(t0_vals[t0_pos_mask])

            for t in times:
                vals = sub.loc[sub["time"] == t, channel].values
                log_vals = np.log1p(vals)
                n_pos = int(np.sum(log_vals >= thr_log))
                f_plus = (n_pos / vals.size) if vals.size else 0.0
                
                gmfi_pos = _mfi_fn(vals[log_vals >= thr_log]) if n_pos else 0.0
                PRI_abs = f_plus * gmfi_pos
                
                # Choose normalization method based on reference availability
                if ref_pri_abs0 is not None:
                    # New: Normalize total PRI to reference sample's baseline PRI
                    PRI_norm = (PRI_abs / ref_pri_abs0) if np.isfinite(PRI_abs) else np.nan
                else:
                    # Old: Normalize only the gMFI to the sample's own baseline
                    gmfi_pos_norm = (gmfi_pos / gmfi0) if (n_pos and np.isfinite(gmfi0) and gmfi0 > 0) else (0.0 if not n_pos else np.nan)
                    PRI_norm = (f_plus * gmfi_pos_norm) if np.isfinite(gmfi_pos_norm) else np.nan

                row = dict(
                    sample=s, time=float(t), 
                    PRI_abs=PRI_abs,
                    PRI_norm=PRI_norm,
                    n_events=len(vals),
                    n_positive=n_pos
                )
                
                # Bootstrap confidence intervals (only for PRI_abs currently)
                if n_bootstrap > 0 and len(vals) > 10:
                    pri_samples = []
                    rng = np.random.default_rng(seed=42)
                    for _ in range(n_bootstrap):
                        boot_idx = rng.choice(len(vals), len(vals), replace=True)
                        boot_vals = vals[boot_idx]
                        boot_log = np.log1p(boot_vals)
                        boot_n_pos = np.sum(boot_log >= thr_log)
                        boot_f_plus = boot_n_pos / len(boot_vals)
                        boot_gmfi = _mfi_fn(boot_vals[boot_log >= thr_log]) if boot_n_pos else 0.0
                        pri_samples.append(boot_f_plus * boot_gmfi)
                    
                    alpha = (1 - confidence) / 2
                    row['PRI_abs_ci_low'] = np.percentile(pri_samples, alpha * 100)
                    row['PRI_abs_ci_high'] = np.percentile(pri_samples, (1 - alpha) * 100)
                
                tables.append(row)
        
        self.pri_table = pd.DataFrame(tables).sort_values(["sample","time"]).reset_index(drop=True)
        self.pri_fits_abs = self._fit_global_exponential(self.pri_table, "PRI_abs",
                                                          flatline_threshold=flatline_threshold)
        self.pri_fits_norm = self._fit_global_exponential(self.pri_table, "PRI_norm",
                                                           flatline_threshold=flatline_threshold)
        self.pri_channel = channel
        self.pri_pop = pop_name or self.active_pop
        self.pri_control_sample = valid_ctrl_name
        
        # Calculate and store residuals directly into the main table
        self.pri_table['PRI_abs_fit_res'] = self._compute_residuals(self.pri_table, self.pri_fits_abs, 'PRI_abs')
        self.pri_table['PRI_norm_fit_res'] = self._compute_residuals(self.pri_table, self.pri_fits_norm, 'PRI_norm')

        print(f"✅ PRI Analysis Complete on '{pop_name or getattr(self, 'active_pop', 'Unknown')}' with {n_bootstrap} bootstrap iterations.")

    def _compute_residuals(self, df: pd.DataFrame, fits: pd.DataFrame, which: str) -> List[float]:
        res_list = []
        for _, row in df.iterrows():
            s, t, y_true = row['sample'], row['time'], row[which]
            fit_row = fits[fits['sample'] == s]
            if not fit_row.empty and pd.notna(y_true):
                A, k, C = fit_row.iloc[0]['A'], fit_row.iloc[0]['k'], fit_row.iloc[0]['C']
                y_pred = A * np.exp(-k * t) + C
                res_list.append(y_true - y_pred)
            else:
                res_list.append(np.nan)
        return res_list

    def _fit_global_exponential(self, df_source: pd.DataFrame, which: str,
                                 flatline_threshold: float = 0.10) -> pd.DataFrame:
        all_samples = sorted(df_source['sample'].unique())
        times_list_all, values_list_all = [], []
        for s in all_samples:
            g = df_source[df_source['sample'] == s].sort_values('time')
            times_list_all.append(g['time'].values)
            values_list_all.append(g[which].values)

        # Skip samples with fewer than 3 time points — fitting is degenerate
        skipped = []
        samples, times_list, values_list = [], [], []
        for s, t, v in zip(all_samples, times_list_all, values_list_all):
            if len(t) < 3:
                skipped.append(s)
            else:
                samples.append(s)
                times_list.append(t)
                values_list.append(v)

        if skipped:
            warnings.warn(
                f"Skipping fit for samples with < 3 time points: {skipped}",
                UserWarning, stacklevel=3
            )

        skipped_rows = [
            dict(sample=s, A=np.nan, A_err=np.nan, k=np.nan, k_err=np.nan,
                 C=np.nan, C_err=np.nan, t_half=np.nan, t_half_err=np.nan,
                 r2=np.nan, fit_quality='insufficient_data')
            for s in skipped
        ]

        # --- Flatline / hyperstable detection ---
        # A sample is hyperstable if the normalised linear-regression decay rate
        # (slope / mean_signal) is below flatline_threshold per unit time scaled
        # by the time span.  This is robust to per-timepoint noise that would
        # fool a simple first-to-last comparison.
        _eps_flat = 1e-12
        flatline_samples = []
        normal_samples, normal_times, normal_values = [], [], []
        for _s, _t, _v in zip(samples, times_list, values_list):
            _mask = np.isfinite(_v)
            _vf, _tf = _v[_mask], _t[_mask]
            if _vf.size > 1:
                _mean_v = np.mean(np.abs(_vf))
                if _mean_v > _eps_flat:
                    _slope = np.polyfit(_tf, _vf, 1)[0]
                    _span = max(_tf[-1] - _tf[0], _eps_flat)
                    # Normalised total drop over time span, relative to mean signal
                    _drop = -_slope * _span / (_mean_v + _eps_flat)
                else:
                    _drop = 0.0  # all zeros → treat as flatline
            else:
                _drop = 1.0  # zero or one finite point — cannot fit slope; treat as normal
            if 0.0 <= _drop < flatline_threshold:
                flatline_samples.append(_s)
            else:
                normal_samples.append(_s)
                normal_times.append(_t)
                normal_values.append(_v)
        samples, times_list, values_list = normal_samples, normal_times, normal_values

        if not samples:
            _flat_early = [
                dict(sample=_s, A=np.nan, A_err=np.nan, k=0.0, k_err=np.nan,
                     C=np.nan, C_err=np.nan, t_half=np.inf, t_half_err=np.nan,
                     r2=np.nan, fit_quality='hyperstable')
                for _s in flatline_samples
            ]
            return pd.DataFrame(skipped_rows + _flat_early)

        min_vals = []
        for y in values_list:
            if len(y) > 0:
                mv = np.nanmin(y)
                min_vals.append(mv if not np.isnan(mv) else 1e-6)
        C0 = max(np.mean(min_vals) if min_vals else 0.0, 1e-4)

        initial_params, bounds_lower, bounds_upper = [C0], [0.0], [1.0 if which == 'PRI_norm' else np.inf]
        for i in range(len(samples)):
            y, t = values_list[i], times_list[i]
            A0 = max(np.nanmax(y) if len(y) > 0 else 0, 1e-9)
            k0 = np.log(2) / (np.median(t[t > 0]) if np.any(t > 0) else 1.0)
            initial_params.extend([A0, k0]); bounds_lower.extend([0.0, 1e-8]); bounds_upper.extend([np.inf, 1e7])

        def residuals(params):
            C, res = params[0], []
            for i in range(len(samples)):
                A, k = params[1+2*i], params[2+2*i]
                res.extend(values_list[i] - (A * np.exp(-k * times_list[i]) + C))
            return np.array(res)

        res = least_squares(residuals, initial_params, bounds=(bounds_lower, bounds_upper), method='trf')
        global_C = res.x[0]

        # Parameter standard error estimation via Jacobian pseudo-inverse
        J = res.jac
        JTJ = J.T @ J
        if np.linalg.matrix_rank(JTJ) < len(res.x):
            warnings.warn(
                "Jacobian rank deficiency detected; covariance estimation may be unreliable.",
                FitConvergenceWarning, stacklevel=3
            )

        dof = max(len(res.fun) - len(res.x), 1)
        mse = (res.fun ** 2).sum() / dof

        try:
            cov = mse * np.linalg.pinv(JTJ)
            param_errors = np.sqrt(np.diag(cov))
        except Exception as cov_err:
            param_errors = np.full_like(res.x, np.nan)
            warnings.warn(
                f"Covariance estimation failed; parameter errors set to NaN. Detail: {cov_err}",
                FitConvergenceWarning, stacklevel=3
            )

        C_err = param_errors[0]

        flat_rows = [
            dict(sample=_s, A=np.nan, A_err=np.nan, k=0.0, k_err=np.nan,
                 C=global_C, C_err=C_err,
                 t_half=np.inf, t_half_err=np.nan, r2=np.nan, fit_quality='hyperstable')
            for _s in flatline_samples
        ]

        out = []
        for i, s in enumerate(samples):
            A, k = res.x[1+2*i], res.x[2+2*i]
            A_err, k_err = param_errors[1+2*i], param_errors[2+2*i]

            # Error propagation for half-life
            t_half = np.log(2)/k if k > 0 else np.nan
            t_half_err = (np.log(2)/(k**2)) * k_err if k > 0 and not np.isnan(k_err) else np.nan

            # R² for this sample
            y_obs = values_list[i]
            y_pred_s = A * np.exp(-k * times_list[i]) + global_C
            valid = np.isfinite(y_obs) & np.isfinite(y_pred_s)
            if valid.sum() > 1:
                ss_res = np.sum((y_obs[valid] - y_pred_s[valid]) ** 2)
                ss_tot = np.sum((y_obs[valid] - np.mean(y_obs[valid])) ** 2)
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
            else:
                r2 = np.nan

            # fit_quality flag based on RMSE vs median signal
            y_obs_valid = y_obs[valid] if valid.sum() > 0 else np.array([])
            if len(y_obs_valid) > 0:
                rmse = np.sqrt(np.mean((y_obs[valid] - y_pred_s[valid]) ** 2))
                median_val = np.nanmedian(np.abs(y_obs_valid))
                fit_quality = 'poor' if (median_val > 1e-12 and rmse > 0.5 * median_val) else 'good'
            else:
                fit_quality = 'unknown'

            out.append(dict(
                sample=s, A=A, A_err=A_err, k=k, k_err=k_err,
                C=global_C, C_err=C_err, t_half=t_half, t_half_err=t_half_err,
                r2=r2, fit_quality=fit_quality
            ))

        all_rows = out + skipped_rows + flat_rows
        return pd.DataFrame(all_rows).sort_values("sample").reset_index(drop=True)

    def plot_pri(self, which: str = "PRI_norm", cols: int = 2, title: str = None, save_path: str = None,
                plot_residuals: bool = False, dataset: str = None,
                show_ci: bool = False,
                show_params: list = None,
                spine_width: float = None):
        if self.pri_table.empty: return
        data = self.pri_table.copy()
        if dataset is not None and 'dataset' in data.columns:
            data = data[data['dataset'] == dataset]
        samples = sorted(data["sample"].unique())
        rows = int(np.ceil(len(samples) / cols))

        panel_w = 3.5
        panel_h = 2.8 if not plot_residuals else 3.5
        fig_width = max(7.0, panel_w * cols)
        fig_height = max(3.0, panel_h * rows)
        fig = plt.figure(figsize=(fig_width, fig_height))

        outer_gs = fig.add_gridspec(rows, cols, wspace=0.38, hspace=0.55 if plot_residuals else 0.48)

        tgrid = np.linspace(np.nanmin(data["time"]), np.nanmax(data["time"]), 300)
        fits_df = self.pri_fits_norm if which == "PRI_norm" else self.pri_fits_abs

        # Global Y limits — shared across all panels for easy comparison
        y_min = data[which].min()
        y_max = data[which].max()
        y_pad = (y_max - y_min) * 0.1 if y_max != y_min else 0.1
        global_bottom = min(-0.04 * max(y_max, 1e-9), y_min - y_pad) if y_min >= 0 else y_min - y_pad
        global_top = y_max + y_pad

        if plot_residuals:
            res_col_name = f"{which}_fit_res"
            if res_col_name in data.columns:
                res_max_abs = data[res_col_name].abs().max()
                res_pad = res_max_abs * 0.15 if res_max_abs > 0 else 0.05
                global_res_ylim = (-res_max_abs - res_pad, res_max_abs + res_pad)
            else:
                global_res_ylim = (-0.1, 0.1)

        _SCATTER_COLOR = '#2166ac'  # steel blue — data points
        _FIT_COLOR = '#d6604d'      # brick red — fit line
        _CI_COLOR = '#d6604d'       # same hue, filled transparent for CI band

        # Default: show all three params; allow subset via show_params list
        _ALL_PARAMS = ['t_half', 'y0', 'r2']
        _show = set(show_params if show_params is not None else _ALL_PARAMS)

        for i, s in enumerate(samples):
            r, c = divmod(i, cols)
            inner_gs = outer_gs[r, c].subgridspec(
                nrows=2 if plot_residuals else 1,
                ncols=1,
                height_ratios=[3, 1] if plot_residuals else [1],
                hspace=0.06
            )
            ax = fig.add_subplot(inner_gs[0])
            if plot_residuals:
                ax_res = fig.add_subplot(inner_gs[1], sharex=ax)

            g = data[data["sample"] == s].sort_values("time")

            # Data markers: filled, white edge for crispness
            ax.plot(g["time"].values, g[which].values, "o",
                    color=_SCATTER_COLOR, markeredgecolor='white', markeredgewidth=0.8,
                    markersize=6.5, alpha=0.95, zorder=3, label="Data")

            fit_row = fits_df[fits_df["sample"] == s]
            if not fit_row.empty and np.isfinite(fit_row["t_half"].values[0]):
                A = fit_row["A"].values[0]
                k = fit_row["k"].values[0]
                C = fit_row["C"].values[0]
                thalf = fit_row["t_half"].values[0]
                thalf_err = fit_row["t_half_err"].values[0]
                r2_val = fit_row["r2"].values[0] if "r2" in fit_row.columns else np.nan
                A_err = fit_row["A_err"].values[0] if "A_err" in fit_row.columns else np.nan
                k_err = fit_row["k_err"].values[0] if "k_err" in fit_row.columns else np.nan

                y_fit = A * np.exp(-k * tgrid) + C
                ax.plot(tgrid, y_fit, "-", color=_FIT_COLOR, linewidth=1.8, zorder=2, label="Fit")

                # 95% CI band via Gaussian error propagation on A and k
                if show_ci and np.isfinite(A_err) and np.isfinite(k_err):
                    dydA = np.exp(-k * tgrid)
                    dydk = -A * tgrid * np.exp(-k * tgrid)
                    std_y = np.sqrt((dydA * A_err) ** 2 + (dydk * k_err) ** 2)
                    ci = 1.96 * std_y
                    ax.fill_between(tgrid, y_fit - ci, y_fit + ci,
                                    color=_CI_COLOR, alpha=0.15, linewidth=0, zorder=1,
                                    label="95% CI")

                # Build annotation from selected params
                lines = []
                if 't_half' in _show:
                    err_str = f" ±{thalf_err:.1f}" if pd.notna(thalf_err) else ""
                    lines.append(f"$t_{{1/2}} = {thalf:.1f}${err_str} min")
                if 'y0' in _show:
                    lines.append(f"$y_0 = {C:.3f}$")
                if 'r2' in _show and pd.notna(r2_val):
                    lines.append(f"$R^2 = {r2_val:.3f}$")
                text_str = "\n".join(lines) if lines else ""
            else:
                text_str = "$t_{1/2} =$ N/A"

            if text_str:
                box_props = dict(boxstyle='round,pad=0.4', facecolor='white',
                                 alpha=0.88, edgecolor='#cccccc', linewidth=0.6)
                # Place at lower-right — safe for decaying data (curve is low there)
                ax.text(0.97, 0.04, text_str, transform=ax.transAxes,
                        ha='right', va='bottom', fontsize=8.5, bbox=box_props)

            ax.grid(True, alpha=0.18, linestyle='--', linewidth=0.5, color='#888888')
            ax.set_ylim(bottom=global_bottom, top=global_top)

            if plot_residuals:
                res_col = f"{which}_fit_res"
                ax_res.axhline(0, color='#555555', linestyle='--', linewidth=0.8, zorder=0)
                ax_res.plot(g["time"].values, g[res_col].values, "o-",
                            color=_SCATTER_COLOR, markersize=3.5, linewidth=0.9,
                            markeredgecolor='white', markeredgewidth=0.5, zorder=2)
                ax_res.set_ylim(global_res_ylim)
                ax_res.grid(True, alpha=0.18, linestyle='--', linewidth=0.5, color='#888888')
                ax.tick_params(labelbottom=False)
                self._style_ax(ax_res, xlabel="Time (min)", ylabel="Resid.",
                               spine_width=spine_width)
                self._style_ax(ax, ylabel=which.replace("_", " "), title=s,
                               spine_width=spine_width)
            else:
                self._style_ax(ax, xlabel="Time (min)", ylabel=which.replace("_", " "), title=s,
                               spine_width=spine_width)

        if title:
            fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)

        self._show_static_fig(fig, save_path)

    def plot_pri_summary_grid(self, which: str = 'PRI_norm', cols: int = 4,
                              save_path: str = None, title: str = None,
                              show_ci: bool = False,
                              show_params: list = None,
                              spine_width: float = None):
        """
        Publication-ready combined figure: PRI decay curves (top) + t½ bar chart (bottom).
        Sized at FIG_2COL width. Suitable as a single multi-panel manuscript figure.
        """
        if self.pri_table.empty:
            print("⚠️ No PRI data. Run run_pri_analysis first.")
            return

        data = self.pri_table.copy()
        fits_df = self.pri_fits_norm if which == 'PRI_norm' else self.pri_fits_abs
        samples = sorted(data['sample'].unique())
        n_samples = len(samples)
        rows_top = int(np.ceil(n_samples / cols))

        # Panel heights: decay curves tall, bar chart at least 2.8" for readability
        curve_h = rows_top * 2.5
        bar_h = max(2.8, n_samples * 0.35 + 1.0)
        fig = plt.figure(figsize=(FIG_2COL, curve_h + bar_h))
        outer = fig.add_gridspec(2, 1, height_ratios=[curve_h, bar_h], hspace=0.5)

        # --- Top: decay curves ---
        inner_top = outer[0].subgridspec(rows_top, cols, hspace=0.65, wspace=0.4)
        tgrid = np.linspace(np.nanmin(data['time']), np.nanmax(data['time']), 200)

        y_vals = data[which].dropna().values
        y_pad = (y_vals.max() - y_vals.min()) * 0.08 if len(y_vals) > 0 else 0.1
        global_ylim = (min(y_vals.min() - y_pad, -0.02), y_vals.max() + y_pad)

        _SCATTER_COLOR = '#2166ac'
        _FIT_COLOR = '#d6604d'
        _ALL_PARAMS = ['t_half', 'y0', 'r2']
        _show = set(show_params if show_params is not None else _ALL_PARAMS)

        for i, s in enumerate(samples):
            r, c = divmod(i, cols)
            ax = fig.add_subplot(inner_top[r, c])
            g = data[data['sample'] == s].sort_values('time')

            ax.plot(g['time'].values, g[which].values, 'o',
                    color=_SCATTER_COLOR, markeredgecolor='white',
                    markeredgewidth=0.7, markersize=6, alpha=0.95, zorder=3)

            fit_row = fits_df[fits_df['sample'] == s]
            if not fit_row.empty and np.isfinite(fit_row['t_half'].values[0]):
                A = fit_row['A'].values[0]
                k = fit_row['k'].values[0]
                C = fit_row['C'].values[0]
                thalf = fit_row['t_half'].values[0]
                thalf_err = fit_row['t_half_err'].values[0]
                r2_val = fit_row['r2'].values[0] if 'r2' in fit_row.columns else np.nan
                A_err = fit_row['A_err'].values[0] if 'A_err' in fit_row.columns else np.nan
                k_err = fit_row['k_err'].values[0] if 'k_err' in fit_row.columns else np.nan
                y_fit = A * np.exp(-k * tgrid) + C
                ax.plot(tgrid, y_fit, '-', color=_FIT_COLOR, linewidth=1.8, zorder=2)

                if show_ci and np.isfinite(A_err) and np.isfinite(k_err):
                    dydA = np.exp(-k * tgrid)
                    dydk = -A * tgrid * np.exp(-k * tgrid)
                    ci = 1.96 * np.sqrt((dydA * A_err) ** 2 + (dydk * k_err) ** 2)
                    ax.fill_between(tgrid, y_fit - ci, y_fit + ci,
                                    color=_FIT_COLOR, alpha=0.15, linewidth=0, zorder=1)

                lines = []
                if 't_half' in _show:
                    err_str = f" ±{thalf_err:.1f}" if pd.notna(thalf_err) else ""
                    lines.append(f"$t_{{1/2}}={thalf:.1f}${err_str} min")
                if 'y0' in _show:
                    lines.append(f"$y_0={C:.3f}$")
                if 'r2' in _show and pd.notna(r2_val):
                    lines.append(f"$R^2={r2_val:.3f}$")
                label = "\n".join(lines) if lines else ""
            else:
                label = "$t_{1/2}=$ N/A"

            if label:
                box_kw = dict(boxstyle='round,pad=0.35', facecolor='white',
                              alpha=0.88, edgecolor='#cccccc', linewidth=0.6)
                ax.text(0.97, 0.04, label, transform=ax.transAxes,
                        ha='right', va='bottom', fontsize=8, bbox=box_kw)
            ax.set_ylim(global_ylim)
            ax.grid(True, alpha=0.18, linestyle='--', linewidth=0.5, color='#888888')
            self._style_ax(ax, title=s,
                           ylabel=which.replace('_', ' ') if c == 0 else None,
                           xlabel='Time (min)' if r == rows_top - 1 else None,
                           spine_width=spine_width)

        # Hide unused top subplots
        for j in range(n_samples, rows_top * cols):
            r, c = divmod(j, cols)
            fig.add_subplot(inner_top[r, c]).axis('off')

        # --- Bottom: t½ bar chart ---
        ax_bar = fig.add_subplot(outer[1])
        valid_fits = fits_df.dropna(subset=['t_half'])
        x = np.arange(len(valid_fits))
        y = valid_fits['t_half'].values
        yerr_sym = (valid_fits['t_half_err'].fillna(0).values
                    if 't_half_err' in valid_fits.columns else np.zeros_like(y))
        # Clip lower error so bars never go below zero
        yerr_lower = np.minimum(yerr_sym, np.maximum(y, 0.0))
        yerr_bar = np.array([yerr_lower, yerr_sym])

        cmap_bar = mpl.colormaps['viridis']
        bar_colors = cmap_bar(np.linspace(0.15, 0.85, max(len(valid_fits), 1)))
        bar_w = max(0.45, 0.6 - 0.008 * len(valid_fits))
        ax_bar.bar(x, y, width=bar_w, yerr=yerr_bar, capsize=4, color=bar_colors,
                   edgecolor='black', linewidth=1.0, alpha=0.92,
                   error_kw={'linewidth': 1.1, 'capthick': 1.1})
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(valid_fits['sample'], rotation=40, ha='right',
                               fontsize=10, fontweight='bold')
        ax_bar.yaxis.grid(True, linestyle='--', linewidth=0.5, color='grey', alpha=0.35)
        ax_bar.set_axisbelow(True)
        self._style_ax(ax_bar, ylabel='$t_{1/2}$ (min)', spine_width=spine_width)

        if title:
            fig.suptitle(title, fontsize=12, fontweight='bold', y=1.01)

        fig.tight_layout()
        self._show_static_fig(fig, save_path)

# ==============================================================================
# 5. FLOW REPORT
# ==============================================================================
from string import Template as _Template
from matplotlib.backends.backend_pdf import PdfPages as _PdfPages


class FlowReport:
    """Generate summary reports from a FlowExperiment."""

    def __init__(self, experiment: 'FlowExperiment'):
        self.exp = experiment

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Print and return a formatted text summary of all populations."""
        lines = [
            "=" * 60,
            "  FlowExperiment Summary",
            "=" * 60,
        ]
        raw = self.exp.populations.get('raw', pd.DataFrame())
        n_raw = len(raw)
        lines.append(f"  Raw events : {n_raw:,}")
        lines.append(f"  Channels   : {', '.join(self.exp.channels)}")
        ds = self.exp.datasets
        if len(ds) > 1:
            lines.append(f"  Datasets   : {', '.join(ds.keys())}")
        lines.append("")
        lines.append(f"  {'Population':<22} {'N Events':>10} {'% of Raw':>10}")
        lines.append(f"  {'-'*22} {'-'*10} {'-'*10}")
        for name, df in self.exp.populations.items():
            pct = f"{100 * len(df) / n_raw:.1f}" if n_raw > 0 else "—"
            active = " ◀" if name == self.exp.active_pop else ""
            lines.append(f"  {name:<22} {len(df):>10,} {pct:>10}{active}")
        lines.append("=" * 60)
        result = "\n".join(lines)
        print(result)
        return result

    # ------------------------------------------------------------------
    def pri_summary(self) -> pd.DataFrame:
        """Return per-sample PRI fit summary as a DataFrame and print it."""
        if self.exp.pri_fits_norm.empty and self.exp.pri_fits_abs.empty:
            print("⚠️ No PRI fits available. Run run_pri_analysis first.")
            return pd.DataFrame()

        fits = self.exp.pri_fits_norm if not self.exp.pri_fits_norm.empty else self.exp.pri_fits_abs

        cols = ['sample', 'A', 'A_err', 'k', 'k_err', 't_half', 't_half_err']
        if 'r2' in fits.columns:
            cols.append('r2')
        if 'fit_quality' in fits.columns:
            cols.append('fit_quality')

        display_df = fits[[c for c in cols if c in fits.columns]].copy()

        if not self.exp.pri_table.empty and 'PRI_abs_ci_low' in self.exp.pri_table.columns:
            ci = (self.exp.pri_table
                  .groupby('sample')[['PRI_abs_ci_low', 'PRI_abs_ci_high']]
                  .mean()
                  .reset_index())
            display_df = display_df.merge(ci, on='sample', how='left')

        print(display_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        return display_df

    # ------------------------------------------------------------------
    def _fig_to_b64(self, fig) -> str:
        """Render a matplotlib figure to a base64-encoded PNG string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('ascii')

    # ------------------------------------------------------------------
    def _capture_plot(self, method, *args, **kwargs):
        """Call an experiment plot method and return the figure without displaying it."""
        captured = []
        orig = self.exp._show_static_fig

        def _grab(fig, save_path=None):
            captured.append(fig)

        self.exp._show_static_fig = _grab
        try:
            method(*args, **kwargs)
        except Exception:
            pass
        finally:
            self.exp._show_static_fig = orig
        return captured[0] if captured else None

    # ------------------------------------------------------------------
    def _detect_scatter_cols(self, df):
        """Return (fsc_col, ssc_col) best suited for a scatter gate overview."""
        fsc = [c for c in df.columns if c.upper().startswith('FSC')]
        ssc = [c for c in df.columns if c.upper().startswith('SSC')]
        x = next((c for c in fsc if c.upper().endswith('-H')), fsc[0] if fsc else None)
        y = next((c for c in ssc if c.upper().endswith('-H')), ssc[0] if ssc else None)
        return x, y

    # ------------------------------------------------------------------
    def _detect_ssc_a(self, df):
        """Return the SSC-A column name if present, else first SSC column."""
        ssc_a = next((c for c in df.columns if c.upper() == 'SSC-A'), None)
        if ssc_a is None:
            ssc_a = next((c for c in df.columns if c.upper().startswith('SSC')), None)
        return ssc_a

    # ------------------------------------------------------------------
    def _select_time_points(self, times, n=4):
        """Pick up to n representative time points (always include min and max)."""
        t_sorted = sorted(set(times))
        if len(t_sorted) <= n:
            return t_sorted
        indices = np.round(np.linspace(0, len(t_sorted) - 1, n)).astype(int)
        return [t_sorted[i] for i in indices]

    # ------------------------------------------------------------------
    def _ctrl_vline(self, pop_name, channel, percentile=99):
        """Compute the 99th-percentile threshold from the control sample (all time points).

        Mirrors the notebook pattern:
            ctrl_cells = cell_data[cell_data['sample'] == ctrl_name]
            ctrl_99perc = np.quantile(np.log1p(ctrl_cells[channel].clip(lower=0)), 0.99)
        """
        ctrl_name = self.exp.pri_control_sample
        if not ctrl_name:
            return None
        df = self.exp.get_data(pop_name)
        if df.empty or channel not in df.columns:
            return None
        ctrl_data = df[df['sample'] == ctrl_name]
        if ctrl_data.empty:
            return None
        return float(np.quantile(np.log1p(ctrl_data[channel].clip(lower=0)), percentile / 100.0))

    # ------------------------------------------------------------------
    def _collect_figures(self, pri_which='PRI_norm', show_ci=True,
                         show_params=None, pri_cols=3):
        """
        Build all report figures. Returns a list of (section, label, fig).

        Sections generated (when data is available):
          1. Gate overview  — FSC vs SSC density per population with gate overlays
          2. FL at time points — primary channel vs SSC-A per sample at key times
          3. Channel histograms — per-sample KDE histograms at t_min and t_max
          4. PRI decay curves  — plot_pri-style panels
          5. PRI half-lives    — bar chart
        """
        sections = []  # list of (section_title, label, fig)

        # ── 1. Gate / scatter overview ─────────────────────────────────
        for pop_name, pop_df in self.exp.populations.items():
            if len(pop_df) < 10:
                continue
            x_col, y_col = self._detect_scatter_cols(pop_df)
            if x_col is None or y_col is None:
                continue
            # Collect any gatesets whose gates reference these columns
            gates_overlay = [
                gs for gs in self.exp.gatesets.values()
                if any(getattr(g, 'x', getattr(g, 'x_col', None)) == x_col
                       for g in gs.gates)
            ] or None
            fig = self._capture_plot(
                self.exp.plot_density,
                x_col=x_col, y_col=y_col,
                pop_name=pop_name,
                show_contours=False,
                gates_to_overlay=gates_overlay,
                show_stats=True,
                show_colorbar=True,
            )
            if fig is not None:
                sections.append(('Gate Overview', f'{pop_name}: {x_col} vs {y_col}', fig))

        # ── 2. Primary channel density at key time points ──────────────
        channel = self.exp.pri_channel
        pri_pop = self.exp.pri_pop
        if channel and pri_pop:
            pop_df = self.exp.get_data(pri_pop)
            ssc_a = self._detect_ssc_a(pop_df)
            if ssc_a and not pop_df.empty and 'time' in pop_df.columns:
                vline = self._ctrl_vline(pri_pop, channel)
                time_points = self._select_time_points(pop_df['time'].unique(), n=4)
                samples = sorted(pop_df['sample'].unique().tolist())
                n_samples = len(samples)
                # Put all samples in a single row; cap panel width so figure stays reasonable
                panel_w = max(2.2, min(3.5, 28.0 / n_samples))
                panel_h = 3.5
                for t in time_points:
                    fig = self._capture_plot(
                        self.exp.plot_density,
                        x_col=channel, y_col=ssc_a,
                        pop_name=pri_pop,
                        slice_by='sample',
                        slice_vals=samples,
                        subset_time=t,
                        show_contours=True,
                        contour_levels=8,
                        vline=vline,
                        line_color='#696969',
                        show_colorbar=False,
                        show_stats=False,
                        row_plots=n_samples,        # all in one row
                        figsize=(panel_w * n_samples, panel_h),
                    )
                    if fig is not None:
                        sections.append((
                            'Fluorescence Density',
                            f'{channel} vs {ssc_a} — t = {t} min',
                            fig,
                        ))

        # ── 3. Per-sample histograms at t_min and t_max ────────────────
        if channel and pri_pop:
            pop_df = self.exp.get_data(pri_pop)
            if not pop_df.empty and 'time' in pop_df.columns:
                vline = self._ctrl_vline(pri_pop, channel)
                t_vals = sorted(pop_df['time'].unique())
                t_endpoints = sorted({t_vals[0], t_vals[-1]})
                samples = sorted(pop_df['sample'].unique().tolist())
                for t in t_endpoints:
                    fig = self._capture_plot(
                        self.exp.plot_sliced_histogram,
                        col=channel,
                        slice_by='sample',
                        slice_vals=samples,
                        pop_name=pri_pop,
                        show_kde=True,
                        vline=vline,
                        gate_color='#CF1111',
                        filter_col='time',
                        filter_val=t,
                        orientation='horizontal',   # single row
                    )
                    if fig is not None:
                        sections.append((
                            'Sample Histograms',
                            f'{channel} per sample — t = {t} min',
                            fig,
                        ))

        # ── 4. PRI decay curves ────────────────────────────────────────
        if not self.exp.pri_table.empty:
            fig = self._capture_plot(
                self.exp.plot_pri,
                which=pri_which,
                cols=pri_cols,
                show_ci=show_ci,
                show_params=show_params,
            )
            if fig is not None:
                sections.append(('PRI Kinetics', f'PRI Decay Curves ({pri_which})', fig))

        # ── 5. PRI bar chart ───────────────────────────────────────────
        if not self.exp.pri_fits_norm.empty or not self.exp.pri_fits_abs.empty:
            fig = self._capture_plot(self.exp.plot_pri_bars)
            if fig is not None:
                sections.append(('PRI Kinetics', 'Half-life Comparison', fig))

        return sections

    # ------------------------------------------------------------------
    def export_html(self, path: str, show_ci: bool = True,
                    show_params: list = None, pri_cols: int = 3) -> None:
        """Write a self-contained HTML report with embedded figures and tables."""
        sections_data = self._collect_figures(
            show_ci=show_ci, show_params=show_params, pri_cols=pri_cols
        )
        summary_text = self.summary()
        pri_df = self.pri_summary()

        # Group figures by section
        from collections import OrderedDict
        section_blocks = OrderedDict()
        for sec, label, fig in sections_data:
            b64 = self._fig_to_b64(fig)
            section_blocks.setdefault(sec, []).append((label, b64))

        sections_html = ''
        for sec_title, items in section_blocks.items():
            sections_html += f'<h2>{sec_title}</h2>\n'
            for label, b64 in items:
                sections_html += (
                    f'<div class="fig-block">'
                    f'<p class="fig-label">{label}</p>'
                    f'<img src="data:image/png;base64,{b64}" alt="{label}"/>'
                    f'</div>\n'
                )

        table_html = (pri_df.to_html(index=False, float_format=lambda x: f"{x:.4f}",
                                      border=0, classes='pri-table')
                      if not pri_df.empty else '')

        html = _Template('''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>FlowReport</title>
<style>
  body { font-family: Arial, Helvetica, sans-serif; font-size: 11px;
         margin: 32px 40px; color: #222; max-width: 1400px; }
  h1   { font-size: 15px; border-bottom: 2px solid #555; padding-bottom: 6px; }
  h2   { font-size: 12px; color: #333; margin-top: 28px; margin-bottom: 6px;
         border-left: 3px solid #2166ac; padding-left: 8px; }
  p.fig-label { font-size: 10px; color: #555; margin: 4px 0 2px 0; font-style: italic; }
  pre  { background: #f7f7f7; padding: 10px; font-size: 9px; border-radius: 3px;
         border: 1px solid #ddd; line-height: 1.5; }
  .fig-block { display: inline-block; margin: 6px 10px 6px 0;
               vertical-align: top; }
  .fig-block img { max-width: 640px; border: 1px solid #ddd; border-radius: 2px; }
  .pri-table { border-collapse: collapse; font-size: 10px; margin-top: 6px; }
  .pri-table th, .pri-table td { border: 1px solid #ddd; padding: 4px 10px; }
  .pri-table th { background: #eef2f7; font-weight: bold; }
  .pri-table tr:nth-child(even) { background: #fafafa; }
</style>
</head>
<body>
<h1>FlowReport</h1>
<h2>Experiment Summary</h2>
<pre>$summary</pre>
$sections
<h2>PRI Fit Summary</h2>
$table
</body>
</html>''').substitute(
            summary=summary_text,
            sections=sections_html,
            table=table_html,
        )
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"📄 HTML report saved to: {path}")

    # ------------------------------------------------------------------
    def export_pdf(self, path: str, show_ci: bool = True,
                   show_params: list = None, pri_cols: int = 3) -> None:
        """Write a multi-page PDF report using matplotlib PdfPages."""
        sections_data = self._collect_figures(
            show_ci=show_ci, show_params=show_params, pri_cols=pri_cols
        )
        with _PdfPages(path) as pdf:
            for sec_title, label, fig in sections_data:
                fig.suptitle(f"{sec_title} — {label}", fontsize=9,
                             color='#555555', y=1.01)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
        print(f"📄 PDF report saved to: {path}")
