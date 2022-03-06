from bokeh.plotting import figure, Figure
from bokeh.models.glyphs import Image, MultiLine
from bokeh.models import HoverTool, ColumnDataSource, TapTool, Slider, TextInput, Select, \
    BoxAnnotation, Patches
from bokeh.models.mappers import LogColorMapper
from bokeh.layouts import gridplot, column, row
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import *
import tifffile
import logging
from .core import BokehCallbackSignal
from matplotlib import cm as matplotlib_color_map

logger = logging.getLogger()
from caiman import load_memmap

_default_image_figure_params = dict(
    plot_height=500,
    plot_width=500,
    tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")],
    output_backend='webgl'
)

_default_curve_figure_params = dict(
    plot_height=250,
    plot_width=1000,
    tools='tap,hover,pan,wheel_zoom,box_zoom,reset',
)


qual_cmaps = ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1',
              'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c']


def auto_colormap(
        n_colors: int,
        cmap: str = 'hsv',
        output: str = 'mpl',
        spacing: str = 'uniform',
        alpha: float = 1.0
    ) \
        -> List[Union[np.ndarray, str]]:
    """
    If non-qualitative map: returns list of colors evenly spread through the chosen colormap.
    If qualitative map: returns subsequent colors from the chosen colormap
    :param n_colors: Numbers of colors to return
    :param cmap:     name of colormap
    :param output:   option: 'mpl' returns RGBA values between 0-1 which matplotlib likes,
                     option: 'pyqt' returns QtGui.QColor instances that correspond to the RGBA values
                     option: 'bokeh' returns hex strings that correspond to the RGBA values which bokeh likes
    :param spacing:  option: 'uniform' returns evenly spaced colors across the entire cmap range
                     option: 'subsequent' returns subsequent colors from the cmap
    :param alpha:    alpha level, 0.0 - 1.0
    :return:         List of colors as either ``QColor``, ``numpy.ndarray``, or hex ``str`` with length ``n_colors``
    """

    valid = ['mpl', 'pyqt', 'bokeh']
    if output not in valid:
        raise ValueError(f'output must be one {valid}')

    valid = ['uniform', 'subsequent']
    if spacing not in valid:
        raise ValueError(f'spacing must be one of either {valid}')

    if alpha < 0.0 or alpha > 1.0:
        raise ValueError('alpha must be within 0.0 and 1.0')

    cm = matplotlib_color_map.get_cmap(cmap)
    cm._init()

    if output == 'pyqt':
        lut = (cm._lut * 255).view(np.ndarray)
    else:
        lut = (cm._lut).view(np.ndarray)

    lut[:, 3] *= alpha

    if spacing == 'uniform':
        if not cmap in qual_cmaps:
            cm_ixs = np.linspace(0, 210, n_colors, dtype=int)
        else:
            if n_colors > len(lut):
                raise ValueError('Too many colors requested for the chosen cmap')
            cm_ixs = np.arange(0, len(lut), dtype=int)
    else:
        cm_ixs = range(n_colors)

    colors = []
    for ix in range(n_colors):
        c = lut[cm_ixs[ix]]

        if output == 'bokeh':
            c = tuple(c[:3] * 255)
            hc = '#%02x%02x%02x' % tuple(map(int, c))
            colors.append(hc)

        else:  # mpl
            colors.append(c)

    return colors


def get_numerical_columns(dataframe: pd.DataFrame):
    return [c for c in dataframe.columns if c.startswith('_')]


def get_categorical_columns(dataframe: pd.DataFrame):
    return [c for c in dataframe.columns if not c.startswith('_')]


class DatapointTracer:
    def __init__(
            self,
            doc,
            tooltip_columns: List[str] = None,
            image_figure_params: dict = None,
    ):
        self.doc = doc

        self.sig_frame_changed = BokehCallbackSignal()

        self.frame: np.ndarray = np.empty(0)

        if image_figure_params is None:
            image_figure_params = dict()

        self.image_figure: Figure = figure(
            **{
                **_default_image_figure_params,
                **image_figure_params,
                'output_backend': "webgl"
            }
        )

        # must initialize with some array else it won't work
        empty_img = np.zeros(shape=(100, 100), dtype=np.uint8)

        self.image_glyph: Image = self.image_figure.image(
            image=[empty_img],
            x=0, y=0,
            dw=10, dh=10,
            level="image"
        )

        self.image_figure.grid.grid_line_width = 0

        self.tooltip_columns = tooltip_columns
        self.tooltips = None

        if self.tooltip_columns is not None:
            self.tooltips = [(col, f'@{col}') for col in tooltip_columns]

        # self.datatable:

        self.current_frame: int = -1
        self.tif: tifffile.TiffFile = None
        self.color_mapper: LogColorMapper = None

        self.frame_slider = Slider(start=0, end=1000, value=1, step=10, title="Frame index:")
        self.frame_slider.on_change('value', self.sig_frame_changed.trigger)
        self.sig_frame_changed.connect(self._set_current_frame)

        self.label_filesize: TextInput = TextInput(value='', title='Filesize (GB):')

    def _set_video(self, vid_path: Union[Path, str]):
        self.tif = tifffile.TiffFile(vid_path)

        self.current_frame = 0
        self.frame = self.tif.asarray(key=self.current_frame)

        # this is basically used for vmin mvax
        self.color_mapper = LogColorMapper(
            palette=auto_colormap(256, 'gnuplot2', output='bokeh'),
            low=np.nanmin(self.frame),
            high=np.nanmax(self.frame)
        )

        self.image_glyph.data_source.data['image'] = [self.frame]
        self.image_glyph.glyph.color_mapper = self.color_mapper

        # shows the file size in gigabytes
        self.label_filesize.update(value=str(os.path.getsize(vid_path) / 1024 / 1024 / 1024))

    def _set_current_frame(self, i: int):
        self.current_frame = i
        frame = self.tif.asarray(key=self.current_frame, maxworkers=20)

        self.image_glyph.data_source.data['image'] = [frame]

    def _get_trimmed_dataframe(self) -> pd.DataFrame:
        """
        Get dataframe for tooltips, JSON serializable.
        """
        return self.dataframe.drop(
            columns=[c for c in self.dataframe.columns if c not in self.tooltip_columns]
        ).copy(deep=True)

    def set_dashboard(self, figures: List[Figure]):
        logger.info('setting dashboard, this might take a few minutes')
        self.doc.add_root(
            column(
                row(*(f for f in figures), self.image_figure),
                row(
                    self.label_filesize,
                ),
                self.frame_slider
            )
        )
