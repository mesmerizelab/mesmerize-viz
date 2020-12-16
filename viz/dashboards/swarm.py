from ..core import BokehCallbackSignal, WebPlot
from . import defaults
import pandas
from typing import *

from bokeh.plotting import figure, Figure, gridplot, curdoc
from bokeh.models import HoverTool, ColumnDataSource, TapTool, Slider, TextInput, Select
from bokeh.transform import jitter
from bokeh.layouts import gridplot, column, row


class Swarm(WebPlot):

    def __init__(
            self,
            doc,
            dataframe: pandas.DataFrame,
            data_column: str,
            groupby_column: str,
            figure_opts: dict = None,
            glyph_opts: dict = None,
            tooltip_columns: List[str] = None,
            source_columns: List[str] = None,
    ):
        self.sig_point_selected = BokehCallbackSignal()
        WebPlot.__init__(self)

        self.doc = doc

        # just setting some attributes
        self.dataframe: pandas.DataFrame = dataframe
        self.data_column = data_column
        self.groupby_column = groupby_column

        if source_columns is None:
            source_columns = []

        # ColumnDataSource is what bokeh uses for plotting
        # it's similar to dataframes but doesn't accept
        # some datatypes like dicts and arrays within dataframe "cells"
        self.source: ColumnDataSource = ColumnDataSource(
            self.dataframe.drop(
                columns=[c for c in self.dataframe.columns if c not in source_columns]
            )
        )

        # need to clean this up to get it within the base WebPlot.__init__()
        # for signal in self.signals:
        self.sig_point_selected.dataframe = self.dataframe
        self.sig_point_selected.source_data = self.source

        # columns used for displaying tooltips
        self.tooltip_columns = tooltip_columns
        self.tooltips = None

        # formatting the display of the tooltips
        if self.tooltip_columns is not None:
            self.tooltips = [(col, f'@{col}') for col in tooltip_columns]

        if figure_opts is None:
            figure_opts = dict()

        # create a figure, combine the default plot opts and any user specific plot opts
        self.figure = figure(
            **{
                **defaults.figure_opts,
                **figure_opts
            },
            x_range=self.dataframe[self.groupby_column].unique(),
            tooltips=self.tooltips
        )

        if glyph_opts is None:
            glyph_opts = dict()

        # jitter along the x axis for the swarm scatter
        x_vals = jitter(self.groupby_column, width=0.6, range=self.figure.x_range)

        # the swarm plot itself
        self.glyph = self.figure.circle(
            x=x_vals,
            y=self.data_column,  # the user specified data column
            source=self.source,  # this is the ColumnDataSource created from the dataframe
            **{
                **defaults.glyph_opts,
                **glyph_opts
            }
        )

        self.source_columns = source_columns

        self.label_point_clicked: TextInput = TextInput(value='', title='Point clicked: ')

    def _update_label(self, text: str):
        self.label_point_clicked.update(value=text)

    def start_app(self, doc):
        """
        Call this from ``bokeh.io.show()`` within a notebook to show the plot
        Can also be allowed standalone from a bokeh server

        :param doc:
        :return:
        """

        print("starting app")

        # when a point is clicked on the scatter plot it will trigger ``sig_point_selected``
        self.glyph.data_source.selected.on_change('indices', self.sig_point_selected.trigger)

        # when ``sig_point_selected`` is triggered it will update the text entry
        self.sig_point_selected.connect_data(self._update_label, 'uuid')

        self.doc.add_root(
            column(
                row(*f for f in [self.figure]),
                self.label_point_clicked
            )
        )
