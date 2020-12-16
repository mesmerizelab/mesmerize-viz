from typing import *
from .signal import BokehCallbackSignal


class WebPlot:
    def __init__(self, *args, **kwargs):
        attrs = dir(self)
        self.signals: List[BokehCallbackSignal] = \
            [getattr(self, a) for a in attrs if isinstance(getattr(self, a), BokehCallbackSignal)]
        print("WEBPLOT INIT")
        print(self.signals)

    @classmethod
    def signal_blocker(cls, func):
        """Block callbacks, used when the plot x and y limits change due to user interaction"""
        print("***** SIGNAL BLOCKER CALLED ******")
        def fn(self, *args, **kwargs):
            print("self.signals is")
            print(self.signals)
            for signal in self.signals:
                print(f"Blocking signal {signal}")
                signal.pause()

            ret = func(self, *args, **kwargs)

            for signal in self.signals:
                signal.unpause()

            return ret
        return fn
