"""
Notebook helper: run a callable with tqdm and logging output separated into
two ipywidgets Output areas so they never collide.

Usage
-----
    from notebooks._widget_run import run_with_split_output

    result = run_with_split_output(
        build_constant_chain_analysis, ds, cfg,
        log_level=logging.INFO,
    )

The function displays:
  - A "progress" Output widget whose content is refreshed in-place (tqdm bar).
  - A scrollable "log" Output widget that accumulates log lines above it.
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from typing import Any, Callable

import ipywidgets as widgets
from IPython.display import display
from tqdm.contrib.logging import logging_redirect_tqdm


class _TqdmWidget:
    """Redirect tqdm writes to a single Output widget (overwrites each time)."""

    def __init__(self, out: widgets.Output) -> None:
        self._out = out

    def write(self, msg: str) -> None:
        # tqdm.auto in notebooks writes HTML; we want plain text in the widget
        if msg.strip():
            self._out.clear_output(wait=True)
            with self._out:
                print(msg, end="")

    def flush(self) -> None:
        pass


class _LogWidgetHandler(logging.Handler):
    """Append each log record as a line in a scrollable Output widget."""

    def __init__(self, out: widgets.Output, fmt: str) -> None:
        super().__init__()
        self.setFormatter(logging.Formatter(fmt))
        self._out = out

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        with self._out:
            print(msg)


@contextmanager
def _redirect_tqdm_to_widget(out: widgets.Output):
    """Patch tqdm's file handle so bar writes go to *out* instead of stderr."""
    import tqdm as _tqdm_module  # noqa: PLC0415

    original_file = _tqdm_module.tqdm.fp if hasattr(_tqdm_module.tqdm, "fp") else None
    fake_file = _TqdmWidget(out)

    # tqdm.auto picks up sys.stderr; replace temporarily
    _orig_stderr = sys.stderr
    sys.stderr = fake_file  # type: ignore[assignment]
    try:
        yield
    finally:
        sys.stderr = _orig_stderr


def run_with_split_output(
    fn: Callable[..., Any],
    /,
    *args: Any,
    log_level: int = logging.INFO,
    log_format: str = "%(name)s  %(levelname)s: %(message)s",
    progress_height: str = "60px",
    log_height: str = "300px",
    **kwargs: Any,
) -> Any:
    """Call *fn*(*args*, **kwargs*) with tqdm and logging in separate widgets.

    Parameters
    ----------
    fn:
        The callable to run (e.g. ``build_constant_chain_analysis``).
    *args, **kwargs:
        Forwarded to *fn*.
    log_level:
        Minimum log level to capture (default ``logging.INFO``).
    log_format:
        Python logging format string for log lines.
    progress_height:
        CSS height of the tqdm widget area.
    log_height:
        CSS height of the scrollable log widget area.

    Returns
    -------
    Whatever *fn* returns.
    """
    progress_out = widgets.Output(
        layout=widgets.Layout(
            padding="4px 8px",
            height=progress_height,
            overflow_y="auto",
        )
    )
    log_out = widgets.Output(
        layout=widgets.Layout(
            padding="4px 8px",
            height=log_height,
            overflow_y="auto",
            font_family="monospace",
            font_size="12px",
        )
    )

    # Two separate Accordions stacked so both can be open simultaneously.
    progress_acc = widgets.Accordion(children=[progress_out])
    progress_acc.set_title(0, "Progress")
    progress_acc.selected_index = 0  # open by default

    log_acc = widgets.Accordion(children=[log_out])
    log_acc.set_title(0, "Log output")
    log_acc.selected_index = 0  # open by default

    ui = widgets.VBox([progress_acc, log_acc])
    display(ui)

    handler = _LogWidgetHandler(log_out, log_format)
    handler.setLevel(log_level)

    root_logger = logging.getLogger()
    prev_level = root_logger.level
    root_logger.setLevel(log_level)
    root_logger.addHandler(handler)

    try:
        with logging_redirect_tqdm(loggers=[root_logger]):
            # tqdm.auto renders as an ipywidget in Jupyter; redirect its plain-text
            # fallback stderr so terminal-style output also lands in progress_out.
            with _redirect_tqdm_to_widget(progress_out):
                # Re-route tqdm's own ipywidget display into progress_out
                with progress_out:
                    return fn(*args, **kwargs)
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(prev_level)
