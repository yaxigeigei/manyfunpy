"""
Publication-oriented figure making utilities for matplotlib.

This module provides functions to format matplotlib figures and axes
according to publication standards, similar to MATLAB's plotting utilities.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from typing import Union, Optional, List, Tuple
import numpy as np


def paperize(
    h: Optional[Union[plt.Figure, plt.Axes, List[Union[plt.Figure, plt.Axes]]]] = None,
    cols_wide: Optional[float] = None,
    cols_high: Optional[float] = None,
    font_size: float = 6,
    font_name: str = 'DejaVu Sans',
    zoom: float = 2,
    aspect_ratio: Optional[float] = None,
    journal_style: str = 'cell'
) -> None:
    """
    Make axes comply with conventions of publication.
    
    Parameters
    ----------
    h : Figure, Axes, or list of Figure/Axes, optional
        Figure or axes handle(s). If None, uses current figure.
        If empty list, operates on all existing axes.
    cols_wide : float, optional
        Number of columns wide for the figure.
    cols_high : float, optional
        Number of columns high for the figure.
    font_size : float, default 6
        Font size for text elements.
    font_name : str, default 'Arial'
        Font family name.
    zoom : float, default 2
        Zoom factor for figure size and font size.
    aspect_ratio : float, optional
        Aspect ratio for the figure (height/width).
    journal_style : {'nature', 'cell'}, default 'cell'
        Journal style for predefined column widths.
        
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> paperize()  # Apply to current figure
    >>> paperize(fig, cols_wide=1)  # Single column width
    >>> paperize(ax, font_size=8)  # Apply to specific axes
    """
    
    # Get journal-specific column widths
    journal_dims = get_journal_dimensions(journal_style)
    width_set = [journal_dims['single'], journal_dims['double']]
    width_set = [x / 25.4 for x in width_set] # convert mm to inches
    
    # Handle default figure/axes
    if h is None:
        h = plt.gcf()
    elif h == []:
        # Get all existing axes
        h = plt.get_fignums()
        h = [plt.figure(num) for num in h]
        all_axes = []
        for fig in h:
            all_axes.extend(fig.get_axes())
        h = all_axes
    
    # Ensure h is a list
    if not isinstance(h, list):
        h = [h]
    
    # Calculate figure width if cols_wide is specified
    fig_width = None
    if cols_wide is not None:
        fig_width = width_set[0] * cols_wide
    
    for handle in h:
        if isinstance(handle, plt.Figure):
            fig = handle
            
            if fig_width is not None:
                # Set figure properties
                fig.patch.set_facecolor('white')
                
                # Apply zoom factor
                width_inches = fig_width * zoom
                
                if cols_high is not None:
                    # Set height based on number of columns
                    height_inches = (fig_width / cols_wide * cols_high * zoom)
                elif aspect_ratio is not None:
                    # Set height based on aspect ratio
                    height_inches = (fig_width * aspect_ratio * zoom)
                else:
                    # Keep current height (default)
                    height_inches = fig.get_size_inches()[1]
                
                fig.set_size_inches(width_inches, height_inches)
            
            # Get all axes in the figure
            axes_list = fig.get_axes()
        
        elif isinstance(handle, plt.Axes):
            axes_list = [handle]
        
        else:
            raise TypeError("Handle must be Figure or Axes object")
        
        # Apply formatting to axes
        for ax in axes_list:
            # Set tick direction
            ax.tick_params(direction='out', which='both')
            
            # Set font properties
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                        ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontfamily(font_name)
                item.set_fontsize(font_size * zoom)
            
            # Set spine properties
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)


def plot_interval_blocks(
    ax,
    x_ranges,
    y_centers=None,
    heights=None,
    y_ranges=None,
    colors=None,
    alpha=1.0,
    edgecolor="none",
    linewidth=0,
    zorder=None,
):
    """
    Plot interval blocks as a patch collection of rectangles.
    """
    # Normalize interval coordinates.
    x_ranges = np.asarray(x_ranges, dtype=float)
    if y_ranges is None:
        y_centers = np.asarray(y_centers, dtype=float)
        heights = np.asarray(heights, dtype=float)
        y_ranges = np.column_stack((y_centers - heights / 2, y_centers + heights / 2))
    else:
        y_ranges = np.asarray(y_ranges, dtype=float)

    # Normalize face colors.
    facecolors = mpl.colors.to_rgba_array(colors if colors is not None else (0.5, 0.5, 0.5))
    if len(facecolors) == 1:
        facecolors = np.repeat(facecolors, len(x_ranges), axis=0)

    # Build rectangle patches.
    rectangles = [
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
        )
        for (x0, x1), (y0, y1) in zip(x_ranges, y_ranges)
    ]

    # Add the patch collection to the axis.
    collection = PatchCollection(
        rectangles,
        facecolors=facecolors,
        edgecolors=edgecolor,
        linewidths=linewidth,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_collection(collection)
    return collection

def get_journal_dimensions(journal_style: str = 'cell') -> dict:
    """
    Get standard column widths for different journals.
    
    Parameters
    ----------
    journal_style : {'nature', 'cell'}, default 'cell'
        Journal style.
        
    Returns
    -------
    dict
        Dictionary containing column widths in mm.
    """
    journal_styles = {
        'nature': {
            'single': 90,      # mm (single column)
            'double': 180,     # mm (double column)
            'full_depth': 170  # mm (full page depth)
        },
        'cell': {
            'single': 85,      # mm (single column)
            'intermediate': 114, # mm (1.5 column)
            'double': 174      # mm (double column)
        }
    }
    
    journal_style = journal_style.lower()
    if journal_style not in journal_styles:
        raise ValueError(f"journal_style must be one of {list(journal_styles.keys())}")
    
    return journal_styles[journal_style]


def axxplane(ax, coord, color=None, alpha=None, ylim=None, zlim=None):
    """
    Plot planes at a specified value (x = constant)
    """
    if ylim is None: ylim = ax.get_ylim()
    if zlim is None: zlim = ax.get_zlim()
    Y_plane, Z_plane = np.meshgrid(ylim, zlim)
    X_plane = np.full_like(Y_plane, float(coord))
    return axplane(ax, X_plane, Y_plane, Z_plane, color=color, alpha=alpha)

def axyplane(ax, coord, color=None, alpha=None, xlim=None, zlim=None):
    """
    Plot planes at a specified value (y = constant)
    """
    if xlim is None: xlim = ax.get_xlim()
    if zlim is None: zlim = ax.get_zlim()
    X_plane, Z_plane = np.meshgrid(xlim, zlim)
    Y_plane = np.full_like(X_plane, float(coord))
    return axplane(ax, X_plane, Y_plane, Z_plane, color=color, alpha=alpha)

def axzplane(ax, coord, color=None, alpha=None, xlim=None, ylim=None):
    """
    Plot planes at a specified value (z = constant)
    """
    if xlim is None: xlim = ax.get_xlim()
    if ylim is None: ylim = ax.get_ylim()
    X_plane, Y_plane = np.meshgrid(xlim, ylim)
    Z_plane = np.full_like(X_plane, float(coord))
    return axplane(ax, X_plane, Y_plane, Z_plane, color=color, alpha=alpha)

def axplane(ax, X_plane, Y_plane, Z_plane, color, alpha):
    if color is None:
        color = (0.5, 0.5, 0.5)
    if alpha is None:
        alpha = 0.25
    surf = ax.plot_surface(
        X_plane,
        Y_plane,
        Z_plane,
        color=color,
        alpha=alpha,
        linewidth=0,
        antialiased=False,
        shade=False,
        clip_on=False,
    )
    return surf
