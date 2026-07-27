import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from manyfunpy.mplot import paperize


def test_paperize_formats_mock_figure_text():
    fig, ax = plt.subplots()
    line, = ax.plot([1e6, 2e6], [2e6, 4e6], label="Axis series")
    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    axes_legend = ax.legend(title="Axes legend")
    figure_legend = fig.legend([line], ["Figure series"], title="Figure legend")
    figure_note = fig.text(0.5, 0.02, "Figure note", fontsize=9)
    suptitle = fig.suptitle("Mock figure", fontsize=13)

    paperize(fig, font_size=7, font_name="DejaVu Serif", zoom=2)
    fig.canvas.draw()

    for legend in [axes_legend, figure_legend]:
        for text in [legend.get_title(), *legend.get_texts()]:
            assert text.get_fontfamily() == ["DejaVu Serif"]
            assert text.get_fontsize() == 14

    for text in [ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text()]:
        assert text.get_text()
        assert text.get_fontfamily() == ["DejaVu Serif"]
        assert text.get_fontsize() == 14

    assert figure_note.get_fontfamily() == ["DejaVu Serif"]
    assert figure_note.get_fontsize() == 9
    assert suptitle.get_fontfamily() == ["DejaVu Serif"]
    assert suptitle.get_fontsize() == 13
    plt.close(fig)


def test_paperize_axes_does_not_format_figure_legend():
    fig, ax = plt.subplots()
    line, = ax.plot([0, 1], [0, 1], label="Axis series")
    axes_legend = ax.legend()
    figure_legend = fig.legend([line], ["Figure series"], fontsize=9)
    figure_legend_family = figure_legend.get_texts()[0].get_fontfamily()
    figure_legend_size = figure_legend.get_texts()[0].get_fontsize()

    paperize(ax, font_size=7, font_name="DejaVu Serif", zoom=2)

    assert axes_legend.get_texts()[0].get_fontfamily() == ["DejaVu Serif"]
    assert axes_legend.get_texts()[0].get_fontsize() == 14
    assert figure_legend.get_texts()[0].get_fontfamily() == figure_legend_family
    assert figure_legend.get_texts()[0].get_fontsize() == figure_legend_size
    plt.close(fig)
