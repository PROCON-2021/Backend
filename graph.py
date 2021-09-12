from io import BytesIO
from matplotlib import pyplot, use
use('Agg')

def graph_bin(count: int, arr1: list, arr2: list, arr3: list):
    x = [i * 16 / count for i in range(0, count)]
    graph1, graph2, graph3 = [], [], []
    pyplot.figure(figsize = (16.0, 2.5), dpi = 96)

    pyplot.clf()
    pyplot.plot(x, arr1, color = "purple")
    pyplot.xlim(0, 16)
    pyplot.ylim(256, 767)
    png_bin = BytesIO()
    pyplot.savefig(png_bin, format = "png")
    graph1 = list(png_bin.getvalue())

    pyplot.clf()
    pyplot.plot(x, arr2, color = "green")
    pyplot.xlim(0, 16)
    pyplot.ylim(256, 767)
    png_bin = BytesIO()
    pyplot.savefig(png_bin, format = "png")
    graph2 = list(png_bin.getvalue())

    pyplot.clf()
    pyplot.plot(x, arr3, color = "blue")
    pyplot.xlim(0, 16)
    pyplot.ylim(256, 767)
    png_bin = BytesIO()
    pyplot.savefig(png_bin, format = "png")
    graph3 = list(png_bin.getvalue())
    
    return (graph1, graph2, graph3)

"""
def graph_bin(count: int, arr1: list, arr2: list, arr3: list):
    x = [i * 16 / count for i in range(0, count)]
    graph1, graph2, graph3 = [], [], []
    pyplot.figure(figsize = (8.0, 4.0), dpi = 96)

    pyplot.clf()
    pyplot.plot(x, arr1, color = "purple")
    pyplot.xlim(0, 16)
    pyplot.ylim(256, 767)
    png_bin = BytesIO()
    pyplot.savefig(png_bin, format = "png")
    graph1 = list(png_bin.getvalue())

    pyplot.clf()
    pyplot.plot(x, arr2, color = "green")
    pyplot.xlim(0, 16)
    pyplot.ylim(256, 767)
    png_bin = BytesIO()
    pyplot.savefig(png_bin, format = "png")
    graph2 = list(png_bin.getvalue())

    pyplot.clf()
    pyplot.plot(x, arr3, color = "blue")
    pyplot.xlim(0, 16)
    pyplot.ylim(256, 767)
    png_bin = BytesIO()
    pyplot.savefig(png_bin, format = "png")
    graph3 = list(png_bin.getvalue())
    
    return (graph1, graph2, graph3)
"""