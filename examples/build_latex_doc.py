import sys
import os
import numpy as np

def main(data):
    subfigs, views = data.shape

    latex_doc_str = "\\documentclass{article}\n" + \
                    "\\usepackage[utf8]{inputenc}\n" + \
                    "\\usepackage{graphicx}\n" + \
                    "\\usepackage{subcaption}\n" + \
                    "\\graphicspath{ {../} }\n" + \
                    "\\begin{document}\n"

    for v in range(views):
        figure_data = []

        for s in range(subfigs):
            figure_data.append(data[s,v])

        captions = ["error","ground truth","image","Analysis of Mask Error for View {}".format(v)]
        labels = ["error{}".format(v),"gt{}".format(v),"img{}".format(v),"error_analys{}".format(v)]
        latex_fig = create_subfigures(figure_data, captions, labels)

        #print(latex_fig)
        latex_doc_str += latex_fig

    latex_doc_str += "\\end{document}\n"

    return latex_doc_str



if __name__=="__main__":
    main()
