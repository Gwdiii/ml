import matplotlib.pyplot as plt

from ipywidgets import Output
from plt_overfit import overfit_example, output

plt.style.use("./deeplearning.mplstyle")

plt.close("all")
display(output) # no idea how this is supposed to work
ofit = overfit_example(False)
