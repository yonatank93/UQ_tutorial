# UQ tutorial
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/yonatank93/UQ_tutorial/main)

Tutorial to do MCMC sampling in Python.

>Any measurement that you make without the knowledge of its uncertainty is completely
>meaningless (W. Lewin)

>No forecast is complete without an estimate of its uncertainty (Tennekes et al. 1986)

The focus of this tutorial is to introduce uncertainty propagation from the data to the
inferred parameters. The tutorials are presented using the Jupyter notebooks.

## Get started
Several ways to run the Jupyter notebook in this repository:
1. Use [Binder](https://mybinder.org/). You can go to the link and add this repository.
   Or, you can just click on the Binder banner above. Be aware that to start a session in
   Binder can take quite a long time, and it can depends on how many resources are available.
   However, this would be my preferred way, if you don't want to install more Python packages
   in your own machine.
2. Use Github Codespaces. On the repository page, you can access codespaces under the Code
   tab. There will be 2 subtabs there: Local and Codespaces, which is the one that you want.
   Then, click on the (+) sign to create your codespace. This will open a web version of VS
   Code and install all the required packages. Before you can run the notebook, follow the
   suggestion that VS Code gives, such as installing Python and Jupyter. You can get the
   suggestion by opening the notebook and trying to run a cell.
   Note: I think the main intention of codespaces is to help development in the cloud.
3. Clone the repo to your own machine and run it locally. All the required packages are
   listed inside requirements.txt. To install these packages,
     ```bash
     git clone https://github.com/yonatank93/UQ_tutorial.git
     cd UQ_tutorial
     python3 -m pip install -r requirements.txt
     ```
   Then, you can just run Jupyter locally on your machine.
   
## Notebook content
* [0_introduction.ipynb](https://github.com/yonatank93/UQ_tutorial/blob/main/0_introduction.ipynb)  
  Start with this notebook to get a basic idea on statistical modeling and uncertainty propagation.

## Contact
Yonatan Kurniawan (kurniawanyo@outlook.com)
