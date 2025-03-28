# UQ tutorial
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/yonatank93/UQ_tutorial/main)

Tutorial to do uncertainty quantification in Python.

>Any measurement that you make without the knowledge of its uncertainty is completely
>meaningless (W. Lewin)

>No forecast is complete without an estimate of its uncertainty (Tennekes et al. 1986)

>Unknown uncertainty is worse than a large uncertainty (V. Bulatov 2019)

The focus of this tutorial is to introduce uncertainty propagation from the data to the
inferred parameters. The tutorials are presented using the Jupyter Notebooks.

## Get started
Several ways to run the Jupyter Notebook in this repository:
1. Use [Binder](https://mybinder.org/). You can go to the link and add this repository.
   Or, you can just click on the Binder banner above. Be aware that to start a session in
   Binder can take quite a long time, and it can depend on how many resources are available.
   However, this would be my preferred way, if you don't want to install more Python packages
   on your own machine.
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
   
### Using virtual environment
First, create the virtual environment, for example by executing,
``` bash
virtualenv myenv
source myenv/bin/activate  # Activate the virtual environment
```
Then, install all dependencies and add the virtual environment to jupyter notebook:
``` bash
python -m pip install -r requirements.txt
python -m ipykernel install --user --name=myenv
```
Now, the envionment is ready to use for running the notebooks in this tutorial!

If you want to remove the virtual environment:
``` bash
deactivate  # Deactivate virtual environment
jupyter kernelspec uninstall myenv  # Remove virtual environment kernel from jupyter
rm -rvf myenv  # Remove the virtual environment
```


## Notebook content
* [0_introduction.ipynb](https://github.com/yonatank93/UQ_tutorial/blob/main/0_introduction.ipynb)  
  Start with this notebook to get a basic idea of statistical modeling and uncertainty propagation.
* [1_fisher_information.ipynb](https://github.com/yonatank93/UQ_tutorial/blob/main/1_fisher_information.ipynb)  
  This notebook continues the discussion about UQ and introduces the Fisher information analysis as a
  local UQ method.
* [2_mcmc.ipynb](https://github.com/yonatank93/UQ_tutorial/blob/main/2_mcmc.ipynb)  
  This notebook contains an introduction to Markov Chain Monte Carlo (MCMC) sampling.

## Contributing guide
To contribute to this repo, fork this repo and create a new branch. Do the updating to
this new branch. Then, open a pull request to merge your changes to this repo.

## Contact
Yonatan Kurniawan (kurniawanyo@outlook.com)
