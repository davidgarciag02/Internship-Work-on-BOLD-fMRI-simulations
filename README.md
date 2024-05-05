# Installation Guide

1. Getting the package from Github
2. Installing Python
3. Installing BOLDswimsuite
4. Starting the Lessons

## 1. Getting the package from Github

### Using Github Desktop
Using Github Desktop (https://desktop.github.com/) is the easiest way to get access to the repository for those without experience with git. 

Once installed, the following steps can be followed to add the package to your computer:
1. Log In to Github Desktop.
2. At the top left (under file), click on the "Current Repository" button.
3. Change the repository to "BOLDswimsuite".
4. Clone the repository.
5. Click on "Current Branch" and switch to "development" (currently needed until the branch is merged with main). 
>Note: Make sure to know where the Github folder is one your computer, this is where the package is located and we will need it during the installation.

## 2. Installing Python

Download Python 3.10.4 at the following link:
https://www.python.org/downloads/release/python-3104/

- For Windows use the "Windows installer (64-bit)", which is the recommended option.
- For Mac, use the "macOS 64-bit universal2 installer".

## 3. Installing BOLDswimsuite

Using the command line, navigate to the location of the package (where it has been cloned either from Github Desktop or git). The directory should have "pyproject.toml" in it (and this very file, "README.md").

The package can be installed either with or without the dependencies required to run the lessons.

To install with the lesson dependencies (for new users that want to do the lessons), execute the following command (from the location of the package):
```
pip install ".[lessons]"
```

To install without the lesson dependencies, execute the following command (from the location of the package):
```
pip install .
``` 

This will install all the dependencies to the Python installation. To test if everything installed properly, from the same directory in the command line, execute the following: 

```
python .\examples\3D-ANA-MC_script.py
```

This should run a short simulation and output an image with three plots, showing the different signals.

## 5. Starting the Lessons

The lessons are made with Jupyter notebook, and so must be opened with it (it has been installed as part of the Poetry dependencies). First open Jupyter by executing the following command in the command line (in the same directory as the last two commands):

```
jupyter notebook
```

This will open Jupyter in a web browser, where you will be greeted with the UI showing the current directory. From there enter the "lessons" directory. The lessons should be listed, they can be opened in the browser. Lesson 0 briefly explains how to use Jupyter notebooks, so anyone unfamiliar with them should start there. Otherwise lesson 1 covers the first topic on BOLDswimsuite.
