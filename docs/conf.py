import sphinx_rtd_theme

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Kolmogorov Arnold Network'
copyright = '2024, Ziming Liu'
author = 'Ziming Liu'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx_rtd_theme",
              "sphinx.ext.autodoc",
              "sphinx.ext.autosectionlabel"
              ]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip

def setup(app):
    app.connect("autodoc-skip-member", skip)
    
autodoc_mock_imports = ["numpy",
                        "torch",
                        "torch.nn",
                        "matplotlib.pyplot",
                        "tqdm",
                        "sympy",
                        "scipy",
                        "sklearn.linear_model"
                        "torch.optim"]
           

source_suffix = [".rst", ".md"]
#source_suffix = [".rst", ".md", ".ipynb"]
#source_suffix = {
#    '.rst': 'restructuredtext',
#    '.ipynb': 'myst-nb',
#    '.myst': 'myst-nb',
#}
