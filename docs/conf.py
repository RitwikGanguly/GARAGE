# Configuration file for the Sphinx documentation builder
# See: https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the GARAGE repository root to sys.path so autodoc can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# -- Project information -----------------------------------------------------
project = 'GARAGE'
author = 'Ritwik Ganguly'
copyright = '2025, GARAGE'
version = '1.0.0'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'myst_parser',
    'sphinx.ext.githubpages',
]

templates_path = ['_templates']
exclude_patterns = [
    '_build', 'Thumbs.db', '.DS_Store',
    'ari.md', 'bioinformatics.md', 'feature.md', 'leiden.md',
    'pca.md', 'project_overview.md', 'umap.md',
]

# -- Intersphinx mappings ----------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# -- Autosummary -------------------------------------------------------------
autosummary_generate = False

# -- Autodoc options ---------------------------------------------------------
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}
autodoc_typehints = 'description'

# -- Options for Markdown files via myst-parser ------------------------------
myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
    "deflist",
    "fieldlist",
    "attrs_inline",
    "attrs_block",
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    'collapse_navigation': False,
    'navigation_depth': 4,
}

html_context = {
    'github_user': 'RitwikGanguly',
    'github_repo': 'GARAGE',
    'github_version': 'main',
    'doc_path': 'docs',
}

html_css_files = [
    'custom.css',
]

html_show_sourcelink = False

# -- Syntax highlighting -----------------------------------------------------
pygments_style = 'sphinx'

# -- TODOs -------------------------------------------------------------------
todo_include_todos = True

# -- MyST heading anchors ----------------------------------------------------
myst_heading_anchors = 3

# -- Suppress specific warnings ----------------------------------------------
suppress_warnings = [
    'myst.xref_missing',
    'autodoc.import_object',
    'autosummary.import_cycle',
]

# -- Mock heavy imports for autodoc on ReadTheDocs (no GPU/deps needed) -------
autodoc_mock_imports = [
    'torch', 'torch.nn', 'torch.optim', 'torch.cuda', 'torch.amp',
    'torch_geometric', 'torch_geometric.nn', 'torch_geometric.nn.pool',
    'torch_geometric.data', 'torch_geometric.utils',
    'scanpy', 'leidenalg',
    'ot',
    'matplotlib',
    'seaborn',
    'scipy',
    'sklearn', 'sklearn.preprocessing', 'sklearn.metrics',
    'sklearn.neighbors', 'sklearn.ensemble', 'sklearn.decomposition',
    'anndata',
    'numpy',
    'pandas',
]
