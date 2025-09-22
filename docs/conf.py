# Configuration file for the Sphinx documentation builder
# See: https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
project = 'GARAGE'
author = 'Ritwik Ganguly'
copyright = '2025, GARAGE'

# -- General configuration ---------------------------------------------------
# Sphinx core extensions
extensions = [
    'sphinx.ext.autodoc',         # Document code from docstrings
    'sphinx.ext.napoleon',        # Google and NumPy docstring support
    # 'sphinx.ext.viewcode',      # Removed to disable "View page source" button
    'sphinx.ext.todo',            # Highlight TODOs in docs
    'myst_parser',                # Markdown support (optional but recommended)
    'sphinx.ext.githubpages',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for Markdown files via myst-parser ------------------------------
myst_enable_extensions = [
    "dollarmath",     # For LaTeX math in markdown files
    "colon_fence",    # For special blocks, e.g., admonitions
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Optional sidebar/logo settings for sphinx_rtd_theme
# html_logo = '_static/logo.png'  # put your logo here, optional
html_theme_options = {
    'display_version': True,      # Show version next to project name
    'collapse_navigation': False, # Expand sidebar by default

    # GitHub options
    'github_url': 'https://github.com/RitwikGanguly/GARAGE',
    'use_edit_page_button': True,  # Shows "Edit on GitHub" button
}

html_context = {
    'github_user': 'RitwikGanguly',
    'github_repo': 'GARAGE',
    'github_version': 'main',      # Branch name, e.g., 'main' or 'master'
    'doc_path': 'docs',            # Path to your docs directory in the repo
}

# -- Remove "View page source" button ---------------------------------------
html_show_sourcelink = False      # This disables the top-right "View page source" button

# -- Extra: Support for code highlighting ------------------------------------
pygments_style = 'sphinx'        # Syntax highlighting for code blocks

# -- Extra: TODOs support ----------------------------------------------------
todo_include_todos = True

# -- Optional: Add custom CSS
# html_css_files = [
#     'custom.css',
# ]

# For more extension options: https://www.sphinx-doc.org/en/master/usage/extensions/index.html
