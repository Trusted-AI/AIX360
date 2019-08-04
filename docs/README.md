# Doc

This folder contains source code for the API documentation.

To generate the documentation in html format, run:

```
make clean && make html
```

You must have `sphinx` installed, otherwise, run:

```
pip install sphinx sphinx_rtd_theme
```

Then, open `_build/html/index.html` in a web browser to view the main page.

To document a new algorithm, add the following lines to the corresponding `.rst`
file of the base class (e.g. `lwbe.rst`):

```rst
Contrastive Explainer
---------------------------

.. autoclass:: aix360.algorithms.contrastive.CEMExplainer
   :members:
```

Note: the dashes should match the subtitle length and the class should be the
full import statement for that class.

Rebuild the docs to view your changes.

## References

* [Sphinx docs](https://www.sphinx-doc.org/en/master/index.html)

* [Docstring examples](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html#example-google)

* [Additional reST syntax](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)



