# Computer Vision Toolkit (CVT)
A collection of some useful computer vision tools.

Please check out the [documentation](https://nburgdorfer.github.io/cvtkit/) for tutorials, explaination of various topics, and references to the library API.


## Contributing
building package for distribution:
```bash
python -m build --sdist
```

publishing to Pypi:
```bash
twine upload dist/cvtkit-<VERSION>.tar.gz 
```

publishing documentation using Mkdocs and Github Pages:
```bash
mkdocs build
mkdocs gh-deploy
```
