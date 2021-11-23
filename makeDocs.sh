#!/bin/sh
set -e -o pipefail
sphinx-apidoc -H Preprocess -f -o ./docs/src/preprocess/ ./src/preprocess/
sphinx-apidoc -H AC-GAN -f -o ./docs/src/acgan/ ./src/acgan/
sphinx-apidoc -H AC-AnoGAN -f -o ./docs/src/acanogan/ ./src/acanogan/
sphinx-apidoc -H Experiments -f -o ./docs/src/experiments/ ./src/experiments/
# sphinx-apidoc -H "Python Module" -f -o ./docs/src/ ./src/
sphinx-build ./docs ./docs/_build
