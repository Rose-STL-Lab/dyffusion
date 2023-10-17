# Lint as: python3
"""
Simple check list from AllenNLP repo: https://github.com/allenai/allennlp/blob/main/setup.py

To create the package for pypi.

1. Run `make pre-release` (or `make pre-patch` for a patch release) then run `make fix-copies` to fix the index of the
   documentation.

   If releasing on a special branch, copy the updated README.md on the main branch for the commit you will make
   for the post-release and run `make fix-copies` on the main branch as well.

2. Unpin specific versions from setup.py that use a git install.

3. Checkout the release branch (v<RELEASE>-release, for example v4.19-release), and commit these changes with the
   message: "Release: <RELEASE>" and push.

4. Wait for the tests on main to be completed and be green (otherwise revert and fix bugs)

5. Add a tag in git to mark the release: "git tag v<RELEASE> -m 'Adds tag v<RELEASE> for pypi' "
   Push the tag to git: git push --tags origin v<RELEASE>-release

6. Build both the sources and the wheel. Do not change anything in setup.py between
   creating the wheel and the source distribution (obviously).

   For the wheel, run: "python setup.py bdist_wheel" in the top level directory.
   (this will build a wheel for the python version you use to build it).

   For the sources, run: "python setup.py sdist"
   You should now have a /dist directory with both .whl and .tar.gz source versions.

   Long story cut short, you need to run both before you can upload the distribution to the
   test pypi and the actual pypi servers:

   python setup.py bdist_wheel && python setup.py sdist

8. Check that everything looks correct by uploading the package to the pypi test server:

   twine upload dist/* -r pypitest
   (pypi suggest using twine as other methods upload files via plaintext.)
   You may have to specify the repository url, use the following command then:
   twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/

   Check that you can install it in a virtualenv by running:
   pip install -i https://testpypi.python.org/pypi dyffusion

   If you are testing from a Colab Notebook, for instance, then do:
   pip install dyffusion && pip uninstall dyffusion
   pip install -i https://testpypi.python.org/pypi dyffusion

   Check you can run the following commands:
   python -c "python -c "from dyffusion import __version__; print(__version__)"
   python -c "from dyffusion import *"

9. Upload the final version to actual pypi:
   twine upload dist/* -r pypi

10. Prepare the release notes and publish them on github once everything is looking hunky-dory.

11. Run `make post-release` (or, for a patch release, `make post-patch`). If you were on a branch for the release,
    you need to go back to main before executing this.
"""

import re

# Import command from setuptools instead of distutils.core.Command for compatibility with Python>3.12
from setuptools import Command, find_packages, setup


# IMPORTANT:
# 1. all dependencies should be listed here with their version requirements if any
# 2. once modified, run: `make deps_table_update` to update src/dyffusion/dependency_versions_table.py
_deps = [
    "black~=23.1",
    "dask",
    "einops",
    "hf-doc-builder>=0.3.0",
    "hydra-core",
    "isort>=5.5.4",
    "netCDF4",
    "numpy",
    "omegaconf",
    "pytest",
    "pytorch-lightning>=2.0",
    "rich",
    "ruff>=0.0.241",
    "regex!=2019.12.17",
    "requests",
    "tensordict",
    "torch>=1.8",
    "torchmetrics==0.11.4",
    "urllib3",
    "wandb",
    "xarray",
    "xskillscore",
]

# this is a lookup table with items like:
#
# packaging: "packaging"
#
# some of the values are versioned whereas others aren't.
deps = {b: a for a, b in (re.findall(r"^(([^!=<>~]+)(?:[!=<>~].*)?$)", x)[0] for x in _deps)}

# since we save this data in src/dependency_versions_table.py it can be easily accessed from
# anywhere. If you need to quickly access the data from this table in a shell, you can do so easily with:
#
# python -c 'import sys; from dyffusion.dependency_versions_table import deps; \
# print(" ".join([ deps[x] for x in sys.argv[1:]]))' tokenizers datasets
#
# Just pass the desired package names to that script as it's shown with 2 packages above.
#
# If dyffusion is not yet installed and the work is done from the cloned repo remember to add `PYTHONPATH=src` to the script above
#
# You can then feed this for example to `pip`:
#
# pip install -U $(python -c 'import sys; from dyffusion.dependency_versions_table import deps; \
# print(" ".join([ deps[x] for x in sys.argv[1:]]))' tokenizers datasets)
#


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


class DepsTableUpdateCommand(Command):
    """
    A custom distutils command that updates the dependency table.
    usage: python setup.py deps_table_update
    """

    description = "build runtime dependency table"
    user_options = [
        # format: (long option, short option, description).
        ("dep-table-update", None, "updates src/dependency_versions_table.py"),
    ]

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        entries = "\n".join([f'    "{k}": "{v}",' for k, v in deps.items()])
        content = [
            "# THIS FILE HAS BEEN AUTOGENERATED. To update:",
            "# 1. modify the `_deps` dict in setup.py",
            "# 2. run `make deps_table_update``",
            "deps = {",
            entries,
            "}",
            "",
        ]
        target = "src/dependency_versions_table.py"
        print(f"updating {target}")
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(content))


extras = {}  # defaultdict(list)
extras["quality"] = deps_list("urllib3", "black", "isort", "ruff", "hf-doc-builder")
extras["docs"] = deps_list("hf-doc-builder")
extras["test"] = deps_list("pytest")
extras["run"] = deps_list("xarray", "netCDF4", "dask", "einops", "hydra-core", "wandb", "xskillscore")
extras["torch"] = deps_list("torch", "pytorch-lightning", "torchmetrics", "tensordict")
extras["train"] = extras["torch"] + extras["run"]
extras["optional"] = deps_list("rich")
extras["dev"] = extras["quality"] + extras["test"] + extras["train"] + extras["docs"] + extras["optional"]

install_requires = [
    deps["numpy"],
    deps["regex"],
    deps["requests"],
]

setup(
    name="dyffusion",
    version="0.0.1",  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)
    description="DYffusion: A Dynamics-informed Diffusion Model for Probabilistic Spatiotemporal Forecasting",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Salva Rühling Cachay",
    author_email="salvaruehling@gmail.com",
    url="https://github.com/Rose-STL-lab/dyffusion",
    download_url="https://github.com/Rose-STL-lab/dyffusion/tags",
    license="Apache 2.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    python_requires=">=3.8.0",
    install_requires=list(install_requires),
    extras_require=extras,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="machine learning dyffusion forecasting spatiotemporal probabilistic diffusion model",
    zip_safe=False,  # Required for mypy to find the py.typed file
    cmdclass={"deps_table_update": DepsTableUpdateCommand},
)

# Release checklist
# 1. Change the version in __init__.py and setup.py.
# 2. Commit these changes with the message: "Release: Release"
# 3. Add a tag in git to mark the release: "git tag RELEASE -m 'Adds tag RELEASE for pypi' "
#    Push the tag to git: git push --tags origin main
# 4. Run the following commands in the top-level directory:
#      python setup.py bdist_wheel
#      python setup.py sdist
# 5. Upload the package to the pypi test server first:
#      twine upload dist/* -r pypitest
#      twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/
# 6. Check that you can install it in a virtualenv by running:
#      pip install -i https://testpypi.python.org/pypi dyffusion
#      dyffusion env
#      dyffusion test
# 7. Upload the final version to actual pypi:
#      twine upload dist/* -r pypi
# 8. Add release notes to the tag in github once everything is looking hunky-dory.
# 9. Update the version in __init__.py, setup.py to the new version "-dev" and push to master
