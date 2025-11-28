import datetime
import setuptools

version = datetime.date.today().strftime("%Y.%m.%d")

setuptools.setup(
    name="fragma",
    version=version,
    author="Flavien Loiseau",
    author_email="flavien.loiseau@ensta.fr",
    description=open("README.md", "r").readlines()[1][:-1],
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/floiseau/fragma",
    project_urls={
        "Documentation": "https://floiseau.github.io/fragma/",
    },
    packages=["fragma", "fragma.utils"],
    license="GPLv3",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "sympy",
        "scipy",
        "pyvista",
        # "gmsh",
    ],
    entry_points={
        "console_scripts": [
            "fragma = fragma:run_fragma",
        ]
    },
)
