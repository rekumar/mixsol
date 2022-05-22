from distutils.core import setup

setup(
    name="mixsol",
    packages=["mixsol"],
    version="0.7.1",
    license="GPLv3",
    description="Planning tool for combinatorial solution mixing. Reach target solutions from mixes of starting solutions, constrained by minimum pipetting volumes. Also aids in computing amounts of powdered reagents required to form solutions with target solutes + molarities.",
    author="Rishi E Kumar",
    author_email="rishi42@gmail.com",
    url="https://github.com/rekumar/mixsol",
    download_url="https://github.com/rekumar/mixsol/archive/refs/tags/v0.6.tar.gz",
    keywords=[
        "Chemistry",
        "Mixing",
        "Combinatoric",
        "Planning",
        "Dilution",
        "Molarity",
        "Solution",
    ],
    install_requires=["scipy", "numpy", "matplotlib", "molmass"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
    ],
)
