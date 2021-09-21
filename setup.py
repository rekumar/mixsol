from distutils.core import setup

setup(
    name="mixsol",  # How you named your package folder (MyLib)
    packages=["mixsol"],  # Chose the same as "name"
    version="0.1",  # Start with a small number and increase it with every change you make
    license="GNU3",  # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    description="Planning tool for combinatorial solution mixing. Reach target solutions from mixes of starting solutions, constrained by minimum pipetting volumes. Also aids in computing amounts of powdered reagents required to form solutions with target solutes + molarities.",  # Give a short description about your library
    author="Rishi E Kumar",  # Type in your name
    author_email="rishi42@gmail.com",  # Type in your E-Mail
    url="https://github.com/rekumar/reponame",  # Provide either the link to your github or to your website
    download_url="https://github.com/rekumar/mixsol/archive/refs/tags/v0.1.tar.gz",  # I explain this later on
    keywords=[
        "Chemistry",
        "Mixing",
        "Combinatoric",
        "Planning",
        "Dilution",
        "Molarity",
        "Solution",
    ],  # Keywords that define your package best
    install_requires=[  # I get to this in a second
        "scipy",
        "numpy",
        "matplotlib",
        "molmass",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",  # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        "Intended Audience :: Developers",  # Define that your audience are developers
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",  # Again, pick a license
        "Programming Language :: Python :: 3",  # Specify which pyhton versions that you want to support
    ],
)
