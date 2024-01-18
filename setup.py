import setuptools

with open('README.md','r',encoding="utf-8") as f:
    long_description=f.read()




__version__= "0.0.0"

REPO_NAME='Titanic_MLOps'
AUTHOR_USER_NAME="Julien.Mahouin Mohamed.Lemrabott Ouassim.Messagier Alexis.Nolière"
SRC_REPO = "mlProject"
AUTHOR_EMAIL="Julien.Mahouin.Etu@univ-lemans.fr Mohamed.Lemrabott.Etu@univ-lemans.fr Ouassim.Messagier.Etu@univ-lemans.fr Alexis.Nolière.Etu@univ-lemans.fr"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description='a small  python package for  mlops ',
    long_description=long_description,
    long_destription_content='text/markdown',
    url="https://git.univ-lemans.fr/Aghilas.Sini/e2e-mlops",
    project_urls={

        "Bug Tracker":"https://git.univ-lemans.fr/Aghilas.Sini/e2e-mlops/-/issues",

    },
    package_dir={"":"src"},
    packages=setuptools.find_packages(where='src'),


)
