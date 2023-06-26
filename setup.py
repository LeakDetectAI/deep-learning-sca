from os.path import realpath, dirname, join
from setuptools import setup, find_packages

DISTNAME = 'deepscapy'
DESCRIPTION = 'Side Channel Attacks using Deep Neural Networks'
MAINTAINER = 'Varun Nandkumar Golani'
MAINTAINER_EMAIL = 'prithag@mail.upb.de'
VERSION = "1.0"

PROJECT_ROOT = dirname(realpath(__file__))
REQUIREMENTS_FILE = join(PROJECT_ROOT, 'requirements.txt')

with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()

if __name__ == "__main__":
    setup(name=DISTNAME,
          version=VERSION,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          packages=find_packages(),
          install_requires=install_reqs,
          package_data={'notebooks': ['*']},
          include_package_data=True)
