# coding: utf-8
#
# Copyright 2018 Paul Andrey
#
# This file is part of ac2art.
#
# ac2art is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ac2art is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ac2art.  If not, see <http://www.gnu.org/licenses/>.

"""Setup for the installation of the 'ac2art' package."""

import setuptools
from setuptools.command.install import install


from preinstall_checks import main as preinstall_checks


class Installer(install):
    """Define an installation protocol by overriding setuptools' one."""

    def run(self):
        """Run the pre-installation tests, then the installation."""
        preinstall_checks()
        self.do_egg_install()


setuptools.setup(
    name='ac2art',
    version='0.1',
    packages=setuptools.find_packages(),
    package_data={'ac2art': ['../config.json', '../phone_symbols.csv']},
    include_package_data=True,
    author='Paul Andrey',
    description='acoustic-to-articulatory inversion using neural networks',
    license='GPLv3',
    install_requires=[
        'h5features >= 1.2',
        'numpy >= 1.12',
        'pandas >= 0.20',
        'scipy >= 1.0',
        'sphfile >= 1.0',
        'tensorflow >= 1.8'
    ],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Operating System :: Linux"
    ],
    cmdclass={'install': Installer}
)
