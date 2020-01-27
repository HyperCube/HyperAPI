import json
import setuptools
import os
from distutils.command.build_py import build_py


class hyperAPI_builder(build_py):
    # Custom class to build the HyperCube API
    def write_metadata(self):
        return "__version__ = '{version}'\n\n".format(version=self.distribution.get_version())

    def run(self):
        if not self.dry_run:
            target_dir = os.path.join(self.build_lib, self.distribution.get_name())
            self.mkpath(target_dir)

            with open(os.path.join(target_dir, 'package_metadata.py'), 'w') as meta_file:
                meta_file.write(self.write_metadata())

        build_py.run(self)
        print(self.build_lib)


with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    requirements = list(_r for _r in requirements if not _r.startswith('./'))

with open('package_metadata.json') as m:
    setup_metadata = json.load(m)

setup_metadata['version'] = os.environ.get('PACKAGE_VERSION')

setup_kwargs = {
    "long_description": "HyperCube API",
    "long_description_content_type": "text/markdown",
    "packages": setuptools.find_packages(),
    "classifiers": [
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    "install_requires": requirements,
    "include_package_data": True,
    "cmdclass": {'build_py': hyperAPI_builder},
}

setup_kwargs.update(setup_metadata)

setuptools.setup(**setup_kwargs)
