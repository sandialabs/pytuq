# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack.package import *


class PyPytuq(PythonPackage):
    """Python-only set of tools for uncertainty quantification."""

    homepage = "https://sandialabs.github.io/pytuq/"
    pypi = "pytuq/pytuq-1.0.0.tar.gz"
    git = "https://github.com/sandialabs/pytuq.git"

    maintainers("baillo")

    license("BSD-3-Clause", checked_by="baillo")

    version("1.0.0", sha256="1fc9fabf7bf183d38e104564e99d1950f7e2103baac5a13960c356173b9997ff")

    depends_on("python@3.8:", type=("build", "run"))
    depends_on("py-setuptools", type="build")
    depends_on("py-wheel", type="build")

    depends_on("py-numpy", type=("build", "run"))
    depends_on("py-scipy", type=("build", "run"))
    depends_on("py-matplotlib", type=("build", "run"))

    # Optional dependencies
    variant("nn", default=False, description="Enable neural network support")
    variant("optim", default=False, description="Enable optimization support")

    depends_on("py-torch", type=("build", "run"), when="+nn")
    depends_on("py-uqinn", type=("build", "run"), when="+nn")
    depends_on("py-pyswarms", type=("build", "run"), when="+optim")
