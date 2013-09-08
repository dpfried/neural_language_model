from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("wordnet_similarity",
                             ["wordnet_similarity.pyx"],
                             extra_compile_args=['-fopenmp'],
                             extra_link_args=['-fopenmp'],
                             )]
)
