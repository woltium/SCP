from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
    name="ikfastpy",
    ext_modules=cythonize([Extension("ikfastpy", 
                                    ["ikfastpy.pyx", 
                                     "ikfast_wrapper.cpp"], 
                                    language="c++", 
                                    libraries=['lapack'])]),
)