Multiscale Finite Element method applied to solve Diffusion Equation: Multiscale Finite Element method is use to solve diffusion equation using C++ library dealii. In order to run code dealii library should be installed. Further cmake creates build files with Eclipse.



To build from the repository, execute the following commands first:

$ mkdir msfem

$ cd msfem

$ git clone https://gitlab.rrz.uni-hamburg.de/bax3843/msfem.git msfem

$ cmake -DDEAL_II_DIR=/path/to/dealii -G"Eclipse CDT4 - Unix Makefiles" ../msfem

$ make debug

$ make run


License:
Please see the file ./LICENSE.md for details


Further information:
For further information have a look at ./doc/index.html and ./doc/users/cmake.html.

Continuous Integration Status:
