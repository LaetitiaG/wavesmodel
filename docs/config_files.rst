============
Config files
============

The toolbox allows you to save configurations in files in order to easily run a set of simulations only by passing a file as argument.
There exists 3 types of configuration files. The 'Entry' config file is the main one, where is stored all the information needed to run a simulation.
It is the only required argument if you run the toolbox in command line.


Entry config file
-----------------
The Entry config file is implemented using `ConfigParser <https://docs.python.org/3/library/configparser.html>`_ which provides a structure similar to what’s found in Microsoft Windows INI files.
The idea is to allow the user to easily modify these configuration files without using any code.
