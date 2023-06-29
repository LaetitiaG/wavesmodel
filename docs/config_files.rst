============
Config files
============

The toolbox allows you to save configurations in files in order to easily run a set of simulations only by passing a file as argument.
There exists 3 types of configuration files. The 'Entry' config file is the main one, where is stored all the information needed to run a simulation.
It is the only required argument if you run the toolbox in command line.


Entry config file
-----------------
The Entry config file is implemented using `ConfigParser <https://docs.python.org/3/library/configparser.html>`_ which provides a structure similar to what is found in Microsoft Windows INI files.
The idea is to allow the user to easily modify these configuration files without using any code.


Parameters config file
----------------------
In order to save different parameters configuration, to easily loads them in different entries of to share them, you can create configuration files for both Simulation or Screen parameters.

Simulation configuration file
"""""""""""""""""""""""""""""
The structure of the simulation configuration file is really simple. It follow the structure of Simulation_params (ADD REF)

.. code-block::
    [simulation]
    freq_temp = 5
    freq_spacial = 0.05
    amplitude = 10e-9
    phase_offset = pi / 2

Screen configuration file
"""""""""""""""""""""""""""""
In the same way, structure of the screen configuration file follows the structure of Screen_params (ADD REF)

.. code-block::
    [screen]
    width = 1920
    height = 1080
    distancefrom = 78
    heightcm = 44.2
