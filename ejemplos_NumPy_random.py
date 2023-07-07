"""

    py -m venv matplotlib
    source matplotlib/Scripts/activate
    pip install matplotlib
    pip install numpy ##### ya instalado con matplotlib

"""
import numpy as np
from numpy import random

#Random Numbers in NumPy

# What is a Random Number?
# Random number does NOT mean a different number every time. Random means something that can not be predicted logically.

# Pseudo Random and True Random.
# Computers work on programs, and programs are definitive set of instructions. So it means there must be some algorithm to generate a random number as well.

# If there is a program to generate random number it can be predicted, thus it is not truly random.

# Random numbers generated through a generation algorithm are called pseudo random.

# Can we make truly random numbers?

# Yes. In order to generate a truly random number on our computers we need to get the random data from some outside source. This outside source is generally our keystrokes, mouse movements, data on network etc.

# We do not need truly random numbers, unless it is related to security (e.g. encryption keys) or the basis of application is the randomness (e.g. Digital roulette wheels).

# In this tutorial we will be using pseudo random numbers.

##############################################################
##############################################################

###### Generate Random Number ######
# NumPy offers the random module to work with random numbers.

# ExampleGet your own Python Server
# Generate a random integer from 0 to 100:

x = random.randint(100)

print(x) #77

#############################################################
# Generate Random Float
# The random module's rand() method returns a random float between 0 and 1.

# Example
# Generate a random float from 0 to 1:

x = random.rand()

print(x) #0.4763316183439461

#############################################################
# Generate Random Array
# In NumPy we work with arrays, and you can use the two methods from the above examples to make random arrays.

# Integers
# The randint() method takes a size parameter where you can specify the shape of an array.

# Example
# Generate a 1-D array containing 5 random integers from 0 to 100:

x=random.randint(100, size=(5))

print(x) #[67 82 56 42 42]

# Example
# Generate a 2-D array with 3 rows, each row containing 5 random integers from 0 to 100:

x = random.randint(100, size=(3, 5))

print(x)
# [[59 48 24 96  7]
#  [33 88 18 90 20]
#  [87 77 93 93 35]]

#############################################################
# Floats
# The rand() method also allows you to specify the shape of the array.

# Example
# Generate a 1-D array containing 5 random floats:

x = random.rand(5)

print(x) #[0.42451919 0.74068702 0.48121708 0.12879421 0.73576713]


# Example
# Generate a 2-D array with 3 rows, each row containing 5 random numbers:

x = random.rand(3, 5)

print(x)
# [[0.88481371 0.07590188 0.4644325  0.08502131 0.314467  ]
#  [0.2772618  0.07914512 0.81541626 0.11182125 0.0136112 ]
#  [0.22466555 0.54816722 0.74087534 0.26257796 0.82542077]]

#############################################################
# Generate Random Number From Array
# The choice() method allows you to generate a random value based on an array of values.

# The choice() method takes an array as a parameter and randomly returns one of the values.

# Example
# Return one of the values in an array:

x = random.choice([3, 5, 7, 9])

print(x) #3

# The choice() method also allows you to return an array of values.

# Add a size parameter to specify the shape of the array.

# Example
# Generate a 2-D array that consists of the values in the array parameter (3, 5, 7, and 9):

x = random.choice([3, 5, 7, 9], size=(3, 5))

print(x)
# [[7 7 3 5 5]
#  [9 5 7 5 9]
#  [7 7 9 7 9]]

##############################################################
##############################################################