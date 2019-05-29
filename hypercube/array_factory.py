#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016 SKA South Africa
#
# This file is part of hypercube.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.

from weakref import WeakKeyDictionary

import functools

import numpy as np

class ArrayDescriptor(object):
    """ Descriptor class for arrays """
    def __init__(self, name):
        self.data = WeakKeyDictionary()
        self.name = name

    def __get__(self, instance, owner):
        return self.data[instance]

    def __set__(self, instance, value):
        self.data[instance] = value

    def __delete__(self, instance):
        del self.data[instance]

def gpuarray_factory(shape, dtype):
    import pycuda.gpuarray as gpuarray
    return gpuarray.empty(shape=shape, dtype=dtype)

def generic_stitch(cube, arrays):
    """
    Creates descriptors associated with array name and
    then sets the array as a member variable
    """
    
    for name, ary in arrays.items():
        if name not in type(cube).__dict__:
            setattr(type(cube), name, ArrayDescriptor(name))

        setattr(cube, name, ary)

def create_local_arrays(reified_arrays, array_factory=None):
    """
    Function that creates arrays, given the definitions in
    the reified_arrays dictionary and the array_factory
    keyword argument.

    Arguments
    ---------
        reified_arrays : dictionary
            Dictionary keyed on array name and array definitions.
            Can be obtained via cube.arrays(reify=True)

    Keyword Arguments
    -----------------
        array_factory : function
            A function used to create array objects. It's signature should
            be array_factory(shape, dtype) and should return a constructed
            array of the supplied shape and data type. If None,
            numpy.empty will be used.

    Returns
    -------
    A dictionary of array objects, keyed on array names
    """


    # By default, create numpy arrays
    if array_factory is None:
        array_factory = np.empty

    # Construct the array dictionary by calling the
    # array_factory for each array
    return { n: array_factory(ra.shape, ra.dtype)
        for n, ra in reified_arrays.items() }
    
def create_local_arrays_on_cube(cube, reified_arrays=None, array_stitch=None, array_factory=None):
    """
    Function that creates arrays on the supplied hypercube, given the supplied
    reified_arrays dictionary and array_stitch and array_factory functions.

    Arguments
    ---------
        cube : HyperCube
            A hypercube object on which arrays will be created.

    Keyword Arguments
    -----------------
        reified_arrays : dictionary
            Dictionary keyed on array name and array definitions.
            If None, obtained from cube.arrays(reify=True)
        array_stitch : function
            A function that stitches array objects onto the cube object.
            It's signature should be array_stitch(cube, arrays)
            where cube is a HyperCube object and arrays is a
            dictionary containing array objects keyed by their name.
            If None, a default function will be used that creates
            python descriptors associated with the individual array objects.
        array_factory : function
            A function that creates array objects. It's signature should
            be array_factory(shape, dtype) and should return a constructed
            array of the supplied shape and data type. If None,
            numpy.empty will be used.

    Returns
    -------
    A dictionary of array objects, keyed on array names

    """

    # Create a default array stitching method
    if array_stitch is None:
        array_stitch = generic_stitch

    # Get reified arrays from the cube if necessary
    if reified_arrays is None:
        reified_arrays = cube.arrays(reify=True)

    arrays = create_local_arrays(reified_arrays, array_factory=array_factory)
    array_stitch(cube, arrays)

    return arrays    

create_local_numpy_arrays_on_cube = functools.partial(
    create_local_arrays_on_cube,
    array_stitch=generic_stitch,
    array_factory=np.empty)

create_local_pycuda_arrays_on_cube = functools.partial(
    create_local_arrays_on_cube,
    array_stitch=generic_stitch,
    array_factory=gpuarray_factory)