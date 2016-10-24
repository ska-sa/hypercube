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

import collections
import itertools
import sys
import types
from weakref import WeakKeyDictionary

import numpy as np
from attrdict import AttrDict
from tabulate import tabulate

from hypercube.dims import create_dimension, Dimension
import hypercube.util as hcu

LOCAL_SIZE = 'local_size'
GLOBAL_SIZE = 'global_size'

class PropertyDescriptor(object):
    """ Descriptor class for properties """
    def __init__(self, record_key, default=None, ):
        self.default = default
        self.record_key = record_key
        self.data = WeakKeyDictionary()

    def __get__(self, instance, owner):
        return self.data.get(instance,self.default)

    def __set__(self, instance, value):
        dtype = instance._properties[self.record_key].dtype
        self.data[instance] = dtype(value)

    def __delete__(self, instance):
        del self.data[instance]

class HyperCube(object):
    """ Hypercube. """

    def __init__(self, *args, **kwargs):
        """
        Hypercube Constructor
        """

        # Dictionaries to store records about our
        # dimensions, arrays and properties
        self._dims = collections.OrderedDict()
        self._arrays = collections.OrderedDict()
        self._properties = collections.OrderedDict()

        # Register any dimensions, arrays and
        # properties we're provided
        dims = kwargs.get('dimensions', None)
        arrays = kwargs.get('arrays', None)
        properties = kwargs.get('properties', None)

        if dims is not None:
            self.register_dimensions(dims)

        if arrays is not None:
            self.register_arrays(arrays)

        if properties is not None:
            self.register_properties(properties)


    def bytes_required(self):
        """ Returns the memory required by all arrays in bytes."""
        return np.sum([hcu.array_bytes(a) for a
            in self.arrays(reify=True).itervalues()])

    def mem_required(self):
        """ Return a string representation of the total memory required """
        return hcu.fmt_bytes(self.bytes_required())

    def register_dimension(self, name, dim_data, **kwargs):
        """
        Registers a dimension with this Solver object

        Arguments
        ---------
            dim_data : integer or Dimension
                if an integer, this will be used to
                define the global_size of the dimension
                and possibly other attributes if they are
                not present in kwargs.
                If a Dimension, it will be updated with
                any appropriate keyword arguments

        Keyword Arguments
        -----------------
            description : string
                The description for this dimension.
                e.g. 'Number of timesteps'.
            local_size : integer or None
                The local size of this dimension
                on this solver. If None, set to
                the global_size.
            lower_extent : integer or None
                The lower extent of this dimension
                within the global space
            upper_extent : integer or None
                The upper extent of this dimension
                within the global space
            zero_valid : boolean
                If True, this dimension may be zero-sized.

        Returns
        -------
        A Dimension object
        """

        if name in self._dims:
            raise AttributeError((
                "Attempted to register dimension '{n}'' "
                "as an attribute of the solver, but "
                "it already exists. Please choose "
                "a different name!").format(n=name))

        # Create the dimension dictionary
        D = self._dims[name] = create_dimension(name,
            dim_data, **kwargs)

        return D

    def register_dimensions(self, dims):
        """
        >>> slvr.register_dimensions([
            {'name' : 'ntime', 'local_size' : 10, 'extents' : [2, 7], 'safety': False },
            {'name' : 'na', 'local_size' : 3, 'extents' : [2, 7]},
            ])
        """

        if isinstance(dims, collections.Mapping):
            dims = dims.itervalues()

        for dim in dims:
            self.register_dimension(dim.name, dim)

    def update_dimensions(self, dims):
        """
        >>> slvr.update_dimensions([
            {'name' : 'ntime', 'local_size' : 10, 'extents' : [2, 7], 'safety': False },
            {'name' : 'na', 'local_size' : 3, 'extents' : [2, 7]},
            ])
        """

        if isinstance(dims, collections.Mapping):
            dims = dims.itervalues()

        for dim in dims:
            # Defer to update dimension for dictionaries
            if isinstance(dim, dict):
                self.update_dimension(**dim)
            # Replace if given a Dimension object
            elif isinstance(dim, Dimension):
                self._dims[dim.name] = dim
            else:
                raise TypeError("Unhandled type '{t}'"
                    "in update_dimensions".format(t=type(dim)))

    def update_dimension(self, name, **update_dict):
        """
        Update the dimension size and extents.

        Arguments
        ---------
            update_dict : dict
        """
        if not name:
            raise AttributeError("A dimension name is required to update "
                "a dimension. Update dictionary {u}."
                    .format(u=update_dict))

        dim = self._dims.get(name, None)

        # Sanity check dimension existence
        if not dim:
            raise ValueError("'Dimension {n}' cannot be updated as it "
                "is not registered in the dimension dictionary."
                    .format(n=name))

        dim.update(**update_dict)

    def _dim_attribute(self, attr, *args):
        """
        Returns a list of dimension attribute attr, for the
        dimensions specified as strings in args.

        ntime, nbl, nchan = slvr._dim_attribute('global_size', ntime, 'nbl', 'nchan')

        or

        ntime, nbl, nchan, nsrc = slvr._dim_attribute('global_size', 'ntime,nbl:nchan nsrc')
        """

        import re

        # If we got a single string argument, try splitting it by separators
        if len(args) == 1 and isinstance(args[0], str):
            args = (s.strip() for s in re.split(',|:|;| ', args[0]))

        # Now get the specific attribute for each argument, parsing
        # any string expressions on the way
        result = [getattr(self._dims[name], attr) for name in args]

        # Return single element if length one else entire list
        return result[0] if len(result) == 1 else result

    def dim_global_size_dict(self):
        """ Returns a mapping of dimension name to global size """
        return { d.name: d.global_size for d in self._dims.itervalues()}

    def dim_local_size_dict(self):
        """ Returns a mapping of dimension name to local size """
        return { d.name: d.local_size for d in self._dims.itervalues()}

    def dim_lower_extent_dict(self):
        """ Returns a mapping of dimension name to lower_extent """
        return { d.name: d.lower_extent for d in self._dims.itervalues()}

    def dim_upper_extent_dict(self):
        """ Returns a mapping of dimension name to upper_extent """
        return { d.name: d.upper_extent for d in self._dims.itervalues()}

    def dim_global_size(self, *args):
        """
        ntime, nbl, nchan = slvr.dim_global_size('ntime, 'nbl', 'nchan')

        or

        ntime, nbl, nchan, nsrc = slvr.dim_global_size('ntime,nbl:nchan nsrc')
        """

        return self._dim_attribute('global_size', *args)

    def dim_local_size(self, *args):
        """
        ntime, nbl, nchan = slvr.dim_local_size('ntime, 'nbl', 'nchan')

        or

        ntime, nbl, nchan, nsrc = slvr.dim_local_size('ntime,nbl:nchan nsrc')
        """

        return self._dim_attribute('local_size', *args)

    def dim_lower_extent(self, *args):
        """
        t_ex, bl_ex, ch_ex = slvr.dim_lower_extent('ntime, 'nbl', 'nchan')

        or

        t_ex, bl_ex, ch_ex, src_ex = slvr.dim_lower_extent('ntime,nbl:nchan nsrc')
        """

        return self._dim_attribute('lower_extent', *args)

    def dim_upper_extent(self, *args):
        """
        t_ex, bl_ex, ch_ex = slvr.dim_upper_extent('ntime, 'nbl', 'nchan')

        or

        t_ex, bl_ex, ch_ex, src_ex = slvr.dim_upper_extent('ntime,nbl:nchan nsrc')
        """

        return self._dim_attribute('upper_extent', *args)

    def dim_extents(self, *args):
        l = self.dim_lower_extent(*args)
        u = self.dim_upper_extent(*args)

        # Handle sequence and singletons differently
        if isinstance(l, collections.Sequence):
            return zip(l, u)
        else:
            return (l, u)

    def dim_extent_size(self, *args):
        extents = self.dim_extents(*args)

        # Handle tuples and sequences differently
        if isinstance(extents, tuple):
            return extents[1] - extents[0]
        else: # isinstance(extents, collections.Sequence):
            return (u-l for l, u in extents)

    def register_array(self, name, shape, dtype, **kwargs):
        """
        Register an array with this Solver object.

        Arguments
        ----------
            name : string
                name of the array.
            shape : integer/string or tuple of integers/strings
                Shape of the array.
            dtype : data-type
                The data-type for the array.

        Returns
        -------
            A dictionary describing this array.
        """

        # Complain if array exists
        if name in self._arrays:
            raise ValueError(('Array %s is already registered '
                'on this solver object.') % name)

        # OK, create a record for this array
        A = self._arrays[name] = AttrDict(name=name,
            dtype=dtype, shape=shape,
            **kwargs)

        return A

    def register_arrays(self, arrays):
        """
        Register arrays using a list of dictionaries defining the arrays.

        The list should itself contain dictionaries. i.e.

        >>> D = [
            { 'name':'uvw', 'shape':(3,'ntime','nbl'),'dtype':np.float32 },
            { 'name':'lm', 'shape':(2,'nsrc'),'dtype':np.float32 }
        ]
        """

        if isinstance(arrays, collections.Mapping):
            arrays = arrays.itervalues()

        for ary in arrays:
            self.register_array(**ary)

    def register_property(self, name, dtype, default, **kwargs):
        """
        Registers a property with this Solver object

        Arguments
        ----------
            name : string
                The name of the property.
            dtype : data-type
                The data-type of this property
            default :
                Default value for the property.

        Keyword Arguments
        -----------------
            setter : boolean or function
                if True, a default 'set_name' member is created, otherwise not.
                If a method, this is used instead.
            setter_docstring : string
                docstring for the default setter.
        """
        if name in self._properties:
            raise ValueError(('Property %s is already registered '
                'on this solver object.') % name)

        P = self._properties[name] = AttrDict(name=name,
            dtype=dtype, default=default)

        #if not hasattr(HyperCube, name):
        if not HyperCube.__dict__.has_key(name):
                # Create the descriptor for this property on the class instance
            setattr(HyperCube, name, PropertyDescriptor(record_key=name, default=default))

        # Set the descriptor on this object instance
        setattr(self, name, default)

        # Should we create a setter for this property?
        setter = kwargs.get('setter_method', True)
        setter_name = hcu.setter_name(name)

        # Yes, create a default setter
        if isinstance(setter, types.BooleanType) and setter is True:
            def set(self, value):
                setattr(self,name,value)

            setter_method = types.MethodType(set, self)
            setattr(self, setter_name, setter_method)

            # Set up the docstring, using the supplied one
            # if it is present, otherwise generating a default
            setter_docstring = kwargs.get('setter_docstring', None)
            getattr(setter_method, '__func__').__doc__ = \
                """ Sets property %s to value. """ % (name) \
                if setter_docstring is None else setter_docstring

        elif isinstance(setter, types.MethodType):
            setattr(self, setter_name, setter)
        else:
            raise TypeError, ('setter keyword argument set',
                ' to an invalid type %s' % (type(setter)))

        return P

    def register_properties(self, properties):
        """
        Register properties using a list defining the properties.

        The dictionary should itself contain dictionaries. i.e.

        >>> D = [
            { 'name':'ref_wave','dtype':np.float32,
                'default':1.41e6, 'registrant':'solver' },
        ]
        """
        if isinstance(properties, collections.Mapping):
            properties = properties.itervalues()

        for prop in properties:
            self.register_property(**prop)

    def properties(self):
        """ Returns a dictionary of properties """
        return self._properties

    def property(self, name):
        """ Returns a property """
        try:
            return self._properties[name]
        except KeyError:
            raise KeyError("Property '{n}' is not registered "
                "on this solver".format(n=name)), None, sys.exc_info()[2]

    def arrays(self, reify=False):
        """
        Returns a dictionary of arrays. If reify is True,
        it will replace any dimension within the array shape with
        the local_size of the dimension.
        """
        return (self._arrays if not reify else
            hcu.reify_arrays(self._arrays, self.dimensions(copy=False)))

    def array(self, name, reify=False):
        """
        Returns an array object. If reify is True,
        it will replace any dimension within the array shape with
        the local_size of the dimension.
        """

        try:
            return (self._arrays[name] if not reify else
                hcu.reify_arrays({name : self._arrays[name]},
                    self.dimensions(copy=False))[name])
        except KeyError:
            raise KeyError("Array '{n}' is not registered on this solver"
                .format(n=name)), None, sys.exc_info()[2]

    def dimensions(self, copy=True):
        """
        Return a dictionary of dimensions

        Keyword Arguments
        -----------------
            copy : boolean
                Returns a copy if True.
        """

        return self._dims.copy() if copy else self._dims

    def dimension(self, name, copy=True):
        """
        Returns the specified dimension object.
        """

        try:
            return create_dimension(name, self._dims[name]) if copy else self._dims[name]
        except KeyError:
            raise KeyError("Dimension '{n}' is not registered "
                "on this solver".format(n=name)), None, sys.exc_info()[2]

    def copy(self):
        """ Return a copy of the hypercube """
        return HyperCube(dimensions=self.dimensions(copy=False),
            arrays=self.arrays(), properties=self.properties())

    def gen_dimension_table(self):
        """
        2D array describing each registered dimension
        together with headers - for use in __str__
        """
        headers = ['Dimension Name', 'Description',
            'Global Size', 'Local Size', 'Extents']

        table = []
        for dimval in sorted(self.dimensions(copy=False).itervalues(),
                             key=lambda dval: dval.name.upper()):

            table.append([dimval.name,
                dimval.description,
                dimval.global_size,
                dimval.local_size,
                (dimval.lower_extent, dimval.upper_extent)])

        return table, headers

    def gen_array_table(self):
        """
        2D array describing each registered array
        together with headers - for use in __str__
        """
        headers = ['Array Name', 'Size', 'Type', 'Shape']

        # Reify arrays to work out their actual size
        reified_arrays = self.arrays(reify=True)

        table = []
        for array in sorted(self.arrays().itervalues(),
                             key=lambda aval: aval.name.upper()):
            # Get the actual size of the array
            nbytes = hcu.array_bytes(reified_arrays[array.name])
            # Print shape tuples without spaces and single quotes
            sshape = '(%s)' % (','.join(map(str, array.shape)),)
            table.append([array.name,
                hcu.fmt_bytes(nbytes),
                np.dtype(array.dtype).name,
                sshape])

        return table, headers

    def gen_property_table(self):
        """
        2D array describing each registered property
        together with headers - for use in __str__
        """
        headers = ['Property Name', 'Type', 'Value', 'Default Value']

        table = []
        for propval in sorted(self._properties.itervalues(),
                              key=lambda pval: pval.name.upper()):
            table.append([propval.name,
                np.dtype(propval.dtype).name,
                getattr(self, propval.name),
                propval.default])

        return table, headers

    def __str__(self):
        """ Outputs a string representation of this object """

        result = []

        if len(self._dims) > 0:
            table, headers = self.gen_dimension_table()
            result.append("Registered Dimensions:\n%s\n\n" % (
                tabulate(table, headers=headers),))

        if len(self._arrays) > 0:
            table, headers = self.gen_array_table()
            table.append(['Local Memory Usage', self.mem_required(), '', ''])
            result.append("Registered Arrays:\n%s\n\n" % (
                tabulate(table, headers=headers),))

        if len(self._properties) > 0:
            table, headers = self.gen_property_table()
            result.append("Registered Properties:\n%s\n\n" % (tabulate(
                table, headers=headers),))

        return ''.join(result)

    def endpoint_iter(self, *dim_strides, **kwargs):
        """
        Recursively iterate over the (dimension, stride)
        tuples specified in dim_strides, returning the start
        and end indices for each chunk.

        For example, the following call effectively produces
        2 loops over the 'ntime' and 'nchan' dimensions
        in chunks of 10 and 4 respectively.

        >>> for (ts, te), (cs, ce) in cube.endpoint_iter(('ntime', 10), ('nchan', 4))
        >>>     print 'Time range [{ts},{te}] Channel Range [{cs},{ce}]'.format(
        >>>         ts=ts, te=te, cs=cs, ce=ce)

        Arguments:
            dim_strides: list
                list of (dimension, stride) tuples

        Keyword Arguments:
            scope: string
                Governs whether iteration occurs over the global or
                local dimension space. Defaults to 'global_size', but
                can also be 'local_size'.

        Returns:
            An iterator
        """

        def _dim_endpoints(size, stride):
            r = xrange(0, size, stride) if stride > 0 else xrange(0, size)
            return ((i, min(i+stride, size)) for i in r)

        dims = self.dimensions(copy=False)
        scope = kwargs.get('scope', GLOBAL_SIZE)
        gens = (_dim_endpoints(getattr(dims[d], scope), s) for d, s in dim_strides)
        return itertools.product(*gens)

    def slice_iter(self, *dim_strides, **kwargs):
        """
        Recursively iterate over the (dimension, stride)
        tuples specified in dim_strides, returning the chunk
        start offsets for each specified dimensions.

        For example, the following call effectively produces
        2 loops over the 'ntime' and 'nchan' dimensions
        in chunks of 10 and 4 respectively.

        >>> A = np.ones(size=(100, 4))
        >>> for ts, cs in cube.endpoint_iter(('ntime', 10), ('nchan', 4))
        >>>     A[ts, cs].sum()
        >>>
        >>> for i cube.endpoint_iter(('ntime', 10), ('nchan', 4))
        >>>     A[i].sum()

        Arguments:
            dim_strides: list
                list of (dimension, stride) tuples

        Keyword Arguments:
            scope: string
                Governs whether iteration occurs over the global or
                local dimension space. Defaults to 'global_size', but
                can also be 'local_size'.

        Returns:
            An iterator
        """
        def _create_slices(*args):
            return tuple(slice(s,e,1) for (s, e) in args)

        return (_create_slices(*s) for s in self.endpoint_iter(
            *dim_strides, **kwargs))

    def dim_iter(self, *dim_strides, **kwargs):
        """
        Recursively iterate over the (dimension, stride)
        tuples specified in dim_strides, returning a tuple
        of dictionaries describing a dimension update.

        For example, the following call effectively produces
        2 loops over the 'ntime' and 'nchan' dimensions
        in chunks of 10 and 4 respectively.

        >>> for d in cube.dim_iter(('ntime', 10), ('nchan', 4))
        >>>     cube.update_dimensions(d)

        Arguments:
            dim_strides: list
                list of (dimension, stride) tuples

        Keyword Arguments:
            scope: string
                Governs whether iteration occurs over the global or
                local dimension space. Defaults to 'global_size', but
                can also be 'local_size'.
            update_local_size : boolean
                If True and scope is 'global_size', the returned dictionaries
                will contain a 'local_size' key, set to the difference of
                the lower and upper extents

        Returns:
            An iterator
        """

        # Extract dimension names
        dims = [ds[0] for ds in dim_strides]

        def _create_dim_dicts(*args):
            return tuple({ 'name': d,
                'lower_extent': s, 'upper_extent': e }
                for (d, (s, e)) in args)

        def _create_dim_dicts_with_local(*args):
            return tuple({ 'name': d, 'local_size' : e - s,
                'lower_extent': s, 'upper_extent': e }
                for (d, (s, e)) in args)

        is_scope_global = kwargs.get('scope', GLOBAL_SIZE) == GLOBAL_SIZE
        local_update_requested = kwargs.get('update_local_size', False)

        if is_scope_global and local_update_requested:
            f = _create_dim_dicts_with_local
        else:
            f = _create_dim_dicts

        # Return a tuple-dict-creating generator
        return (f(*zip(dims, s)) for s
            in self.endpoint_iter(*dim_strides, **kwargs))

    def cube_iter(self, *dim_strides, **kwargs):
        """
        Recursively iterate over the (dimension, stride)
        tuples specified in dim_strides, returning cloned hypercubes
        with each of the specified dimensions modified accordingly.

        For example, the following call effectively produces
        2 loops over the 'ntime' and 'nchan' dimensions
        in chunks of 10 and 4 respectively.

        >>> A = np.ones(size=(100, 4))
        >>> for c in cube.cube_iter(('ntime', 10), ('nchan', 4))
        >>>     assert c.dim_local_size('ntime', 'nchan') == (10, 4)

        Arguments:
            dim_strides: list
                list of (dimension, stride) tuples

        Keyword Arguments:
            scope: string
                Governs whether iteration occurs over the global or
                local dimension space. Defaults to 'global_size', but
                can also be 'local_size'.

        Returns:
            An iterator

        """
        def _make_cube(dims, arrays, *args):
            """
            Create a hypercube given reified dimensions and a list of
            (dim_name, dim_slice) tuples
            """

            # Create new hypercube, registering everything in rdims
            cube = HyperCube()
            cube.register_dimensions(dims)
            cube.register_arrays(arrays)

            # Now update dimensions given slice information
            for (d, (s, e)) in args:
                cube.update_dimension(name=d,
                    local_size=dims[d].local_size,
                    global_size=dims[d].global_size,
                    lower_extent=s, upper_extent=e)

            return cube

        # Extract dimension names
        dim_names = [ds[0] for ds in dim_strides]
        arrays = (hcu.reify_arrays(self.arrays(), self.dimensions(copy=False))
            if kwargs.get('arrays', False) else {})

        # Return a cube-creating generator
        return (_make_cube(self.dimensions(), arrays, *zip(dim_names, s))
            for s in self.endpoint_iter(*dim_strides, **kwargs))

    def slice_index(self, *slice_dims, **kwargs):
        """
        Returns a tuple of slices, each slice equal to the
        slice(lower_extent, upper_extent, 1) of the dimensions
        supplied in slice_dims. If the dimension is integer d,
        slice(0, d, 1) will be used instead of the lower and upper extents

        e.g.
        >>> A = np.ones(ntime, na)
        >>> idx = cube.slice_index('ntime','na', 3)
        >>> A[idx].sum()
        >>> ntime, na, components = cube.slice_index('ntime', 'na', 3)
        >>> A[ntime, na, components].sum()


        Arguments:
            dims: list
                list of dimensions which should have slice
                objects returned.

        Returns:
            A tuple containing slices for each dimension in dims
        """
        dims = self.dimensions(copy=False)

        return tuple(slice(dims[d].lower_extent, dims[d].upper_extent, 1)
            if d in dims else slice(0, d, 1)
            for d in slice_dims)