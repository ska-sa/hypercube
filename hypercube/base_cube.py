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

import numpy as np
import types

from weakref import WeakKeyDictionary
from attrdict import AttrDict
from collections import OrderedDict

from hypercube.dims import DimData, create_dim_data
import hypercube.util as hcu

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
        self._dims = OrderedDict()
        self._arrays = OrderedDict()
        self._properties = OrderedDict()

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
            dim_data : integer or dict


        Keyword Arguments
        -----------------
            description : string
                The description for this dimension.
                e.g. 'Number of timesteps'.
            global_size : integer
                The global size of this dimension across
                all solvers.
            local_size : integer or None
                The local size of this dimension
                on this solver. If None, set to
                the global_size.
            extents : list or tuple of length 2
                The extent of the dimension on the solver.
                E[0] < E[1] <= local_size must hold.
            zero_valid : boolean
                If True, this dimension may be zero-sized.

        Returns
        -------
        A dictionary describing this dimension
        """

        if name in self._dims:
            raise AttributeError((
                "Attempted to register dimension '{n}'' "
                "as an attribute of the solver, but "
                "it already exists. Please choose "
                "a different name!").format(n=name))

        # Create the dimension dictionary
        D = self._dims[name] = create_dim_data(name, dim_data, **kwargs)

        return D

    def register_dimensions(self, dim_list):
        """
        >>> slvr.register_dimensions([
            {'name' : 'ntime', 'local_size' : 10, 'extents' : [2, 7], 'safety': False },
            {'name' : 'na', 'local_size' : 3, 'extents' : [2, 7]},
            ])
        """

        for dim in dim_list:
            self.register_dimension(dim.name, dim)

    def update_dimensions(self, dim_list):
        """
        >>> slvr.update_dimensions([
            {'name' : 'ntime', 'local_size' : 10, 'extents' : [2, 7], 'safety': False },
            {'name' : 'na', 'local_size' : 3, 'extents' : [2, 7]},
            ])
        """
        for dim_data in dim_list:
            self.update_dimension(**dim_data)


    def update_dimension(self, **update_dict):
        """
        Update the dimension size and extents.

        Arguments
        ---------
            update_dict : dict
        """
        name = update_dict.get(DimData.NAME, None)

        if not name:
            raise AttributeError("A dimension name is required to update "
                "a dimension. Update dictionary {u}."
                    .format(u=update_dict))

        dim = self._dims.get(name, None)

        # Sanity check dimension existence
        if not dim:
            montblanc.log.warn("'Dimension {n}' cannot be updated as it "
                "is not registered in the dimension dictionary."
                    .format(n=name))

            return

        dim.update(update_dict)

    def __dim_attribute(self, attr, *args):
        """
        Returns a list of dimension attribute attr, for the
        dimensions specified as strings in args.

        ntime, nbl, nchan = slvr.__dim_attribute('global_size', ntime, 'nbl', 'nchan')
        
        or

        ntime, nbl, nchan, nsrc = slvr.__dim_attribute('global_size', 'ntime,nbl:nchan nsrc')
        """

        import re
        from expressions import parse_expression as pe

        # If we got a single string argument, try splitting it by separators
        if len(args) == 1 and isinstance(args[0], str):
            args = (s.strip() for s in re.split(',|:|;| ', args[0]))

        # Create a variable template from this attribute
        # for use in parse_expressions below
        T = { d.name: d[attr] for d in self._dims.itervalues() }

        # Now get the specific attribute for each argument, parsing
        # any string expressions on the way
        result = [pe(self._dims[name][attr], variables=T, expand=True)
            for name in args]

        # Return single element if length one else entire list
        return result[0] if len(result) == 1 else result

    def dim_global_size(self, *args):
        """
        ntime, nbl, nchan = slvr.dim_global_size('ntime, 'nbl', 'nchan')
        
        or

        ntime, nbl, nchan, nsrc = slvr.dim_global_size('ntime,nbl:nchan nsrc')
        """

        return self.__dim_attribute(DimData.GLOBAL_SIZE, *args)

    def dim_local_size(self, *args):
        """
        ntime, nbl, nchan = slvr.dim_local_size('ntime, 'nbl', 'nchan')
        
        or

        ntime, nbl, nchan, nsrc = slvr.dim_local_size('ntime,nbl:nchan nsrc')
        """

        return self.__dim_attribute(DimData.LOCAL_SIZE, *args)

    def dim_global_size_dict(self):
        """ Returns a mapping of dimension name to global size """
        from expressions import expand_expression_map

        return expand_expression_map({ d.name: d.global_size
            for d in self._dims.itervalues()})

    def dim_local_size_dict(self):
        """ Returns a mapping of dimension name to local size """
        from expressions import expand_expression_map

        return expand_expression_map({ d.name: d.local_size
            for d in self._dims.itervalues()})

    def dim_extents(self, *args):
        """
        t_ex, bl_ex, ch_ex = slvr.dim_extents('ntime, 'nbl', 'nchan')
        
        or

        t_ex, bl_ex, ch_ex, src_ex = slvr.dim_extents('ntime,nbl:nchan nsrc')
        """

        return self.__dim_attribute(DimData.EXTENTS, *args)

    def dim_extents_dict(self):
        """ Returns a mapping of dimension name to extents """
        return { d.name: d.extents for d in self.__dims.itervalues() }

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

        Keyword Arguments
        -----------------
            shape_member : boolean
                True if a member called 'name_shape' should be
                created on the Solver object.
            dtype_member : boolean
                True if a member called 'name_dtype' should be
                created on the Solver object.

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

    def register_arrays(self, array_list):
        """
        Register arrays using a list of dictionaries defining the arrays.

        The list should itself contain dictionaries. i.e.

        >>> D = [
            { 'name':'uvw', 'shape':(3,'ntime','nbl'),'dtype':np.float32 },
            { 'name':'lm', 'shape':(2,'nsrc'),'dtype':np.float32 }
        ]
        """
        for ary in array_list:
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

    def register_properties(self, property_list):
        """
        Register properties using a list defining the properties.

        The dictionary should itself contain dictionaries. i.e.

        >>> D = [
            { 'name':'ref_wave','dtype':np.float32,
                'default':1.41e6, 'registrant':'solver' },
        ]
        """
        for prop in property_list:
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
                "on this solver".format(n=name))

    def arrays(self, reify=False):
        """
        Returns a dictionary of arrays. If reify is True,
        it will replace any dimension within the array shape with
        the local_size of the dimension.
        """
        return (hcu.reify_arrays(self._arrays, self.dimensions(reify=True))
            if reify else self._arrays)

    def array(self, name, reify=False):
        """
        Returns an array object. If reify is True,
        it will replace any dimension within the array shape with
        the local_size of the dimension.

        Reifying arrays "individually" is expensive since, in practice,
        all dimensions must be reified to handle dependent expressions.
        """

        # Complain if the array doesn't exist
        if name not in self._arrays:
            raise KeyError("Array '{n}' is not registered on this solver"
                .format(n=name))

        # Just return the array if we're not reifying
        if not reify:
            return self._arrays[name]

        return hcu.reify_arrays({name : self._arrays[name]},
            self.dimensions(reify=True))[name]

    def dimensions(self, reify=False):
        """
        Return a dictionary of dimensions

        Keyword Arguments
        -----------------
            reify : boolean
                if True, converts any expressions in the dimension
                information to integers.

        """


        return hcu.reify_dims(self._dims) if reify else self._dims

    def dimension(self, name, reify=False):
        """
        Returns a dimension object.

        Reifying dimensions "individually" is expensive since, in practice,
        all dimensions must be reified to handle dependent expressions.
        """
        # Complain if the array doesn't exist
        if name not in self._dims:
            raise KeyError("Dimension '{n}' is not registered on this solver"
                .format(n=name))

        # Just return the array if we're not reifying
        if not reify:
            return self._dims[name]

        # Reifies everything just to get this dimension, expensive
        return hcu.reify_dims(self._dims)[name]

    def fmt_dimension_line(self, name, description, global_size, local_size, extents):
        return '%-*s%-*s%-*s%-*s%-*s' % (
            15,name,
            20,description,
            12,global_size,
            12,local_size,
            15,extents)


    def fmt_array_line(self, name, size, dtype, shape):
        """ Format array parameters on an 80 character width line """
        return '%-*s%-*s%-*s%-*s' % (
            20,name,
            10,size,
            15,dtype,
            35,shape)

    def fmt_property_line(self, name, dtype, value, default):
        return '%-*s%-*s%-*s%-*s' % (
            20,name,
            10,dtype,
            20,value,
            20,default)

    def gen_dimension_descriptions(self):
        """ Generator generating string describing each registered dimension """
        yield 'Registered Dimensions'
        yield '-'*80
        yield self.fmt_dimension_line('Dimension Name', 'Description', 'Global Size',
            'Local Size', 'Extents')
        yield '-'*80

        for d in sorted(self.dimensions(reify=True).itervalues(),
            key=lambda x: x.name.upper()):

            yield self.fmt_dimension_line(
                d.name, d.description, d.global_size, d.local_size, d.extents)

    def gen_array_descriptions(self):
        """ Generator generating strings describing each registered array """
        yield 'Registered Arrays'
        yield '-'*80
        yield self.fmt_array_line('Array Name','Size','Type','Shape')
        yield '-'*80

        # Reify arrays to work out their actual size
        reified_arrays = self.arrays(reify=True)

        for a in sorted(self.arrays().itervalues(),
            key=lambda x: x.name.upper()):

            # Get the actual size of the array
            nbytes = hcu.array_bytes(reified_arrays[a.name])
            # Print shape tuples without spaces and single quotes
            sshape = '({s})'.format(s=','.join(
                [str(v) if not isinstance(v, str) else v
                for v in a.shape]))

            yield self.fmt_array_line(a.name,
                hcu.fmt_bytes(nbytes),
                np.dtype(a.dtype).name,
                sshape)

    def gen_property_descriptions(self):
        """ Generator generating string describing each registered property """
        yield 'Registered Properties'
        yield '-'*80
        yield self.fmt_property_line('Property Name',
            'Type', 'Value', 'Default Value')
        yield '-'*80

        for p in sorted(self._properties.itervalues(), key=lambda x: x.name.upper()):
            yield self.fmt_property_line(
                p.name, np.dtype(p.dtype).name,
                getattr(self, p.name), p.default)

    def __str__(self):
        """ Outputs a string representation of this object """

        l = []

        if len(self._dims) > 0:
            l.extend([s for s in self.gen_dimension_descriptions()])
            l.append('')

        if len(self._arrays) > 0:
            l.extend([s for s in self.gen_array_descriptions()])
            l.append('-'*80)
            l.append('%-*s: %s' % (18,'Local Memory Usage', self.mem_required()))
            l.append('-'*80)
            l.append('')

        if len(self._properties) > 0:
            l.extend([s for s in self.gen_property_descriptions()])

        return '\n'.join(l)