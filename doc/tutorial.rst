Tutorial
========

Dimensions
----------

The use of python_ and numpy_ in the Radio Astronomy community naturally
results in representation of data as multi-dimensional numpy_ arrays.
hypercube, similarly to xray_ and pandas_ utilises the concept of
labelling Dimensions. For example, a hypercube can be created and
**Time**, **Baseline**, and **Channel** dimensions
can be registered with various global sizes.


::

    from hypercube import HyperCube

    cube = HyperCube()
    cube.register_dimension("ntime", 10000,
        description="Timesteps")
    cube.register_dimension("nbl", 2016,
        description="Baselines")
    cube.register_dimension("nchan", 16384,
        description="Channels")

Printing the cube then yields information about the registered Dimensions.
Note that the **Global Size** matches the **Extents**.

::

    print cube

    Registered Dimensions:
    Dimension Name    Description      Global Size  Extents
    ----------------  -------------  -------------  ----------
    nbl               Baselines               2016  (0, 2016)
    nchan             Channels               16384  (0, 16384)
    ntime             Timesteps              10000  (0, 10000)


Arrays
------

Then we can register an abstract definition, or schema, of the
**Model Visibility** array on the hypercube defined using the
names of the previously registered dimensions.

::

    cube.register_array("uvw", ("ntime", "nbl", 3), np.float64)
    cube.register_array("frequency", ("nchan",), np.float64)
    cube.register_array("model_vis", ("ntime", "nbl", "nchan", 4),
        np.complex128)


Printing the cube now displays additional information about the
arrays and their sizes in terms of the dimension **extents**.

::

    Registered Dimensions:
    Dimension Name    Description      Global Size  Extents
    ----------------  -------------  -------------  ----------
    nbl               Baselines               2016  (0, 2016)
    nchan             Channels               16384  (0, 16384)
    ntime             Timesteps              10000  (0, 10000)

    Registered Arrays:
    Array Name          Size     Type        Shape
    ------------------  -------  ----------  -------------------
    frequency           128.0KB  float64     (nchan)
    model_vis           19.2TB   complex128  (ntime,nbl,nchan,4)
    uvw                 461.4MB  float64     (ntime,nbl,3)
    Local Memory Usage  19.2TB

Modifying Dimension Extents
---------------------------

The problem in the previous section is too large (19.2TB) to fit
within a single compute node's memory, so it is necessary to
subdivide or tile the problem.
The **extents** of the **Time** and **Channel** dimension
are modified as follows:

::

    cube.update_dimension("ntime", lower_extent=0, upper_extent=100)
    cube.update_dimension("nchan", lower_extent=0, upper_extent=64)

    print cube

    Registered Dimensions:
    Dimension Name    Description      Global Size  Extents
    ----------------  -------------  -------------  ---------
    nbl               Baselines               2016  (0, 2016)
    nchan             Channels               16384  (0, 64)
    ntime             Timesteps              10000  (0, 100)

    Registered Arrays:
    Array Name          Size     Type        Shape
    ------------------  -------  ----------  -------------------
    frequency           512.0B   float64     (nchan)
    model_vis           787.5MB  complex128  (ntime,nbl,nchan,4)
    uvw                 4.6MB    float64     (ntime,nbl,3)
    Local Memory Usage  792.1MB

Note how the dimension extents of the **Time** and **Channel** dimensions
have changed. The problem now fits within a reasonable memory budget of
792.1MB.

Querying Dimension Extents
--------------------------

The dimension extents can be queried on the cube:

::

    cube.dim_lower_extent("ntime,nbl,nchan")
    [0, 0, 0]

    cube.dim_upper_extent("ntime,nbl,nchan")
    [100, 2016, 64]

    cube.dim_extent_size("ntime,nbl,nchan")
    [100, 2016, 64]

    cube.dim_extents("ntime,nbl,nchan")
    [(0, 100), (0, 2016), (0, 64)]

Iterating over Cubes
--------------------

The cube supports iteration over tiles defined by dimensions.
The :meth:`hypercube.base_cube.HyperCube.extent_iter` method
produces tuples of lower extents for each dimension provided to it.
Here, it produces extents for tiles of 100 Timesteps and
64 Channels.

::

    for (lt, ut), (lc, uc) in cube.extent_iter(("ntime", 100), ("nchan", 64)):
        print ("lower time {} upper time {} "
                "lower channel {} upper channel{}".format(
                    lt, ut, lc, uc)

    lower time 0 upper time 100 lower channel 0 upper channel 64
    lower time 0 upper time 100 lower channel 64 upper channel 128
    lower time 0 upper time 100 lower channel 128 upper channel 192
    lower time 0 upper time 100 lower channel 192 upper channel 256
    lower time 0 upper time 100 lower channel 256 upper channel 320

Other methods of iteration include producing dictionaries defining dimension updates

::

    for d in cube.dim_iter(("ntime", 100), ("nchan", 64)):
        print d
        cube.update_dimensions(d)

    ({'lower_extent': 0, 'upper_extent': 100, 'name': 'ntime'},
     {'lower_extent': 64, 'upper_extent': 128, 'name': 'nchan'})


and producing cubes defining the tile on each iteration

::

    for c in cube.dim_iter(("ntime", 100), ("nchan", 64)):
        ...


.. _python: http://www.python.org
.. _numpy: http://www.numpy.org
.. _xray: http://xarray.pydata.org/en/stable/
.. _pandas: http://pandas.pydata.org/

