Introduction
============

hypercube is a package for reasoning about Radio Interferometry problem sizes.

Radio Interferometry data, such as visibilities, is both large and dense.
Computation on this data is therefore generally suited to distributed,
data parallel computation.
To achieve this parallelism it is necessary to subdivide this data into tiles,
large enough to fit within the memory budgets of individual nodes and GPUs.

hypercube is an simple abstraction that uses labelled Dimensions
and abstract numpy_ array shapes, defined in terms of these Dimensions,
to:

    * Reason about memory budgets
    * Choose array subdivision strategies
    * Iterate over a problem space

::

    from hypercube import HyperCube

    cube = HyperCube()
    cube.register_dimension("ntime", 10000,
        description="Timesteps")
    cube.register_dimension("nbl", 2016,
        description="Baselines")
    cube.register_dimension("nchan", 16384,
        description="Channels")

    cube.register_array("uvw", ("ntime", "nbl", 3),
        np.float64)
    cube.register_array("frequency", ("nchan",),
        np.float64)
    cube.register_array("model_vis", ("ntime", "nbl", "nchan", 4),
        np.complex128)

::

    print cube

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


.. _numpy: http://www.numpy.org
