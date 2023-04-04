## Introduction

This is a partial implementation of the [ASAM Open Drive](https://www.asam.net/standards/detail/opendrive/) specification, it is purely done for educational purposes, it is not designed to be fast, complete and/or efficient. There are some example files included as well as a Jupyter Notebook to visualize the resulting mesh.

## Spec Coverage

The coverage of the spec is far from complete, but the basic features are implemented and it can be further extended for complete coverage. 

Notable short commings:
- [ ] For reference line geometry, only line/spiral/arc geometry is implemented, not the (legacy?) poly geometry
- [ ] Road height and superelevation is implemented, but there is no support for road shapes
- [ ] Lane width implemented using lane width only, border implementation for mesh generation not supported (not hard to do)
- [ ] Basic road marker support, the more detailed implementation is not supported
- [ ] No support for signals, objects, controllers, etc.

## Mesh Generation

The mesh generation process uses a fixed step and samples each lane independently. This is necessary, since each lane can have individual height offsets, lane type and materials. A basic color coding is used to distinguish the lane types. The lane markers are also converted to a mesh individually and layed slightly above the road surface.

Support to convert to convert to an `.obj` or some other 3d modelling format is not yet implemented.