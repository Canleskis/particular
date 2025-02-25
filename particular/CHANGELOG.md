# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased - 2024-10-08

### Added

- `Between` struct.
- `SimdInteraction`, `ReduceSimdInteraction`, `BarnesHutInteraction` and `TreeInteraction` traits.
- `SequentialCompute`, `ParallelCompute` and `GpuCompute` traits.
- `sequential::Interactions` and `sequential::Interactions` iterator structs.
- `Acceleration` and `AccelerationSoftened` structs.
- `Position` and `Mass` traits.
- `ToSimd`, `TreeData`, `AccelerationAt`, `AccelerationPaired` and `AccelerationGPU` helper traits for the implementation of `xInteration` traits for `Acceleration` and `AccelerationSoftened` structs.
- `glam`, `ultraviolet` and `nalgebra` features, required to enable implementations using these libraries.

### Changed

- Renamed `ComputeMethod` trait to `Interaction`.
- Built-in algorithms (`BruteForce`, `BarnesHut`, ...) now generic over the interaction to compute which allows them to be more unified. `BruteForceScalar` is now named simply `BruteForce` and can compute over scalar or simd types. `BruteForceSimd` converts scalar types to simd types before computing interactions.
- Renamed `ParticleTree` to `RootedOrthtree`, `ParticleOrdered` to `Ordered`, and `ParticleReordered` to `Reordered`. These storages are also generic over their particles.
- Renamed `PointMass` to `GravitationalField` and renamed field `mass` to `m`. Removed most methods which are now implemented with the helper traits.
- Moved `sequential`, `parallel` and `gpu` modules to the root of the crate.

### Removed

- `Particle` trait, replaced by the `Position` and `Mass` traits.
- `math` module and its traits.
- `accelerations` method on iterators. This method implicitly collected the particles in a `Reordered` storage, which is now required to be explicit.
- `ParticleSystem` struct.
- `BruteForceSoftened` variants. Softening information is now carried by the interaction.

## [0.7.0] - 2024-03-21

### Added

- `ParticleSystem`, `ParticleTree`, `ParticleOrdered` and `ParticleReordered` structs used for storages in built-in compute methods.
- `Array`, `Zero`, `One`, `Infinty`, `FloatOps`, `Float`, `IntoArray`, `FloatVectorOps`, `FloatVector`, `SIMD`, `SIMDElement`, `Reduce`, `FromPrimitive` and `AsPrimitive` traits for math operation abstractions.
- `force_scalar`, `force_simd` methods and various constructors for `PointMass`.
- `ScalarArray` trait to bind a `FloatVector` and an array for `Particle` to `PointMass` conversion.
- Marker `ReorderedCompute` trait.
- `Softened` variants for all compute methods.
- `GpuResources` struct used for `gpu::BruteForce`.
- `compute_shared.wgsl` compute shader.
- `MemoryStrategy` to allow user selection between shared and global memory for massive particles within the compute shader used for `gpu::BruteForce`.

### Changed

- Built-in compute methods implemented with the different provided storage structs.
- `ComputeMethod` no longer generic over `V`.
- Renamed `WgpuData` to `WgpuResources` and `wgpu_data` module to `gpu_compute`.
- `gpu_compute` (internals of `gpu::BruteForce`) is public.
- `gpu::BruteForce` holds a reference to a `wgpu::Device`, `wgpu::Queue` and a mutable reference to a `GpuResources`.
- `write_particle_data` and `read_accelerations` not generic and use `ultraviolet::Vec3`.
- `read_accelerations` is async.
- Update `wgpu` to 0.17
- Renamed `BruteForce` compute methods to `BruteForceScalar` for better differentiation with simd variants.
- `BruteForceSIMD` generic over the lane count.
- `build_node` takes `position` and `compute` functions instead of using traits and computes a square `BoundingBox`. Use `build_node_with` to provide a specific `BoundingBox`.
- `Particle` trait only has the `Array` associated type and `position` expects this array type.
- Deriving `Particle` requires `#[dim]` attribute.
- `accelerations` returns an iterator of arrays.
- Merge `algorithms` and `compute_method` in one module with the name of the latter.
- The size of `SizedOrthant` is no longer generic and is instead a `BoundingBox`
- 3D f32 vectors now fully scalar (unlike `glam::Vec3A`). Use `BruteForceSIMD` compute methods for simd benefits.

### Removed

- `glam` dependency.
- `BruteForcePairsAlt` and `BruteForcePairsCore` compute methods.
- `MassiveAffected`, `MassiveAffectedSIMD` and `ParticleSet` structs.
- `Storage`, `Compute`, `TreeData`, `Positionable` and `BarnesHutTree` traits.
- `acceleration_*` and `total_acceleration_*` methods for `PointMass`.
- `vector` module and its members.

## [0.6.1] - 2023-07-19

### Added

- `acceleration_*` and `total_acceleration_*` methods for `PointMass` to calculate the acceleration between particles of vectors implementing `simd` and `internal`.

### Changed

- Built-in compute methods use `total_acceleration_*` methods when applicable.
- `sequential::BruteForcePairs` is more efficient with less allocations.

## [0.6.0] - 2023-07-14

### Added

- `sequential::BruteForcePairsAlt` compute method.
- `parallel::BruteForceSIMD` and `sequential::BruteForceSIMD` compute methods making use of explicit SIMD instructions for major performance benefits on compatible platforms using [ultraviolet](https://github.com/fu5ha/ultraviolet).
- `Tree`, `Node`, `SizedOrthant`, `BoundingBox` structs and `TreeData`, `Subdivide`, `Positionable`, `BarnesHutTree` traits backing BarnesHut compute methods.
- `PointMass` struct representing a particle for built-in storages.
- `internal::Scalar` and `internal::Vector` traits to help genericity of built-in non-SIMD compute methods.
- `simd::IntoVectorElement`, `simd::SIMD`, `simd::Scalar`, `simd::Vector` and `simd::ReduceAdd` traits to help genericity of built-in SIMD compute methods.
- `MassiveAffected` and `ParticleSet` structs implementing `Storage` backing non-SIMD compute methods.
- `MassiveAffectedSIMD` struct implementing `Storage` backing SIMD compute methods.
- `Storage` trait for inputs of `ComputeMethod::compute`.
- `Compute` trait extending `Iterator` with a `compute` method used by the `Accelerations` trait.
- `algorithms` module.

### Changed

- renamed `sequential::BruteForce` to `sequential::BruteForcePairs`. `sequential::BruteForce` is changed to a naive implementation iterating over all pairs.
- `Compute` trait renamed to `Accelerations`, no longer generic and only returns the computed accelerations. Zip its output with your collection instead.
- `ComputeMethod` generic over a storage and output type.
- `ComputeMethod::compute` expects a storage type.
- `ComputeMethod` has `Output` associated type returned by `ComputeMethod::compute`.
- `Vector` trait renamed to `IntoVectorArray` and its associated type to `Vector` and moved to `internal` submodule.
- `InternalVector` trait renamed to `Vector` and moved with `Scalar` trait to `internal` submodule.
- `compute_method` no longer glob imported in prelude.
- `tree` module and submodules made public.
- `vector` module made public and part of `algorithms`.
- Built-in compute methods moved to `algorithms` module.

### Removed

- `Accelerations` iterator.
- `MapCompute` trait.

## [0.5.2] - 2023-05-16

### Changed

- Use clone instead of copy for tuple particles.

### Fixed

- `NaN` acceleration output when particles are in a certain mass order for `gpu::BruteForce`.

## [0.5.1] - 2023-03-22

### Changed

- `iterators` module renamed to `iterator`.

### Fixed

- Panic when `BarnesHut` tree is empty.

## [0.5.0] - 2023-03-22

### Added

- `BarnesHut` compute method available for 2D and 3D float vectors. Available in sequential and parallel versions (only force computation is parallelized).
- Tuples of a position and a gravitational parameter implement `Particle`.
- `accelerations` method for iterators of particles that returns the particle and its computed acceleration.
- `map_accelerations` method for all iterators that returns the item and its computed acceleration using the mapped value (that implements `Particle`).
- `Accelerations` iterator obtained through the `accelerations` and `map_accelerations` methods available on iterators.

### Changed

- `ComputeMethod::compute` takes slice of all the particles instead of two vectors, massive and massless.
- Generic brute-force compute methods implemented directly using `Normed` trait for internal vectors.
- `Vector` trait generic over full array type instead of dimension and scalar.
- 2D f32 vectors use `glam::Vec2` instead of `glam::Vec3A`.
- `gpu::BruteForce` only available for 3D f32 vectors.

### Removed

- `ParticleSet`. Use `accelerations` and `map_accelerations` methods on iterators instead.

## [0.4.0] - 2023-02-22

### Added

- `result_mut` method of `ParticleSet` that returns mutable references and acceleration.
- `ComputeMethod` trait defining how the acceleration of particles is computed.
- Sequential, parallel and GPU `BruteForce` ComputeMethod implementations (structs in their respective modules).
- `Scalar` trait to represent components of a vector and the type of the `gravitational parameter`.

### Changed

- `result` method of `ParticleSet` returns immutable references. Use `result_mut` instead for mutable references.
- `Vector` trait is now generic with a scalar to represent vector types of any float.
- A dimension is no longer required to be bound to a given vector by the user.
- `result` and `accelerations` methods of `ParticleSet` take a `ComputeMethod` as a parameter.
- Associated type `Vector` of `Particle` does not need to be wrapped in `VectorDescriptor`.
- `Particle` implementations require associated type `Scalar`.

### Removed

- #[dim] attribute and `particle` attribute macro.
- `Descriptor`, `IntoVector` and `FromVector` traits.
- `VectorDescriptor` wrapper.

## [0.3.1] - 2022-11-26

### Changed

- Improve documentation.

## [0.3.0] - 2022-10-01

### Added

- New useful methods for iterating over the `ParticleSet`.
- `particle` attribute macro to derive `Particle`. Its argument is the dimension of the vector type used for the position.
- `Descriptor` trait for the type of an arbitrary vector.
- `VectorDescriptor` struct for an arbitrary vector and its dimension.
- `FromVector` and `IntoVector` traits to convert between an arbitrary vector and a SIMD vector for computations.

### Changed

- Return tuple of `result` in `ParticleSet` is reversed.
- Associated type `Vector` of `Particle` now requires vector type to be wrapped with a `VectorDescriptor`, binding a dimension to it.
- Deriving `Particle` requires #[dim] attribute to associate a dimension to the vector type used for the position.
- `Vector` trait is now generic with a constant and only cares about From & Into arrays implementations.

### Removed

- `Vector` and `NormedVector` traits.
- Features `glam`, `nalgebra` and `cgmath`.
- `normed_vector!` macro.

## [0.2.0] - 2022-09-17

### Added

- `Vector` and `NormedVector` traits to describe generic vectors.
- Features `glam` (default feature), `nalgebra` and `cgmath` implements `Vector` and `NormedVector` for their vector types.
- `normed_vector!` macro to easily implement (recommended) `Vector` and `NormedVector` for a user-defined vector.

### Changed

- `Particle` now generic regarding its position type to allow for any vector to be used (any dimension, SIMD, ...). Uses new `Vector` and `NormedVector` traits.

## 0.1.0 - 2022-08-18

### Added

- Initial release.

[0.7.0]: https://github.com/Canleskis/particular/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/Canleskis/particular/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/Canleskis/particular/compare/v0.5.2...v0.6.0
[0.5.2]: https://github.com/Canleskis/particular/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/Canleskis/particular/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/Canleskis/particular/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Canleskis/particular/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/Canleskis/particular/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/Canleskis/particular/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Canleskis/particular/compare/v0.1.6...v0.2.0
