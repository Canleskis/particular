# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased - 2023-22-02

### Breaking changes

- `ParticleSet::add` and similar now expect an iterator of particles. Use `ParticleSet::add_one` instead.

### Added

- Methods to create filled `ParticleSets`.

### Changed

- Generic brute-force compute methods implemented directly using `Normed` trait.

### Removed

- Private `Computable` trait.

## [0.4.0] - 2023-22-02

### Breaking changes

- Remove #[dim] attribute to derive `Particle` and `particle` attribute macro.
- Associated type `Vector` of `Particle` does not need to be wrapped in `VectorDescriptor`.
- `Particle` implementations require associated type `Scalar`.
- `result` method of `ParticleSet` returns immutable references.
- `result` and `accelerations` methods of `ParticleSet` take a `ComputeMethod` as a parameter.

### Added

- `result_mut` method of `ParticleSet` that returns mutable references and acceleration.
- `ComputeMethod` trait defining how the acceleration of particles is computed.
- Sequential, parallel and GPU `BruteForce` ComputeMethod implementations (structs in their respective modules).
- `Scalar` trait to represent components of a vector and the type of the `gravitational parameter`.

### Changed

- `result` method of `ParticleSet` returns immutable references. Use `result_mut` instead for mutable references.
- `Vector` trait is now generic with a scalar to represent vector types of any float.
- A dimension is no longer required to be bound to a given vector by the user.
- `ParticleSet` is agnostic to the computation of the acceleration.

### Removed

- #[dim] attribute and `particle` attribute macro.
- `Descriptor`, `IntoVector` and `FromVector` traits.
- `VectorDescriptor` wrapper.

## [0.3.1] - 2022-26-11

### Changed

- Improve documentation.

## [0.3.0] - 2022-10-01

### Breaking changes

- Return tuple of `result` in `ParticleSet` is reversed.
- Associated type `Vector` of `Particle` now requires vector type to be wrapped with a `VectorDescriptor`, binding a dimension to it.
- Deriving `Particle` requires #[dim] attribute to associate a dimension to the vector type used for the position.

### Added

- New useful methods for iterating over the `ParticleSet`.
- `particle` attribute macro to derive `Particle`. Its argument is the dimension of the vector type used for the position.
- `Descriptor` trait for the type of an arbitrary vector.
- `VectorDescriptor` struct for an arbitrary vector and its dimension.
- `FromVector` and `IntoVector` traits to convert between an arbitrary vector and a SIMD vector for computations.

### Changed

- `Vector` trait is now generic with a constant and only cares about From & Into arrays implementations.

### Removed

- `Vector` and `NormedVector` traits.
- Features `glam`, `nalgebra` and `cgmath`.
- `normed_vector!` macro.

## [0.2.0] - 2022-09-17

### Breaking changes

- `Particle` now generic regarding its position type to allow for any vector to be used (any dimension, SIMD, ...). Uses new `Vector` and `NormedVector` traits.

### Added

- `Vector` and `NormedVector` traits to describe generic vectors.
- Features `glam` (default feature), `nalgebra` and `cgmath` implements `Vector` and `NormedVector` for their vector types.
- `normed_vector!` macro to easily implement (recommended) `Vector` and `NormedVector` for a user-defined vector.

## 0.1.0 - 2022-08-18

### Added

- Initial release.

[0.4.0]: https://github.com/Canleskis/particular/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/Canleskis/particular/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/Canleskis/particular/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Canleskis/particular/compare/v0.1.6...v0.2.0
