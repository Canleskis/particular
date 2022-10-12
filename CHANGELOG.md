# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2022-10-01

### Breaking changes

- Return tuple of `result` in `ParticleSet` is reversed.
- Associated type `Vector` of `Particle` now requires vector type to be wrapped with a `VectorDescriptor`, binding a dimension to it.
- Deriving `Particle` requires #[dim] attribute tagged on the position field to bind a dimension to the vector type used.

### Added

- `VectorDescriptor` struct for an arbitrary vector and its dimension.
- `Descriptor` trait for the type of an arbitrary vector.
- `FromVector` and `IntoVector` traits to convert between an arbitrary vector and a SIMD vector for computations.
- `gpu` feature to use compute shaders. The particle set needs a device and and queue to be initialized and get its result with this feature enabled.

### Changed

- `Vector` trait is now generic with a constant and only cares about Send and From + Into arrays implementations.

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

[0.3.0]: https://github.com/Canleskis/particular/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Canleskis/particular/compare/v0.1.6...v0.2.0
