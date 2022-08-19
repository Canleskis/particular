use glam::Vec3;

/// Trait to describe a particle which consists of a `position` and a gravitational parameter `mu`.
/// 
/// ### Deriving:
/// 
/// Used in most cases, when your type has fields named `position` and `mu`
/// ```
/// #[derive(Particle)]
/// pub struct Body {
///     position: Vec3,
///     mu: f32,
///     ...
/// }
/// ```
/// 
/// ### Manual implementation:
/// 
/// Used when your type has more complex fields and cannot directly provide a position and a gravitational parameter.
/// ```
/// impl Particle for Body {
///     fn position(&self) -> Vec3 {
///         self.position
///     }
///     
///     fn mu(&self) -> f32 {
///         self.mass * G
///     }
/// }
/// ```
pub trait Particle {
    fn position(&self) -> Vec3;

    fn mu(&self) -> f32;
}

pub(crate) type PointMass = (Vec3, f32);

pub(crate) trait ToPointMass {
    fn to_point_mass(&self) -> PointMass;
}

impl<P: Particle> ToPointMass for P {
    fn to_point_mass(&self) -> PointMass {
        (self.position(), self.mu())
    }
}
