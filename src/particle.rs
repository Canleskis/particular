use glam::Vec3;

pub trait Particle {
    fn position(&self) -> Vec3;
    
    fn mu(&self) -> f32;
}

pub(crate) type PointMass = (Vec3, f32);

pub(crate) trait ToPointMass {
    fn point_mass(&self) -> PointMass;
}

impl<P: Particle> ToPointMass for P {
    fn point_mass(&self) -> PointMass {
        (self.position(), self.mu())
    }
}