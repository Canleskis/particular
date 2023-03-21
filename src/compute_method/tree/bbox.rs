use super::orthtree::Orthant;

#[derive(Clone, Copy)]
pub(crate) struct BoundingBox<V> {
    min: V,
    max: V,
}

pub(crate) trait BoundingBoxOps: Copy {
    type Orthant;

    type Array;

    const IDENTITY: Self;

    fn new(min: Self::Array, max: Self::Array) -> Self;

    fn min(self) -> Self::Array;

    fn max(self) -> Self::Array;

    fn center(self) -> Self::Array;

    fn size(self) -> Self::Array;
}

pub(crate) trait BoundingBoxExtend: BoundingBoxOps {
    type Vector;

    fn containing(positions: impl Iterator<Item = Self::Vector>) -> Self {
        let mut result = Self::IDENTITY;
        for position in positions {
            result.extend(&position)
        }
        result
    }

    fn extend(&mut self, with: &Self::Vector);
}

macro_rules! impl_bbox {
    ($v: ty, $s: ty, $dim: literal, $o: literal) => {
        impl BoundingBoxOps for BoundingBox<$v> {
            type Orthant = Orthant<$o, $s>;

            type Array = [$s; $dim];

            const IDENTITY: Self = Self {
                min: <$v>::splat(<$s>::INFINITY),
                max: <$v>::splat(<$s>::NEG_INFINITY),
            };

            fn new(min: [$s; $dim], max: [$s; $dim]) -> Self {
                Self {
                    min: min.into(),
                    max: max.into(),
                }
            }

            fn min(self) -> [$s; $dim] {
                self.min.into()
            }

            fn max(self) -> [$s; $dim] {
                self.max.into()
            }

            fn center(self) -> [$s; $dim] {
                ((self.min + self.max) / 2.0).into()
            }

            fn size(self) -> [$s; $dim] {
                (self.max - self.min).into()
            }
        }

        impl BoundingBoxExtend for BoundingBox<$v> {
            type Vector = $v;

            fn extend(&mut self, with: &Self::Vector) {
                self.min = self.min.min(*with);
                self.max = self.max.max(*with);
            }
        }
    };
}

impl_bbox!(glam::Vec2, f32, 2, 4);
impl_bbox!(glam::DVec2, f64, 2, 4);
impl_bbox!(glam::Vec3A, f32, 3, 8);
impl_bbox!(glam::DVec3, f64, 3, 8);
