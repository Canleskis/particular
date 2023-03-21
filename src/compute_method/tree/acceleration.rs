use std::ops::{Add, Div, Mul, Sub};

use crate::vector::Normed;

use super::{orthtree::Orthant, Node, NodeID, Tree, TreeData};

pub(crate) trait TreeAcceleration<T, S> {
    fn acceleration_at(&self, position: T, node: Option<NodeID>, theta: S) -> T;
}

impl<const N: usize, T, S> TreeAcceleration<T, S> for Tree<Orthant<N, S>, (T, S)>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<S, Output = T>
        + Div<S, Output = T>
        + Normed<Output = S>,
    S: Copy + Default + PartialOrd + Mul<Output = S> + Div<Output = S>,
    (T, S): TreeData<Output = (T, S)>,
{
    fn acceleration_at(&self, position: T, node: Option<NodeID>, theta: S) -> T {
        let Some(id) = node else {
            return T::default();
        };
        let id = id as usize;

        let (position2, mu) = self.data[id];

        let dir = position2 - position;
        let mag_2 = dir.length_squared();
        let mag = T::sqrt(mag_2);

        match self.nodes[id] {
            Node::Internal(Orthant(o, width)) if theta < width / mag => {
                o.iter().fold(T::default(), |acceleration, &q| {
                    acceleration + self.acceleration_at(position, q, theta)
                })
            }
            _ => {
                if mag_2 == S::default() {
                    return dir;
                }

                dir * mu / (mag_2 * mag)
            }
        }
    }
}
