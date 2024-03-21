mod pyramid;
mod scale_space_extrema;

/// assumed gaussian blur for input image
const SIFT_INIT_SIGMA: f32 = 0.5;

use float_ord::FloatOrd;
use image::{DynamicImage, ImageBuffer, Luma};
use imageproc::filter::gaussian_blur_f32;
use pyramid::Pyramid;

use crate::scale_space_extrema::find_scale_space_extrema;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

type GrayImageBuffer = ImageBuffer<Luma<f32>, Vec<f32>>;

/// A point of interest in an image.
/// This pretty much follows from OpenCV conventions.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KeyPoint {
    /// The horizontal coordinate in a coordinate system is
    /// defined s.t. +x faces right and starts from the top
    /// of the image.
    /// the vertical coordinate in a coordinate system is defined
    /// s.t. +y faces toward the bottom of an image and starts
    /// from the left side of the image.
    pub point: (f32, f32),
    /// The magnitude of response from the detector.
    pub response: f32,

    /// The radius defining the extent of the keypoint, in pixel units
    pub size: f32,

    /// The level of scale space in which the keypoint was detected.
    pub octave: u32,

    /// The orientation angle
    pub angle: f32,
}

pub struct Sift {
    num_features: usize,
    num_octave_layers: usize,
    contrast_threshold: f32,
    edge_threshold: f32,
    sigma: f32,
}

impl Default for Sift {
    fn default() -> Self {
        Sift {
            num_features: 0,
            num_octave_layers: 3,
            contrast_threshold: 0.04,
            edge_threshold: 10.0,
            sigma: 1.6,
        }
    }
}

/// Doubles the image size and blurs
fn create_initial_image(image: &DynamicImage, sigma: f32) -> GrayImageBuffer {
    let mut image = image.to_luma32f();
    let image = image::imageops::resize(
        &image,
        image.width() * 2,
        image.height() * 2,
        image::imageops::FilterType::Triangle,
    );
    let sig_diff = std::cmp::max(
        FloatOrd(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4.),
        FloatOrd(0.01),
    )
    .0
    .sqrt();
    gaussian_blur_f32(&image, sig_diff)
}

impl Sift {
    pub fn extract(&self, image: &DynamicImage) -> Vec<KeyPoint> {
        let num_octaves = ((std::cmp::min(image.width(), image.height()) as f64).ln() / 2f64.ln()
            - 2f64)
            .round() as usize;

        let base = create_initial_image(image, self.sigma);
        let gaussian_pyramid =
            Pyramid::build_gaussian(base, num_octaves, self.num_octave_layers, self.sigma);
        let dog_pyramid = gaussian_pyramid.derive_difference_of_gaussians();

        let mut keypoints = find_scale_space_extrema(
            &gaussian_pyramid,
            &dog_pyramid,
            self.contrast_threshold,
            self.edge_threshold,
            self.sigma,
        );

        // keypoints.sort_by_key(|kp| {
        //     (
        //         FloatOrd(kp.point.0),
        //         FloatOrd(kp.point.1),
        //         FloatOrd(kp.size),
        //         FloatOrd(kp.angle),
        //         FloatOrd(-kp.response),
        //         // kp.octave,
        //     )
        // });
        keypoints.sort_by_key(|kp| FloatOrd(-kp.response));

        // resize all keypoints, as we have doubled the image size
        let scale = 0.5;
        for kp in &mut keypoints {
            kp.point.0 *= scale;
            kp.point.1 *= scale;
            kp.size *= scale;
            // TODO: remap the octave
        }

        keypoints

        // todo!()
    }
}
