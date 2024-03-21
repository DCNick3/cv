mod orientation_histogram;

use crate::pyramid::{DifferenceOfGaussians, Gaussian, Octave, Pyramid};
use crate::scale_space_extrema::orientation_histogram::calc_orientation_histogram;
use crate::KeyPoint;
use cv_core::nalgebra::{Matrix2, Matrix3, Vector3};
use float_ord::FloatOrd;

/// default width of descriptor histogram array
const SIFT_DESCR_WIDTH: usize = 4;

/// default number of bins per histogram in descriptor array
const SIFT_DESCR_HIST_BINS: usize = 8;

/// width of border in which to ignore keypoints
const SIFT_IMG_BORDER: u32 = 5;

/// maximum steps of keypoint interpolation before failure
const SIFT_MAX_INTERP_STEPS: usize = 5;

/// default number of bins in histogram for orientation assignment
const SIFT_ORI_HIST_BINS: usize = 36;

/// determines gaussian sigma for orientation assignment
const SIFT_ORI_SIG_FCTR: f32 = 1.5;

/// determines the radius of the region used in orientation assignment
const SIFT_ORI_RADIUS: f32 = 4.5; // 3 * SIFT_ORI_SIG_FCTR;

/// orientation magnitude relative to max that results in new feature
const SIFT_ORI_PEAK_RATIO: f32 = 0.8;

/// determines the size of a single descriptor orientation histogram
const SIFT_DESCR_SCL_FCTR: f32 = 3.;

/// threshold on magnitude of elements of descriptor vector
const SIFT_DESCR_MAG_THR: f32 = 0.2;

/// factor used to convert floating-point descriptor to unsigned char
const SIFT_INT_DESCR_FCTR: f32 = 512.;

pub struct UnorientedKeyPoint {
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
    ///
    /// OpenCV does some bit-bang shenanigans; we do too.
    pub octave: u32,
}

impl UnorientedKeyPoint {
    pub fn with_angle(&self, angle: f32) -> KeyPoint {
        KeyPoint {
            point: self.point,
            response: self.response,
            size: self.size,
            octave: self.octave,
            angle,
        }
    }
}

#[inline]
fn is_extremum(
    octave: &Octave<DifferenceOfGaussians>,
    value: f32,
    layer: u32,
    x: u32,
    y: u32,
) -> bool {
    // check for maximum
    if value > 0. {
        for i in -1i32..=1 {
            for j in -1i32..=1 {
                for k in -1i32..=1 {
                    if value
                        < octave.get_pixel(
                            (x as i32 + j) as u32,
                            (y as i32 + k) as u32,
                            (layer as i32 + i) as u32,
                        )
                    {
                        return false;
                    }
                }
            }
        }
    }
    // check for minimum
    else {
        for i in -1i32..=1 {
            for j in -1i32..=1 {
                for k in -1i32..=1 {
                    if value
                        > octave.get_pixel(
                            (x as i32 + j) as u32,
                            (y as i32 + k) as u32,
                            (layer as i32 + i) as u32,
                        )
                    {
                        return false;
                    }
                }
            }
        }
    }

    return true;
}

/// Computes the partial derivatives in x, y, and scale of a pixel in the DoG scale space pyramid.
#[inline]
fn deriv_3d(octave: &Octave<DifferenceOfGaussians>, x: u32, y: u32, i: u32) -> Vector3<f32> {
    let dx = 0.5 * (octave.get_pixel(x + 1, y, i) - octave.get_pixel(x - 1, y, i));
    let dy = 0.5 * (octave.get_pixel(x, y + 1, i) - octave.get_pixel(x, y - 1, i));
    let di = 0.5 * (octave.get_pixel(x, y, i + 1) - octave.get_pixel(x, y, i - 1));

    Vector3::new(dx, dy, di)
}

/// Computes the 3D Hessian matrix for a pixel in the DoG scale space pyramid.
///
/// ```ignore
/// / Ixx  Ixy  Ixi \
/// | Ixy  Iyy  Iyi |
/// \ Ixi  Iyi  Iii /
///```
#[inline]
fn hessian_3d(octave: &Octave<DifferenceOfGaussians>, x: u32, y: u32, i: u32) -> Matrix3<f32> {
    let v = octave.get_pixel(x, y, i);
    let dxx = octave.get_pixel(x + 1, y, i) - 2. * v + octave.get_pixel(x - 1, y, i);
    let dyy = octave.get_pixel(x, y + 1, i) - 2. * v + octave.get_pixel(x, y - 1, i);
    let dii = octave.get_pixel(x, y, i + 1) - 2. * v + octave.get_pixel(x, y, i - 1);

    let dxy = 0.25
        * (octave.get_pixel(x + 1, y + 1, i)
            - octave.get_pixel(x - 1, y + 1, i)
            - octave.get_pixel(x + 1, y - 1, i)
            + octave.get_pixel(x - 1, y - 1, i));
    let dxi = 0.25
        * (octave.get_pixel(x + 1, y, i + 1)
            - octave.get_pixel(x - 1, y, i + 1)
            - octave.get_pixel(x + 1, y, i - 1)
            + octave.get_pixel(x - 1, y, i - 1));
    let dyi = 0.25
        * (octave.get_pixel(x, y + 1, i + 1)
            - octave.get_pixel(x, y - 1, i + 1)
            - octave.get_pixel(x, y + 1, i - 1)
            + octave.get_pixel(x, y - 1, i - 1));

    Matrix3::new(dxx, dxy, dxi, dxy, dyy, dyi, dxi, dyi, dii)
}

/// Computes the 2D Hessian matrix for a pixel in one of the layers of the DoG scale space pyramid.
///
/// ```ignore
/// / Ixx  Ixy \
/// \ Ixy  Iyy /
#[inline]
fn hessian_2d(octave: &Octave<DifferenceOfGaussians>, x: u32, y: u32, i: u32) -> Matrix2<f32> {
    let d = octave.get_pixel(x, y, i);
    let dxx = octave.get_pixel(x + 1, y, i) - 2. * d + octave.get_pixel(x - 1, y, i);
    let dyy = octave.get_pixel(x, y + 1, i) - 2. * d + octave.get_pixel(x, y - 1, i);
    let dxy = 0.25
        * (octave.get_pixel(x + 1, y + 1, i)
            - octave.get_pixel(x - 1, y + 1, i)
            - octave.get_pixel(x + 1, y - 1, i)
            + octave.get_pixel(x - 1, y - 1, i));

    Matrix2::new(dxx, dxy, dxy, dyy)
}

/// Interpolates a scale-space extremum's location and scale to subpixel
/// accuracy to form an image feature. Rejects features with low contrast.
/// Based on Section 4 of Lowe's paper.
fn adjust_local_extrema(
    octave_idx: u32,
    octave: &Octave<DifferenceOfGaussians>,
    mut x: u32,
    mut y: u32,
    mut i: u32,
    contrast_threshold: f32,
    edge_threshold: f32,
    sigma: f32,
) -> Option<(UnorientedKeyPoint, (u32, u32, u32))> {
    let width = octave.width();
    let height = octave.height();

    // move the sample point if `interp_step` reports a change > 0.5
    let mut step = 0;
    let (dx, dy, di) = loop {
        let d = deriv_3d(octave, x, y, i);
        let h = hessian_3d(octave, x, y, i);

        let res = h.lu().solve(&d).expect("Linear resolution failed");
        let (dx, dy, di) = (res[0], res[1], res[2]);

        if dx.abs() < 0.5 && dy.abs() < 0.5 && di.abs() < 0.5 {
            // we've converged!
            break (dx, dy, di);
        }

        step += 1;
        // we did not converge in the allowed number of steps
        if step >= SIFT_MAX_INTERP_STEPS {
            return None;
        }

        // reject stuff that's risking overflow
        const OUT_OF_RANGE: f32 = i32::MAX as f32 / 3.;
        if dx.abs() > OUT_OF_RANGE || dy.abs() > OUT_OF_RANGE || di.abs() > OUT_OF_RANGE {
            return None;
        }

        // we might go out-of-bounds on the negative side, so get it as a signed value to check
        let x_signed = (x as i32) + (dx.round() as i32);
        let y_signed = (y as i32) + (dy.round() as i32);
        let i_signed = (i as i32) + (di.round() as i32);

        // reject stuff that out of bounds
        if !(SIFT_IMG_BORDER as i32..(width - SIFT_IMG_BORDER) as i32).contains(&x_signed)
            || !(SIFT_IMG_BORDER as i32..(height - SIFT_IMG_BORDER) as i32).contains(&y_signed)
            || !(1..=octave.num_octave_layers() as i32).contains(&i_signed)
        {
            return None;
        }

        // update the sample point
        x = x_signed as u32;
        y = y_signed as u32;
        i = i_signed as u32;
    };

    // reject features with low contrast
    let d = deriv_3d(octave, x, y, i);
    let x_hat = Vector3::new(dx, dy, di);

    let contrast = octave.get_pixel(x, y, i) + 0.5 * d.dot(&x_hat);

    if contrast.abs() < contrast_threshold / octave.num_octave_layers() as f32 {
        return None;
    }

    // reject features which a too edge-like
    let h = hessian_2d(octave, x, y, i);
    let trace = h.trace();
    let det = h.determinant();

    // negative determinant -> curvatures have different signs; reject feature
    if det <= 0. || trace.powi(2) / det >= (edge_threshold + 1.).powi(2) / edge_threshold {
        return None;
    }

    return Some((
        UnorientedKeyPoint {
            point: (
                (x as f32 + dx) * (1 << octave_idx) as f32,
                (y as f32 + dy) * (1 << octave_idx) as f32,
            ),
            response: contrast.abs(),
            size: sigma
                * 2f32.powf((i as f32 + di) / octave.num_octave_layers() as f32)
                * (1 << octave_idx) as f32
                * 2.0,
            octave: octave_idx + (i << 8) + ((((di + 0.5) * 255.).round() as u32) << 16),
        },
        (x, y, i),
    ));
}

fn emit_angles(
    keypoints: &mut Vec<KeyPoint>,
    keypoint: &UnorientedKeyPoint,
    hist: [FloatOrd<f32>; SIFT_ORI_HIST_BINS],
) {
    let histogram_max = *hist.iter().max().unwrap();
    let magnitude_threshold = FloatOrd(histogram_max.0 * SIFT_ORI_PEAK_RATIO);

    for j in 0..SIFT_ORI_HIST_BINS {
        // get left and right, taking into account the fact that we are using a circular histogram
        let l = if j > 0 { j - 1 } else { SIFT_ORI_HIST_BINS - 1 };
        let r = if j < SIFT_ORI_HIST_BINS - 1 { j + 1 } else { 0 };

        // if this is a local maximum that is greater than the threshold, add it to the list of keypoints
        if hist[j] > hist[l] && hist[j] > hist[r] && hist[j] >= magnitude_threshold {
            // get some additional precision (fractional bin coordinates) by intrapolating the peak as a quadratic function
            // see https://www.desmos.com/calculator/ipec8bpjeo
            let bin =
                j as f32 + 0.5 * (hist[l].0 - hist[r].0) / (hist[l].0 - 2. * hist[j].0 + hist[r].0);
            let bin = if bin < 0. {
                SIFT_ORI_HIST_BINS as f32 + bin
            } else if bin >= SIFT_ORI_HIST_BINS as f32 {
                bin - SIFT_ORI_HIST_BINS as f32
            } else {
                bin
            };

            // unlike OpenCV, return the orientation in radians
            let mut angle = std::f32::consts::PI * 2.
                - bin * std::f32::consts::PI * 2. / SIFT_ORI_HIST_BINS as f32;
            if (angle - std::f32::consts::PI * 2.).abs() < f32::EPSILON {
                angle = 0.;
            }

            keypoints.push(keypoint.with_angle(angle));
        }
    }
}

pub fn find_scale_space_extrema(
    gaussian_pyramid: &Pyramid<Gaussian>,
    pyramid: &Pyramid<DifferenceOfGaussians>,
    contrast_threshold: f32,
    edge_threshold: f32,
    sigma: f32,
) -> Vec<KeyPoint> {
    let prelim_contr_thr = 0.5 * contrast_threshold / pyramid.num_octave_layers() as f32;

    let mut keypoints = Vec::new();

    for ((o, octave), gaussian_octave) in pyramid.iter_enumerate().zip(gaussian_pyramid.iter()) {
        for (i, layer) in octave.enumerate_middle() {
            let height = layer.height();
            let width = layer.width();

            for y in SIFT_IMG_BORDER..height - SIFT_IMG_BORDER {
                for x in SIFT_IMG_BORDER..width - SIFT_IMG_BORDER {
                    let value = layer.get_pixel(x, y).0[0];
                    if value.abs() < prelim_contr_thr {
                        continue;
                    }
                    if !is_extremum(&octave, value, i, x, y) {
                        continue;
                    }

                    let Some((kpt, (x1, y1, i1))) = adjust_local_extrema(
                        o,
                        octave,
                        x,
                        y,
                        i,
                        contrast_threshold,
                        edge_threshold,
                        sigma,
                    ) else {
                        continue;
                    };

                    let scl_octv = kpt.size * 0.5 / (1 << o) as f32;

                    let hist = calc_orientation_histogram(
                        &gaussian_octave[i1],
                        x1,
                        y1,
                        (SIFT_ORI_RADIUS * scl_octv).round() as i32,
                        SIFT_ORI_SIG_FCTR * scl_octv,
                    );

                    emit_angles(&mut keypoints, &kpt, hist);
                }
            }
        }
    }

    // keypoints.sort_by_key(|kp| FloatOrd(-kp.response));

    keypoints
}
