use crate::GrayImageBuffer;
use image::Luma;
use imageproc::filter::gaussian_blur_f32;
use std::marker::PhantomData;

// use type-state pattern to segregate gaussian pyramid from DoG pyramid
pub struct Gaussian;
pub struct DifferenceOfGaussians;

pub struct Octave<T>(pub Vec<GrayImageBuffer>, PhantomData<T>);

impl<T> Octave<T> {
    #[inline]
    pub fn last(&self) -> &GrayImageBuffer {
        self.0.last().unwrap()
    }

    #[inline]
    pub fn iter(&self) -> std::slice::Iter<GrayImageBuffer> {
        self.0.iter()
    }

    /// Same as [`iter`] but skips the first and last images
    #[inline]
    pub fn enumerate_middle(&self) -> impl Iterator<Item = (u32, &GrayImageBuffer)> {
        self.0
            .iter()
            .enumerate()
            .skip(1)
            .take(self.0.len() - 2)
            .map(|(i, img)| (i as u32, img))
    }

    #[inline]
    pub fn get_pixel(&self, x: u32, y: u32, i: u32) -> f32 {
        self.0[i as usize].get_pixel(x, y).0[0]
    }

    #[inline]
    pub fn width(&self) -> u32 {
        self.0[0].width()
    }

    #[inline]
    pub fn height(&self) -> u32 {
        self.0[0].height()
    }
}

impl<T> std::ops::Index<usize> for Octave<T> {
    type Output = GrayImageBuffer;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T> std::ops::Index<u32> for Octave<T> {
    type Output = GrayImageBuffer;

    fn index(&self, index: u32) -> &Self::Output {
        &self.0[index as usize]
    }
}

impl Octave<Gaussian> {
    fn build_gaussian(base: GrayImageBuffer, sig: &[f32], num_octave_layers: usize) -> Self {
        let mut result = Vec::with_capacity(num_octave_layers + 3);
        result.push(base);

        // num_octave_layers + 3 to get num_octave_layers + 2 octaves for DoG
        // and then we look only at the middle images, as those want to look at all neighbors
        for i in 1..num_octave_layers + 3 {
            result.push(gaussian_blur_f32(&result[i - 1], sig[i]));
        }

        Self(result, PhantomData)
    }

    /// Precompute Gaussian sigmas using the following formula:
    //
    //  \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
    //
    //  `sig[i]` is the incremental sigma value needed to compute
    //  the actual sigma of level i. Keeping track of incremental
    //  sigmas vs. total sigmas keeps the gaussian kernel small.
    fn precompute_gaussian_sigmas(sigma: f32, num_octave_layers: usize) -> Vec<f32> {
        let mut sig = Vec::with_capacity(num_octave_layers + 3);

        sig.push(sigma);
        let k = 2f32.powf(1f32 / num_octave_layers as f32);
        for i in 1..num_octave_layers + 3 {
            let sig_prev = k.powi(i as i32 - 1) * sigma;
            let sig_total = sig_prev * k;
            sig.push((sig_total.powi(2) - sig_prev.powi(2)).sqrt());
        }

        sig
    }

    pub fn derive_difference_of_gaussians(&self) -> Octave<DifferenceOfGaussians> {
        let mut result = Vec::with_capacity(self.0.len() - 1);
        for i in 1..self.0.len() {
            let dog =
                // TODO: does it matter what we subtract from what?
                imageproc::map::map_colors2(&self.0[i], &self.0[i - 1], |a, b| Luma([a[0] - b[0]]));
            result.push(dog);
        }
        Octave(result, PhantomData)
    }

    pub fn num_octave_layers(&self) -> usize {
        self.0.len() - 3
    }
}

impl Octave<DifferenceOfGaussians> {
    pub fn num_octave_layers(&self) -> usize {
        self.0.len() - 2
    }
}

pub struct Pyramid<T>(pub Vec<Octave<T>>, PhantomData<T>);

impl Pyramid<Gaussian> {
    pub fn build_gaussian(
        mut base: GrayImageBuffer,
        num_octaves: usize,
        num_octave_layers: usize,
        sigma: f32,
    ) -> Self {
        let sig = Octave::precompute_gaussian_sigmas(sigma, num_octave_layers);
        let mut pyramid = Vec::with_capacity(num_octaves);

        assert!(num_octaves >= 1);

        for _ in 0..num_octaves - 1 {
            let octave = Octave::build_gaussian(base, &sig, num_octave_layers);

            // get a new base for the next octave by downsampling the last image
            let last = octave.last();
            base = image::imageops::resize(
                last,
                last.width() / 2,
                last.height() / 2,
                image::imageops::FilterType::Nearest,
            );

            pyramid.push(octave);
        }

        // handle the last octave separately (we don't need to downsample the last image)
        let octave = Octave::build_gaussian(base, &sig, num_octave_layers);
        pyramid.push(octave);

        Self(pyramid, PhantomData)
    }

    pub fn derive_difference_of_gaussians(&self) -> Pyramid<DifferenceOfGaussians> {
        let mut result = Vec::with_capacity(self.0.len());
        for octave in &self.0 {
            result.push(octave.derive_difference_of_gaussians());
        }
        Pyramid(result, PhantomData)
    }

    pub fn num_octave_layers(&self) -> usize {
        self.0[0].num_octave_layers()
    }
}

impl Pyramid<DifferenceOfGaussians> {
    pub fn num_octave_layers(&self) -> usize {
        self.0[0].num_octave_layers()
    }
}

impl<T> Pyramid<T> {
    pub fn num_octaves(&self) -> usize {
        self.0.len()
    }

    pub fn iter(&self) -> std::slice::Iter<Octave<T>> {
        self.0.iter()
    }

    pub fn iter_enumerate(&self) -> impl Iterator<Item = (u32, &Octave<T>)> {
        self.0.iter().enumerate().map(|(i, img)| (i as u32, img))
    }
}
