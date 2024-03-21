use crate::scale_space_extrema::SIFT_ORI_HIST_BINS;
use crate::GrayImageBuffer;
use float_ord::FloatOrd;

pub fn calc_orientation_histogram(
    img: &GrayImageBuffer,
    x: u32,
    y: u32,
    radius: i32,
    sigma: f32,
) -> [FloatOrd<f32>; SIFT_ORI_HIST_BINS] {
    let len = (radius * 2 + 1).pow(2) as usize;

    let expf_scale = -1. / (2. * sigma * sigma);

    let mut x_vals = Vec::with_capacity(len);
    let mut y_vals = Vec::with_capacity(len);
    let mut ori_vals = Vec::with_capacity(len);
    let mut w_vals = Vec::with_capacity(len);

    for i in -radius..=radius {
        let x = x as i32 + i;
        if x <= 0 || x >= img.width() as i32 - 1 {
            continue;
        }
        let x = x as u32;
        for j in -radius..=radius {
            let y = y as i32 + j;
            if y <= 0 || y >= img.height() as i32 - 1 {
                continue;
            }
            let y = y as u32;

            let dx = img.get_pixel(x + 1, y).0[0] - img.get_pixel(x - 1, y).0[0];
            let dy = img.get_pixel(x, y + 1).0[0] - img.get_pixel(x, y - 1).0[0];

            x_vals.push(dx);
            y_vals.push(dy);
            w_vals.push(((i as f32).powi(2) + (j as f32).powi(2)) * expf_scale);
        }
    }

    // TODO: use simd
    w_vals.iter_mut().for_each(|w| *w = w.exp());
    y_vals.iter().zip(x_vals.iter()).for_each(|(&y, &x)| {
        ori_vals.push(y.atan2(x));
    });
    x_vals.iter_mut().zip(y_vals.iter()).for_each(|(dest, &y)| {
        let x = *dest;
        *dest = (x.powi(2) + y.powi(2)).sqrt();
    });

    let mag_vals = x_vals;

    let mut temphist = [0.; SIFT_ORI_HIST_BINS + 4]; // 2 extra bins in the beginning and end
                                                     // TODO: use simd
    for ((&mag, &ori), &w) in mag_vals.iter().zip(ori_vals.iter()).zip(w_vals.iter()) {
        let mut bin =
            (ori / (2. * std::f32::consts::PI) * SIFT_ORI_HIST_BINS as f32).round() as i32;
        if bin >= SIFT_ORI_HIST_BINS as i32 {
            bin -= SIFT_ORI_HIST_BINS as i32;
        }
        if bin < 0 {
            bin += SIFT_ORI_HIST_BINS as i32;
        }
        temphist[(bin + 2) as usize] += mag * w;
    }

    // smooth the histogram
    temphist[1] = temphist[SIFT_ORI_HIST_BINS + 1];
    temphist[0] = temphist[SIFT_ORI_HIST_BINS];
    temphist[SIFT_ORI_HIST_BINS + 2] = temphist[2];
    temphist[SIFT_ORI_HIST_BINS + 3] = temphist[3];

    // TODO: use simd
    let mut hist = [FloatOrd(0f32); SIFT_ORI_HIST_BINS];
    for i in 2..SIFT_ORI_HIST_BINS + 2 {
        hist[i - 2] = FloatOrd(
            (temphist[i - 2] + temphist[i + 2]) * (1. / 16.)
                + (temphist[i - 1] + temphist[i + 1]) * (4. / 16.)
                + temphist[i] * (6. / 16.),
        );
    }

    hist
}
