use cv_sift::Sift;
use std::path::Path;

fn image_to_kps(path: impl AsRef<Path>) -> (Vec<cv_sift::KeyPoint>) {
    Sift::default().extract(&image::open(path).unwrap())
}

#[test]
fn tsukuba() {
    let kps = image_to_kps("../res/tsukuba.png");

    println!("Found {} keypoints", kps.len());
    for kp in kps {
        println!("{:?}", kp);
    }
}
