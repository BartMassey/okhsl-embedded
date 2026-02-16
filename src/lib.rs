// Copyright(c) 2021 Björn Ottosson
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files(the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#![no_std]
#![warn(missing_docs)]
#![allow(clippy::excessive_precision)]

//! A `no_std` Rust implementation of OKHSL and OKHSV color space conversions.
//!
//! This library provides perceptually uniform color spaces (OKHSL and OKHSV) based on
//! Björn Ottosson's Oklab color space. All functions use f32 floating point and are
//! compatible with embedded systems.
//!
//! # Features
//!
//! - `no_std` compatible for embedded systems
//! - OKHSL, OKHSV, and Oklab color spaces
//! - sRGB and custom gamma correction
//! - Multiple gamut clipping strategies
//! - Perceptually uniform color manipulation
//!
//! # Example
//!
//! ```
//! use okhsl_embedded::{HSL, RGB, okhsl_to_srgb, srgb_to_okhsl};
//!
//! // Convert OKHSL to sRGB
//! let hsl = HSL { h: 0.5, s: 0.8, l: 0.6 };
//! let rgb = okhsl_to_srgb(hsl);
//!
//! // Convert back
//! let hsl2 = srgb_to_okhsl(rgb);
//! ```
//!
//! # Color Spaces
//!
//! - **Oklab**: Perceptually uniform color space with L (lightness), a, b (chromaticity)
//! - **OKHSL**: Hue, Saturation, Lightness in perceptually uniform space
//! - **OKHSV**: Hue, Saturation, Value in perceptually uniform space
//! - **sRGB**: Standard RGB color space with gamma correction
//!
//! # References
//!
//! Original implementation: <https://bottosson.github.io/posts/oklab/>

use core::f32::consts::PI;
use libm::{atan2f, cbrtf, cosf, fmaxf, fminf, powf, sinf, sqrtf};

/// Lab color in Oklab color space
///
/// Oklab is a perceptually uniform color space with:
/// - `l`: Perceived lightness (0.0 to 1.0)
/// - `a`: Green-red axis
/// - `b`: Blue-yellow axis
#[derive(Debug, Clone, Copy)]
pub struct Lab {
    /// Perceived lightness (0.0 to 1.0)
    pub l: f32,
    /// Green-red axis
    pub a: f32,
    /// Blue-yellow axis
    pub b: f32,
}

/// RGB color representation with linear or gamma-corrected values
///
/// Values typically range from 0.0 to 1.0. The interpretation (linear vs gamma-corrected)
/// depends on the conversion function used.
#[derive(Debug, Clone, Copy)]
pub struct RGB {
    /// Red component (0.0 to 1.0)
    pub r: f32,
    /// Green component (0.0 to 1.0)
    pub g: f32,
    /// Blue component (0.0 to 1.0)
    pub b: f32,
}

/// HSV color in Okhsv color space
///
/// Okhsv provides a perceptually uniform hue-saturation-value representation:
/// - `h`: Hue (0.0 to 1.0, wraps around)
/// - `s`: Saturation (0.0 to 1.0, where 0 is gray and 1 is fully saturated)
/// - `v`: Value/brightness (0.0 to 1.0, where 0 is black and 1 is maximum brightness)
#[derive(Debug, Clone, Copy)]
pub struct HSV {
    /// Hue (0.0 to 1.0, wraps around)
    pub h: f32,
    /// Saturation (0.0 to 1.0)
    pub s: f32,
    /// Value/brightness (0.0 to 1.0)
    pub v: f32,
}

/// HSL color in Okhsl color space
///
/// Okhsl provides a perceptually uniform hue-saturation-lightness representation:
/// - `h`: Hue (0.0 to 1.0, wraps around)
/// - `s`: Saturation (0.0 to 1.0, where 0 is gray and 1 is fully saturated)
/// - `l`: Lightness (0.0 to 1.0, where 0 is black, 0.5 is mid-tone, 1 is white)
#[derive(Debug, Clone, Copy)]
pub struct HSL {
    /// Hue (0.0 to 1.0, wraps around)
    pub h: f32,
    /// Saturation (0.0 to 1.0)
    pub s: f32,
    /// Lightness (0.0 to 1.0)
    pub l: f32,
}

/// Lightness and chroma pair
///
/// Represents a color's position in the gamut by its lightness and chroma values:
/// - `l`: Lightness component
/// - `c`: Chroma (colorfulness) component
#[derive(Debug, Clone, Copy)]
pub struct LC {
    /// Lightness component
    pub l: f32,
    /// Chroma component
    pub c: f32,
}

/// Alternative representation of (L_cusp, C_cusp)
///
/// Encoded so S = C_cusp/L_cusp and T = C_cusp/(1-L_cusp).
/// The maximum value for C in the triangle is then found as fmin(S*L, T*(1-L)), for a given L.
#[derive(Debug, Clone, Copy)]
pub struct ST {
    /// S parameter: C_cusp/L_cusp
    pub s: f32,
    /// T parameter: C_cusp/(1-L_cusp)
    pub t: f32,
}

/// Chroma boundaries for a given hue and lightness
///
/// Contains the minimum, mid-range, and maximum chroma values:
/// - `c_0`: Minimum chroma
/// - `c_mid`: Mid-range chroma
/// - `c_max`: Maximum chroma
#[derive(Debug, Clone, Copy)]
pub struct Cs {
    /// Minimum chroma
    pub c_0: f32,
    /// Mid-range chroma
    pub c_mid: f32,
    /// Maximum chroma
    pub c_max: f32,
}

#[inline]
fn clamp(x: f32, min: f32, max: f32) -> f32 {
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}

#[inline]
fn sgn(x: f32) -> f32 {
    (0.0f32 < x) as i32 as f32 - (x < 0.0f32) as i32 as f32
}

#[inline]
fn srgb_transfer_function(a: f32) -> f32 {
    if 0.0031308f32 >= a {
        12.92f32 * a
    } else {
        1.055f32 * powf(a, 0.4166666666666667f32) - 0.055f32
    }
}

#[inline]
fn srgb_transfer_function_inv(a: f32) -> f32 {
    if 0.04045f32 < a {
        powf((a + 0.055f32) / 1.055f32, 2.4f32)
    } else {
        a / 12.92f32
    }
}

/// Applies gamma correction to a linear RGB value
/// gamma: typical values are 2.2 (common for displays), 2.4 (sRGB linear segment), 1.8 (Mac historical)
#[inline]
pub fn gamma_transfer_function(a: f32, gamma: f32) -> f32 {
    if a <= 0.0 {
        0.0
    } else {
        powf(a, 1.0 / gamma)
    }
}

/// Converts gamma-corrected RGB to linear RGB
/// gamma: typical values are 2.2 (common for displays), 2.4 (sRGB linear segment), 1.8 (Mac historical)
#[inline]
pub fn gamma_transfer_function_inv(a: f32, gamma: f32) -> f32 {
    if a <= 0.0 {
        0.0
    } else {
        powf(a, gamma)
    }
}

/// Converts linear RGB to gamma-corrected RGB
///
/// # Parameters
///
/// * `rgb` - Linear RGB color to convert
/// * `gamma` - Gamma value (typical: 2.2 for displays, 2.4 for sRGB, 1.8 for Mac)
///
/// # Returns
///
/// Gamma-corrected RGB color
pub fn linear_rgb_to_gamma_rgb(rgb: RGB, gamma: f32) -> RGB {
    RGB {
        r: gamma_transfer_function(rgb.r, gamma),
        g: gamma_transfer_function(rgb.g, gamma),
        b: gamma_transfer_function(rgb.b, gamma),
    }
}

/// Converts gamma-corrected RGB to linear RGB
///
/// # Parameters
///
/// * `rgb` - Gamma-corrected RGB color to convert
/// * `gamma` - Gamma value (typical: 2.2 for displays, 2.4 for sRGB, 1.8 for Mac)
///
/// # Returns
///
/// Linear RGB color
pub fn gamma_rgb_to_linear_rgb(rgb: RGB, gamma: f32) -> RGB {
    RGB {
        r: gamma_transfer_function_inv(rgb.r, gamma),
        g: gamma_transfer_function_inv(rgb.g, gamma),
        b: gamma_transfer_function_inv(rgb.b, gamma),
    }
}

/// Converts linear RGB to standard sRGB
///
/// Applies the sRGB transfer function (gamma correction with special handling
/// for dark values) to convert from linear RGB to sRGB.
///
/// # Parameters
///
/// * `rgb` - Linear RGB color to convert
///
/// # Returns
///
/// sRGB color with gamma correction applied
pub fn linear_rgb_to_srgb(rgb: RGB) -> RGB {
    RGB {
        r: srgb_transfer_function(rgb.r),
        g: srgb_transfer_function(rgb.g),
        b: srgb_transfer_function(rgb.b),
    }
}

/// Converts standard sRGB to linear RGB
///
/// Applies the inverse sRGB transfer function to convert from sRGB to linear RGB.
///
/// # Parameters
///
/// * `rgb` - sRGB color to convert
///
/// # Returns
///
/// Linear RGB color
pub fn srgb_to_linear_rgb(rgb: RGB) -> RGB {
    RGB {
        r: srgb_transfer_function_inv(rgb.r),
        g: srgb_transfer_function_inv(rgb.g),
        b: srgb_transfer_function_inv(rgb.b),
    }
}

/// Converts linear sRGB to Oklab color space
///
/// # Parameters
///
/// * `c` - Linear sRGB color to convert
///
/// # Returns
///
/// Oklab color representation
pub fn linear_srgb_to_oklab(c: RGB) -> Lab {
    let l = 0.4122214708f32 * c.r + 0.5363325363f32 * c.g + 0.0514459929f32 * c.b;
    let m = 0.2119034982f32 * c.r + 0.6806995451f32 * c.g + 0.1073969566f32 * c.b;
    let s = 0.0883024619f32 * c.r + 0.2817188376f32 * c.g + 0.6299787005f32 * c.b;

    let l_ = cbrtf(l);
    let m_ = cbrtf(m);
    let s_ = cbrtf(s);

    Lab {
        l: 0.2104542553f32 * l_ + 0.7936177850f32 * m_ - 0.0040720468f32 * s_,
        a: 1.9779984951f32 * l_ - 2.4285922050f32 * m_ + 0.4505937099f32 * s_,
        b: 0.0259040371f32 * l_ + 0.7827717662f32 * m_ - 0.8086757660f32 * s_,
    }
}

/// Converts Oklab to linear sRGB color space
///
/// # Parameters
///
/// * `c` - Oklab color to convert
///
/// # Returns
///
/// Linear sRGB color representation (may be out of gamut)
pub fn oklab_to_linear_srgb(c: Lab) -> RGB {
    let l_ = c.l + 0.3963377774f32 * c.a + 0.2158037573f32 * c.b;
    let m_ = c.l - 0.1055613458f32 * c.a - 0.0638541728f32 * c.b;
    let s_ = c.l - 0.0894841775f32 * c.a - 1.2914855480f32 * c.b;

    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;

    RGB {
        r: 4.0767416621f32 * l - 3.3077115913f32 * m + 0.2309699292f32 * s,
        g: -1.2684380046f32 * l + 2.6097574011f32 * m - 0.3413193965f32 * s,
        b: -0.0041960863f32 * l - 0.7034186147f32 * m + 1.7076147010f32 * s,
    }
}

/// Finds the maximum saturation possible for a given hue that fits in sRGB
///
/// Saturation here is defined as S = C/L.
///
/// # Parameters
///
/// * `a` - Normalized a component (must satisfy a² + b² = 1)
/// * `b` - Normalized b component (must satisfy a² + b² = 1)
///
/// # Returns
///
/// Maximum saturation value for the given hue
pub fn compute_max_saturation(a: f32, b: f32) -> f32 {
    // Max saturation will be when one of r, g or b goes below zero.

    // Select different coefficients depending on which component goes below zero first
    let (k0, k1, k2, k3, k4, wl, wm, ws) = if -1.88170328f32 * a - 0.80936493f32 * b > 1.0 {
        // Red component
        (
            1.19086277f32,
            1.76576728f32,
            0.59662641f32,
            0.75515197f32,
            0.56771245f32,
            4.0767416621f32,
            -3.3077115913f32,
            0.2309699292f32,
        )
    } else if 1.81444104f32 * a - 1.19445276f32 * b > 1.0 {
        // Green component
        (
            0.73956515f32,
            -0.45954404f32,
            0.08285427f32,
            0.12541070f32,
            0.14503204f32,
            -1.2684380046f32,
            2.6097574011f32,
            -0.3413193965f32,
        )
    } else {
        // Blue component
        (
            1.35733652f32,
            -0.00915799f32,
            -1.15130210f32,
            -0.50559606f32,
            0.00692167f32,
            -0.0041960863f32,
            -0.7034186147f32,
            1.7076147010f32,
        )
    };

    // Approximate max saturation using a polynomial:
    let mut s = k0 + k1 * a + k2 * b + k3 * a * a + k4 * a * b;

    // Do one step Halley's method to get closer
    // this gives an error less than 10e6, except for some blue hues where the dS/dh is close to infinite
    // this should be sufficient for most applications, otherwise do two/three steps

    let k_l = 0.3963377774f32 * a + 0.2158037573f32 * b;
    let k_m = -0.1055613458f32 * a - 0.0638541728f32 * b;
    let k_s = -0.0894841775f32 * a - 1.2914855480f32 * b;

    {
        let l_ = 1.0f32 + s * k_l;
        let m_ = 1.0f32 + s * k_m;
        let s_ = 1.0f32 + s * k_s;

        let l = l_ * l_ * l_;
        let m = m_ * m_ * m_;
        let s_val = s_ * s_ * s_;

        let l_ds = 3.0f32 * k_l * l_ * l_;
        let m_ds = 3.0f32 * k_m * m_ * m_;
        let s_ds = 3.0f32 * k_s * s_ * s_;

        let l_ds2 = 6.0f32 * k_l * k_l * l_;
        let m_ds2 = 6.0f32 * k_m * k_m * m_;
        let s_ds2 = 6.0f32 * k_s * k_s * s_;

        let f = wl * l + wm * m + ws * s_val;
        let f1 = wl * l_ds + wm * m_ds + ws * s_ds;
        let f2 = wl * l_ds2 + wm * m_ds2 + ws * s_ds2;

        s -= f * f1 / (f1 * f1 - 0.5f32 * f * f2);
    }

    s
}

/// Finds L_cusp and C_cusp for a given hue
///
/// The cusp is the point of maximum chroma for a given hue.
///
/// # Parameters
///
/// * `a` - Normalized a component (must satisfy a² + b² = 1)
/// * `b` - Normalized b component (must satisfy a² + b² = 1)
///
/// # Returns
///
/// LC struct containing the lightness and chroma at the cusp
pub fn find_cusp(a: f32, b: f32) -> LC {
    // First, find the maximum saturation (saturation S = C/L)
    let s_cusp = compute_max_saturation(a, b);

    // Convert to linear sRGB to find the first point where at least one of r,g or b >= 1:
    let rgb_at_max = oklab_to_linear_srgb(Lab {
        l: 1.0,
        a: s_cusp * a,
        b: s_cusp * b,
    });
    let l_cusp = cbrtf(1.0f32 / fmaxf(fmaxf(rgb_at_max.r, rgb_at_max.g), rgb_at_max.b));
    let c_cusp = l_cusp * s_cusp;

    LC {
        l: l_cusp,
        c: c_cusp,
    }
}

/// Finds intersection of a line with the sRGB gamut boundary
///
/// The line is defined by:
/// - L = L0 * (1 - t) + t * L1
/// - C = t * C1
///
/// # Parameters
///
/// * `a` - Normalized a component (must satisfy a² + b² = 1)
/// * `b` - Normalized b component (must satisfy a² + b² = 1)
/// * `l1` - Target lightness
/// * `c1` - Target chroma
/// * `l0` - Starting lightness
/// * `cusp` - The cusp point for this hue
///
/// # Returns
///
/// Parameter t at the gamut intersection
pub fn find_gamut_intersection(a: f32, b: f32, l1: f32, c1: f32, l0: f32, cusp: LC) -> f32 {
    // Find the intersection for upper and lower half separately
    if ((l1 - l0) * cusp.c - (cusp.l - l0) * c1) <= 0.0f32 {
        // Lower half
        cusp.c * l0 / (c1 * cusp.l + cusp.c * (l0 - l1))
    } else {
        // Upper half

        // First intersect with triangle
        let mut t_val = cusp.c * (l0 - 1.0f32) / (c1 * (cusp.l - 1.0f32) + cusp.c * (l0 - l1));

        // Then one step Halley's method
        {
            let dl = l1 - l0;
            let dc = c1;

            let k_l = 0.3963377774f32 * a + 0.2158037573f32 * b;
            let k_m = -0.1055613458f32 * a - 0.0638541728f32 * b;
            let k_s = -0.0894841775f32 * a - 1.2914855480f32 * b;

            let l_dt = dl + dc * k_l;
            let m_dt = dl + dc * k_m;
            let s_dt = dl + dc * k_s;

            // If higher accuracy is required, 2 or 3 iterations of the following block can be used:
            {
                let l = l0 * (1.0f32 - t_val) + t_val * l1;
                let c = t_val * c1;

                let l_ = l + c * k_l;
                let m_ = l + c * k_m;
                let s_ = l + c * k_s;

                let l = l_ * l_ * l_;
                let m = m_ * m_ * m_;
                let s = s_ * s_ * s_;

                let ldt = 3.0 * l_dt * l_ * l_;
                let mdt = 3.0 * m_dt * m_ * m_;
                let sdt = 3.0 * s_dt * s_ * s_;

                let ldt2 = 6.0 * l_dt * l_dt * l_;
                let mdt2 = 6.0 * m_dt * m_dt * m_;
                let sdt2 = 6.0 * s_dt * s_dt * s_;

                let r = 4.0767416621f32 * l - 3.3077115913f32 * m + 0.2309699292f32 * s - 1.0;
                let r1 = 4.0767416621f32 * ldt - 3.3077115913f32 * mdt + 0.2309699292f32 * sdt;
                let r2 = 4.0767416621f32 * ldt2 - 3.3077115913f32 * mdt2 + 0.2309699292f32 * sdt2;

                let u_r = r1 / (r1 * r1 - 0.5f32 * r * r2);
                let t_r = -r * u_r;

                let g = -1.2684380046f32 * l + 2.6097574011f32 * m - 0.3413193965f32 * s - 1.0;
                let g1 = -1.2684380046f32 * ldt + 2.6097574011f32 * mdt - 0.3413193965f32 * sdt;
                let g2 = -1.2684380046f32 * ldt2 + 2.6097574011f32 * mdt2 - 0.3413193965f32 * sdt2;

                let u_g = g1 / (g1 * g1 - 0.5f32 * g * g2);
                let t_g = -g * u_g;

                let b_val = -0.0041960863f32 * l - 0.7034186147f32 * m + 1.7076147010f32 * s - 1.0;
                let b1 = -0.0041960863f32 * ldt - 0.7034186147f32 * mdt + 1.7076147010f32 * sdt;
                let b2 = -0.0041960863f32 * ldt2 - 0.7034186147f32 * mdt2 + 1.7076147010f32 * sdt2;

                let u_b = b1 / (b1 * b1 - 0.5f32 * b_val * b2);
                let t_b = -b_val * u_b;

                let t_r = if u_r >= 0.0f32 { t_r } else { f32::MAX };
                let t_g = if u_g >= 0.0f32 { t_g } else { f32::MAX };
                let t_b = if u_b >= 0.0f32 { t_b } else { f32::MAX };

                t_val += fminf(t_r, fminf(t_g, t_b));
            }
        }

        t_val
    }
}

/// Finds intersection with the gamut boundary (simplified version)
///
/// This function computes the cusp internally before calling `find_gamut_intersection`.
///
/// # Parameters
///
/// * `a` - Normalized a component (must satisfy a² + b² = 1)
/// * `b` - Normalized b component (must satisfy a² + b² = 1)
/// * `l1` - Target lightness
/// * `c1` - Target chroma
/// * `l0` - Starting lightness
///
/// # Returns
///
/// Parameter t at the gamut intersection
pub fn find_gamut_intersection_simple(a: f32, b: f32, l1: f32, c1: f32, l0: f32) -> f32 {
    // Find the cusp of the gamut triangle
    let cusp = find_cusp(a, b);

    find_gamut_intersection(a, b, l1, c1, l0, cusp)
}

/// Clips an out-of-gamut color to the sRGB gamut while preserving chroma
///
/// Projects colors outside the sRGB gamut onto the gamut boundary while
/// attempting to preserve the chroma (colorfulness) as much as possible.
///
/// # Parameters
///
/// * `rgb` - Linear RGB color (may be out of gamut)
///
/// # Returns
///
/// Linear RGB color clipped to valid sRGB gamut
pub fn gamut_clip_preserve_chroma(rgb: RGB) -> RGB {
    if rgb.r < 1.0 && rgb.g < 1.0 && rgb.b < 1.0 && rgb.r > 0.0 && rgb.g > 0.0 && rgb.b > 0.0 {
        return rgb;
    }

    let lab = linear_srgb_to_oklab(rgb);

    let l = lab.l;
    let eps = 0.00001f32;
    let c = fmaxf(eps, sqrtf(lab.a * lab.a + lab.b * lab.b));
    let a_ = lab.a / c;
    let b_ = lab.b / c;

    let l0 = clamp(l, 0.0, 1.0);

    let t = find_gamut_intersection_simple(a_, b_, l, c, l0);
    let l_clipped = l0 * (1.0 - t) + t * l;
    let c_clipped = t * c;

    oklab_to_linear_srgb(Lab {
        l: l_clipped,
        a: c_clipped * a_,
        b: c_clipped * b_,
    })
}

/// Clips an out-of-gamut color by projecting to L=0.5
///
/// Projects colors outside the sRGB gamut onto the gamut boundary by
/// finding the intersection with a line toward L=0.5.
///
/// # Parameters
///
/// * `rgb` - Linear RGB color (may be out of gamut)
///
/// # Returns
///
/// Linear RGB color clipped to valid sRGB gamut
pub fn gamut_clip_project_to_0_5(rgb: RGB) -> RGB {
    if rgb.r < 1.0 && rgb.g < 1.0 && rgb.b < 1.0 && rgb.r > 0.0 && rgb.g > 0.0 && rgb.b > 0.0 {
        return rgb;
    }

    let lab = linear_srgb_to_oklab(rgb);

    let l = lab.l;
    let eps = 0.00001f32;
    let c = fmaxf(eps, sqrtf(lab.a * lab.a + lab.b * lab.b));
    let a_ = lab.a / c;
    let b_ = lab.b / c;

    let l0 = 0.5;

    let t = find_gamut_intersection_simple(a_, b_, l, c, l0);
    let l_clipped = l0 * (1.0 - t) + t * l;
    let c_clipped = t * c;

    oklab_to_linear_srgb(Lab {
        l: l_clipped,
        a: c_clipped * a_,
        b: c_clipped * b_,
    })
}

/// Clips an out-of-gamut color by projecting to L_cusp
///
/// Projects colors outside the sRGB gamut onto the gamut boundary by
/// finding the intersection with a line toward the cusp lightness for the hue.
///
/// # Parameters
///
/// * `rgb` - Linear RGB color (may be out of gamut)
///
/// # Returns
///
/// Linear RGB color clipped to valid sRGB gamut
pub fn gamut_clip_project_to_l_cusp(rgb: RGB) -> RGB {
    if rgb.r < 1.0 && rgb.g < 1.0 && rgb.b < 1.0 && rgb.r > 0.0 && rgb.g > 0.0 && rgb.b > 0.0 {
        return rgb;
    }

    let lab = linear_srgb_to_oklab(rgb);

    let l = lab.l;
    let eps = 0.00001f32;
    let c = fmaxf(eps, sqrtf(lab.a * lab.a + lab.b * lab.b));
    let a_ = lab.a / c;
    let b_ = lab.b / c;

    // The cusp is computed here and in find_gamut_intersection, an optimized solution would only compute it once.
    let cusp = find_cusp(a_, b_);

    let l0 = cusp.l;

    let t = find_gamut_intersection(a_, b_, l, c, l0, cusp);

    let l_clipped = l0 * (1.0 - t) + t * l;
    let c_clipped = t * c;

    oklab_to_linear_srgb(Lab {
        l: l_clipped,
        a: c_clipped * a_,
        b: c_clipped * b_,
    })
}

/// Clips an out-of-gamut color using adaptive L0 around 0.5
///
/// Uses an adaptive algorithm to determine the projection target lightness
/// based on the alpha parameter and chroma.
///
/// # Parameters
///
/// * `rgb` - Linear RGB color (may be out of gamut)
/// * `alpha` - Alpha parameter controlling the adaptation strength
///
/// # Returns
///
/// Linear RGB color clipped to valid sRGB gamut
pub fn gamut_clip_adaptive_l0_0_5(rgb: RGB, alpha: f32) -> RGB {
    if rgb.r < 1.0 && rgb.g < 1.0 && rgb.b < 1.0 && rgb.r > 0.0 && rgb.g > 0.0 && rgb.b > 0.0 {
        return rgb;
    }

    let lab = linear_srgb_to_oklab(rgb);

    let l = lab.l;
    let eps = 0.00001f32;
    let c = fmaxf(eps, sqrtf(lab.a * lab.a + lab.b * lab.b));
    let a_ = lab.a / c;
    let b_ = lab.b / c;

    let ld = l - 0.5f32;
    let e1 = 0.5f32 + libm::fabsf(ld) + alpha * c;
    let l0 = 0.5f32 * (1.0f32 + sgn(ld) * (e1 - sqrtf(e1 * e1 - 2.0f32 * libm::fabsf(ld))));

    let t = find_gamut_intersection_simple(a_, b_, l, c, l0);
    let l_clipped = l0 * (1.0f32 - t) + t * l;
    let c_clipped = t * c;

    oklab_to_linear_srgb(Lab {
        l: l_clipped,
        a: c_clipped * a_,
        b: c_clipped * b_,
    })
}

/// Clips an out-of-gamut color using adaptive L0 around L_cusp
///
/// Uses an adaptive algorithm to determine the projection target lightness
/// based on the alpha parameter, chroma, and cusp lightness.
///
/// # Parameters
///
/// * `rgb` - Linear RGB color (may be out of gamut)
/// * `alpha` - Alpha parameter controlling the adaptation strength
///
/// # Returns
///
/// Linear RGB color clipped to valid sRGB gamut
pub fn gamut_clip_adaptive_l0_l_cusp(rgb: RGB, alpha: f32) -> RGB {
    if rgb.r < 1.0 && rgb.g < 1.0 && rgb.b < 1.0 && rgb.r > 0.0 && rgb.g > 0.0 && rgb.b > 0.0 {
        return rgb;
    }

    let lab = linear_srgb_to_oklab(rgb);

    let l = lab.l;
    let eps = 0.00001f32;
    let c = fmaxf(eps, sqrtf(lab.a * lab.a + lab.b * lab.b));
    let a_ = lab.a / c;
    let b_ = lab.b / c;

    // The cusp is computed here and in find_gamut_intersection, an optimized solution would only compute it once.
    let cusp = find_cusp(a_, b_);

    let ld = l - cusp.l;
    let k = 2.0f32 * if ld > 0.0 { 1.0f32 - cusp.l } else { cusp.l };

    let e1 = 0.5f32 * k + libm::fabsf(ld) + alpha * c / k;
    let l0 = cusp.l + 0.5f32 * (sgn(ld) * (e1 - sqrtf(e1 * e1 - 2.0f32 * k * libm::fabsf(ld))));

    let t = find_gamut_intersection(a_, b_, l, c, l0, cusp);
    let l_clipped = l0 * (1.0f32 - t) + t * l;
    let c_clipped = t * c;

    oklab_to_linear_srgb(Lab {
        l: l_clipped,
        a: c_clipped * a_,
        b: c_clipped * b_,
    })
}

#[inline]
fn toe(x: f32) -> f32 {
    const K_1: f32 = 0.206f32;
    const K_2: f32 = 0.03f32;
    const K_3: f32 = (1.0f32 + K_1) / (1.0f32 + K_2);
    0.5f32 * (K_3 * x - K_1 + sqrtf((K_3 * x - K_1) * (K_3 * x - K_1) + 4.0 * K_2 * K_3 * x))
}

#[inline]
fn toe_inv(x: f32) -> f32 {
    const K_1: f32 = 0.206f32;
    const K_2: f32 = 0.03f32;
    const K_3: f32 = (1.0f32 + K_1) / (1.0f32 + K_2);
    (x * x + K_1 * x) / (K_3 * (x + K_2))
}

/// Converts LC (lightness-chroma) to ST representation
///
/// # Parameters
///
/// * `cusp` - LC coordinates at the cusp
///
/// # Returns
///
/// ST representation where S = C/L and T = C/(1-L)
pub fn to_st(cusp: LC) -> ST {
    let l = cusp.l;
    let c = cusp.c;
    ST {
        s: c / l,
        t: c / (1.0 - l),
    }
}

/// Returns a smooth approximation of the location of the cusp
///
/// This polynomial was created by an optimization process.
/// It has been designed so that S_mid < S_max and T_mid < T_max.
///
/// # Parameters
///
/// * `a_` - Normalized a component
/// * `b_` - Normalized b component
///
/// # Returns
///
/// ST values at the mid point
pub fn get_st_mid(a_: f32, b_: f32) -> ST {
    let s = 0.11516993f32
        + 1.0f32
            / (7.44778970f32
                + 4.15901240f32 * b_
                + a_ * (-2.19557347f32
                    + 1.75198401f32 * b_
                    + a_ * (-2.13704948f32 - 10.02301043f32 * b_
                        + a_ * (-4.24894561f32 + 5.38770819f32 * b_ + 4.69891013f32 * a_))));

    let t = 0.11239642f32
        + 1.0f32
            / (1.61320320f32 - 0.68124379f32 * b_
                + a_ * (0.40370612f32
                    + 0.90148123f32 * b_
                    + a_ * (-0.27087943f32
                        + 0.61223990f32 * b_
                        + a_ * (0.00299215f32 - 0.45399568f32 * b_ - 0.14661872f32 * a_))));

    ST { s, t }
}

/// Gets chroma boundaries (C_0, C_mid, C_max) for a given lightness and hue
///
/// # Parameters
///
/// * `l` - Lightness value
/// * `a_` - Normalized a component
/// * `b_` - Normalized b component
///
/// # Returns
///
/// Cs struct containing minimum, mid, and maximum chroma values
pub fn get_cs(l: f32, a_: f32, b_: f32) -> Cs {
    let cusp = find_cusp(a_, b_);

    let c_max = find_gamut_intersection(a_, b_, l, 1.0, l, cusp);
    let st_max = to_st(cusp);

    // Scale factor to compensate for the curved part of gamut shape:
    let k = c_max / fminf(l * st_max.s, (1.0 - l) * st_max.t);

    let c_mid = {
        let st_mid = get_st_mid(a_, b_);

        // Use a soft minimum function, instead of a sharp triangle shape to get a smooth value for chroma.
        let c_a = l * st_mid.s;
        let c_b = (1.0f32 - l) * st_mid.t;
        0.9f32
            * k
            * sqrtf(sqrtf(
                1.0f32 / (1.0f32 / (c_a * c_a * c_a * c_a) + 1.0f32 / (c_b * c_b * c_b * c_b)),
            ))
    };

    let c_0 = {
        // for C_0, the shape is independent of hue, so ST are constant. Values picked to roughly be the average values of ST.
        let c_a = l * 0.4f32;
        let c_b = (1.0f32 - l) * 0.8f32;

        // Use a soft minimum function, instead of a sharp triangle shape to get a smooth value for chroma.
        sqrtf(1.0f32 / (1.0f32 / (c_a * c_a) + 1.0f32 / (c_b * c_b)))
    };

    Cs { c_0, c_mid, c_max }
}

/// Converts OKHSL color to sRGB
///
/// # Parameters
///
/// * `hsl` - OKHSL color to convert
///
/// # Returns
///
/// sRGB color with gamma correction applied
///
/// # Example
///
/// ```
/// use okhsl_embedded::{HSL, okhsl_to_srgb};
///
/// let hsl = HSL { h: 0.5, s: 0.8, l: 0.6 };
/// let rgb = okhsl_to_srgb(hsl);
/// ```
pub fn okhsl_to_srgb(hsl: HSL) -> RGB {
    let h = hsl.h;
    let s = hsl.s;
    let l = hsl.l;

    if l == 1.0f32 {
        return RGB {
            r: 1.0f32,
            g: 1.0f32,
            b: 1.0f32,
        };
    } else if l == 0.0f32 {
        return RGB {
            r: 0.0f32,
            g: 0.0f32,
            b: 0.0f32,
        };
    }

    let a_ = cosf(2.0f32 * PI * h);
    let b_ = sinf(2.0f32 * PI * h);
    let l_val = toe_inv(l);

    let cs = get_cs(l_val, a_, b_);
    let c_0 = cs.c_0;
    let c_mid = cs.c_mid;
    let c_max = cs.c_max;

    let mid = 0.8f32;
    let mid_inv = 1.25f32;

    let c = if s < mid {
        let t = mid_inv * s;

        let k_1 = mid * c_0;
        let k_2 = 1.0f32 - k_1 / c_mid;

        t * k_1 / (1.0f32 - k_2 * t)
    } else {
        let t = (s - mid) / (1.0 - mid);

        let k_0 = c_mid;
        let k_1 = (1.0f32 - mid) * c_mid * c_mid * mid_inv * mid_inv / c_0;
        let k_2 = 1.0f32 - k_1 / (c_max - c_mid);

        k_0 + t * k_1 / (1.0f32 - k_2 * t)
    };

    let rgb = oklab_to_linear_srgb(Lab {
        l: l_val,
        a: c * a_,
        b: c * b_,
    });
    RGB {
        r: srgb_transfer_function(rgb.r),
        g: srgb_transfer_function(rgb.g),
        b: srgb_transfer_function(rgb.b),
    }
}

/// Converts sRGB color to OKHSL
///
/// # Parameters
///
/// * `rgb` - sRGB color to convert
///
/// # Returns
///
/// OKHSL color representation
///
/// # Example
///
/// ```
/// use okhsl_embedded::{RGB, srgb_to_okhsl};
///
/// let rgb = RGB { r: 0.5, g: 0.8, b: 0.3 };
/// let hsl = srgb_to_okhsl(rgb);
/// ```
pub fn srgb_to_okhsl(rgb: RGB) -> HSL {
    let lab = linear_srgb_to_oklab(RGB {
        r: srgb_transfer_function_inv(rgb.r),
        g: srgb_transfer_function_inv(rgb.g),
        b: srgb_transfer_function_inv(rgb.b),
    });

    let c = sqrtf(lab.a * lab.a + lab.b * lab.b);
    let a_ = lab.a / c;
    let b_ = lab.b / c;

    let l = lab.l;
    let h = 0.5f32 + 0.5f32 * atan2f(-lab.b, -lab.a) / PI;

    let cs = get_cs(l, a_, b_);
    let c_0 = cs.c_0;
    let c_mid = cs.c_mid;
    let c_max = cs.c_max;

    // Inverse of the interpolation in okhsl_to_srgb:

    let mid = 0.8f32;
    let mid_inv = 1.25f32;

    let s = if c < c_mid {
        let k_1 = mid * c_0;
        let k_2 = 1.0f32 - k_1 / c_mid;

        let t = c / (k_1 + k_2 * c);
        t * mid
    } else {
        let k_0 = c_mid;
        let k_1 = (1.0f32 - mid) * c_mid * c_mid * mid_inv * mid_inv / c_0;
        let k_2 = 1.0f32 - k_1 / (c_max - c_mid);

        let t = (c - k_0) / (k_1 + k_2 * (c - k_0));
        mid + (1.0f32 - mid) * t
    };

    let l = toe(l);
    HSL { h, s, l }
}

/// Converts OKHSV color to sRGB
///
/// # Parameters
///
/// * `hsv` - OKHSV color to convert
///
/// # Returns
///
/// sRGB color with gamma correction applied
///
/// # Example
///
/// ```
/// use okhsl_embedded::{HSV, okhsv_to_srgb};
///
/// let hsv = HSV { h: 0.5, s: 0.8, v: 0.9 };
/// let rgb = okhsv_to_srgb(hsv);
/// ```
pub fn okhsv_to_srgb(hsv: HSV) -> RGB {
    let h = hsv.h;
    let s = hsv.s;
    let v = hsv.v;

    let a_ = cosf(2.0f32 * PI * h);
    let b_ = sinf(2.0f32 * PI * h);

    let cusp = find_cusp(a_, b_);
    let st_max = to_st(cusp);
    let s_max = st_max.s;
    let t_max = st_max.t;
    let s_0 = 0.5f32;
    let k = 1.0 - s_0 / s_max;

    // first we compute L and V as if the gamut is a perfect triangle:

    // L, C when v==1:
    let l_v = 1.0 - s * s_0 / (s_0 + t_max - t_max * k * s);
    let c_v = s * t_max * s_0 / (s_0 + t_max - t_max * k * s);

    let mut l = v * l_v;
    let mut c = v * c_v;

    // then we compensate for both toe and the curved top part of the triangle:
    let l_vt = toe_inv(l_v);
    let c_vt = c_v * l_vt / l_v;

    let l_new = toe_inv(l);
    c *= l_new / l;
    l = l_new;

    let rgb_scale = oklab_to_linear_srgb(Lab {
        l: l_vt,
        a: a_ * c_vt,
        b: b_ * c_vt,
    });
    let scale_l =
        cbrtf(1.0f32 / fmaxf(fmaxf(rgb_scale.r, rgb_scale.g), fmaxf(rgb_scale.b, 0.0f32)));

    l *= scale_l;
    c *= scale_l;

    let rgb = oklab_to_linear_srgb(Lab {
        l,
        a: c * a_,
        b: c * b_,
    });
    RGB {
        r: srgb_transfer_function(rgb.r),
        g: srgb_transfer_function(rgb.g),
        b: srgb_transfer_function(rgb.b),
    }
}

/// Converts sRGB color to OKHSV
///
/// # Parameters
///
/// * `rgb` - sRGB color to convert
///
/// # Returns
///
/// OKHSV color representation
///
/// # Example
///
/// ```
/// use okhsl_embedded::{RGB, srgb_to_okhsv};
///
/// let rgb = RGB { r: 0.5, g: 0.8, b: 0.3 };
/// let hsv = srgb_to_okhsv(rgb);
/// ```
pub fn srgb_to_okhsv(rgb: RGB) -> HSV {
    let lab = linear_srgb_to_oklab(RGB {
        r: srgb_transfer_function_inv(rgb.r),
        g: srgb_transfer_function_inv(rgb.g),
        b: srgb_transfer_function_inv(rgb.b),
    });

    let c = sqrtf(lab.a * lab.a + lab.b * lab.b);
    let a_ = lab.a / c;
    let b_ = lab.b / c;

    let mut l = lab.l;
    let h = 0.5f32 + 0.5f32 * atan2f(-lab.b, -lab.a) / PI;

    let cusp = find_cusp(a_, b_);
    let st_max = to_st(cusp);
    let s_max = st_max.s;
    let t_max = st_max.t;
    let s_0 = 0.5f32;
    let k = 1.0 - s_0 / s_max;

    // first we find L_v, C_v, L_vt and C_vt

    let t = t_max / (c + l * t_max);
    let l_v = t * l;
    let c_v = t * c;

    let l_vt = toe_inv(l_v);
    let c_vt = c_v * l_vt / l_v;

    // we can then use these to invert the step that compensates for the toe and the curved top part of the triangle:
    let rgb_scale = oklab_to_linear_srgb(Lab {
        l: l_vt,
        a: a_ * c_vt,
        b: b_ * c_vt,
    });
    let scale_l =
        cbrtf(1.0f32 / fmaxf(fmaxf(rgb_scale.r, rgb_scale.g), fmaxf(rgb_scale.b, 0.0f32)));

    l /= scale_l;

    l = toe(l);

    // we can now compute v and s:

    let v = l / l_v;
    let s = (s_0 + t_max) * c_v / ((t_max * s_0) + t_max * k * c_v);

    HSV { h, s, v }
}

// Gamma-corrected RGB conversion functions

/// Converts OKHSL to gamma-corrected RGB with custom gamma value
///
/// # Parameters
///
/// * `hsl` - OKHSL color to convert
/// * `gamma` - Gamma value (typical: 2.2 for displays, 2.4 for sRGB, 1.8 for Mac)
///
/// # Returns
///
/// Gamma-corrected RGB color
pub fn okhsl_to_gamma_rgb(hsl: HSL, gamma: f32) -> RGB {
    let h = hsl.h;
    let s = hsl.s;
    let l = hsl.l;

    if l == 1.0f32 {
        return RGB {
            r: 1.0f32,
            g: 1.0f32,
            b: 1.0f32,
        };
    } else if l == 0.0f32 {
        return RGB {
            r: 0.0f32,
            g: 0.0f32,
            b: 0.0f32,
        };
    }

    let a_ = cosf(2.0f32 * PI * h);
    let b_ = sinf(2.0f32 * PI * h);
    let l_val = toe_inv(l);

    let cs = get_cs(l_val, a_, b_);
    let c_0 = cs.c_0;
    let c_mid = cs.c_mid;
    let c_max = cs.c_max;

    let mid = 0.8f32;
    let mid_inv = 1.25f32;

    let c = if s < mid {
        let t = mid_inv * s;

        let k_1 = mid * c_0;
        let k_2 = 1.0f32 - k_1 / c_mid;

        t * k_1 / (1.0f32 - k_2 * t)
    } else {
        let t = (s - mid) / (1.0 - mid);

        let k_0 = c_mid;
        let k_1 = (1.0f32 - mid) * c_mid * c_mid * mid_inv * mid_inv / c_0;
        let k_2 = 1.0f32 - k_1 / (c_max - c_mid);

        k_0 + t * k_1 / (1.0f32 - k_2 * t)
    };

    let rgb = oklab_to_linear_srgb(Lab {
        l: l_val,
        a: c * a_,
        b: c * b_,
    });
    RGB {
        r: gamma_transfer_function(rgb.r, gamma),
        g: gamma_transfer_function(rgb.g, gamma),
        b: gamma_transfer_function(rgb.b, gamma),
    }
}

/// Converts gamma-corrected RGB to OKHSL with custom gamma value
///
/// # Parameters
///
/// * `rgb` - Gamma-corrected RGB color to convert
/// * `gamma` - Gamma value (typical: 2.2 for displays, 2.4 for sRGB, 1.8 for Mac)
///
/// # Returns
///
/// OKHSL color representation
pub fn gamma_rgb_to_okhsl(rgb: RGB, gamma: f32) -> HSL {
    let lab = linear_srgb_to_oklab(RGB {
        r: gamma_transfer_function_inv(rgb.r, gamma),
        g: gamma_transfer_function_inv(rgb.g, gamma),
        b: gamma_transfer_function_inv(rgb.b, gamma),
    });

    let c = sqrtf(lab.a * lab.a + lab.b * lab.b);
    let a_ = lab.a / c;
    let b_ = lab.b / c;

    let l = lab.l;
    let h = 0.5f32 + 0.5f32 * atan2f(-lab.b, -lab.a) / PI;

    let cs = get_cs(l, a_, b_);
    let c_0 = cs.c_0;
    let c_mid = cs.c_mid;
    let c_max = cs.c_max;

    // Inverse of the interpolation in okhsl_to_gamma_rgb:

    let mid = 0.8f32;
    let mid_inv = 1.25f32;

    let s = if c < c_mid {
        let k_1 = mid * c_0;
        let k_2 = 1.0f32 - k_1 / c_mid;

        let t = c / (k_1 + k_2 * c);
        t * mid
    } else {
        let k_0 = c_mid;
        let k_1 = (1.0f32 - mid) * c_mid * c_mid * mid_inv * mid_inv / c_0;
        let k_2 = 1.0f32 - k_1 / (c_max - c_mid);

        let t = (c - k_0) / (k_1 + k_2 * (c - k_0));
        mid + (1.0f32 - mid) * t
    };

    let l = toe(l);
    HSL { h, s, l }
}

/// Converts OKHSV to gamma-corrected RGB with custom gamma value
///
/// # Parameters
///
/// * `hsv` - OKHSV color to convert
/// * `gamma` - Gamma value (typical: 2.2 for displays, 2.4 for sRGB, 1.8 for Mac)
///
/// # Returns
///
/// Gamma-corrected RGB color
pub fn okhsv_to_gamma_rgb(hsv: HSV, gamma: f32) -> RGB {
    let h = hsv.h;
    let s = hsv.s;
    let v = hsv.v;

    let a_ = cosf(2.0f32 * PI * h);
    let b_ = sinf(2.0f32 * PI * h);

    let cusp = find_cusp(a_, b_);
    let st_max = to_st(cusp);
    let s_max = st_max.s;
    let t_max = st_max.t;
    let s_0 = 0.5f32;
    let k = 1.0 - s_0 / s_max;

    // first we compute L and V as if the gamut is a perfect triangle:

    // L, C when v==1:
    let l_v = 1.0 - s * s_0 / (s_0 + t_max - t_max * k * s);
    let c_v = s * t_max * s_0 / (s_0 + t_max - t_max * k * s);

    let mut l = v * l_v;
    let mut c = v * c_v;

    // then we compensate for both toe and the curved top part of the triangle:
    let l_vt = toe_inv(l_v);
    let c_vt = c_v * l_vt / l_v;

    let l_new = toe_inv(l);
    c *= l_new / l;
    l = l_new;

    let rgb_scale = oklab_to_linear_srgb(Lab {
        l: l_vt,
        a: a_ * c_vt,
        b: b_ * c_vt,
    });
    let scale_l =
        cbrtf(1.0f32 / fmaxf(fmaxf(rgb_scale.r, rgb_scale.g), fmaxf(rgb_scale.b, 0.0f32)));

    l *= scale_l;
    c *= scale_l;

    let rgb = oklab_to_linear_srgb(Lab {
        l,
        a: c * a_,
        b: c * b_,
    });
    RGB {
        r: gamma_transfer_function(rgb.r, gamma),
        g: gamma_transfer_function(rgb.g, gamma),
        b: gamma_transfer_function(rgb.b, gamma),
    }
}

/// Converts gamma-corrected RGB to OKHSV with custom gamma value
///
/// # Parameters
///
/// * `rgb` - Gamma-corrected RGB color to convert
/// * `gamma` - Gamma value (typical: 2.2 for displays, 2.4 for sRGB, 1.8 for Mac)
///
/// # Returns
///
/// OKHSV color representation
pub fn gamma_rgb_to_okhsv(rgb: RGB, gamma: f32) -> HSV {
    let lab = linear_srgb_to_oklab(RGB {
        r: gamma_transfer_function_inv(rgb.r, gamma),
        g: gamma_transfer_function_inv(rgb.g, gamma),
        b: gamma_transfer_function_inv(rgb.b, gamma),
    });

    let c = sqrtf(lab.a * lab.a + lab.b * lab.b);
    let a_ = lab.a / c;
    let b_ = lab.b / c;

    let mut l = lab.l;
    let h = 0.5f32 + 0.5f32 * atan2f(-lab.b, -lab.a) / PI;

    let cusp = find_cusp(a_, b_);
    let st_max = to_st(cusp);
    let s_max = st_max.s;
    let t_max = st_max.t;
    let s_0 = 0.5f32;
    let k = 1.0 - s_0 / s_max;

    // first we find L_v, C_v, L_vt and C_vt

    let t = t_max / (c + l * t_max);
    let l_v = t * l;
    let c_v = t * c;

    let l_vt = toe_inv(l_v);
    let c_vt = c_v * l_vt / l_v;

    // we can then use these to invert the step that compensates for the toe and the curved top part of the triangle:
    let rgb_scale = oklab_to_linear_srgb(Lab {
        l: l_vt,
        a: a_ * c_vt,
        b: b_ * c_vt,
    });
    let scale_l =
        cbrtf(1.0f32 / fmaxf(fmaxf(rgb_scale.r, rgb_scale.g), fmaxf(rgb_scale.b, 0.0f32)));

    l /= scale_l;

    l = toe(l);

    // we can now compute v and s:

    let v = l / l_v;
    let s = (s_0 + t_max) * c_v / ((t_max * s_0) + t_max * k * c_v);

    HSV { h, s, v }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    fn rgb_approx_eq(a: RGB, b: RGB, epsilon: f32) -> bool {
        approx_eq(a.r, b.r, epsilon) && approx_eq(a.g, b.g, epsilon) && approx_eq(a.b, b.b, epsilon)
    }

    fn hsl_approx_eq(a: HSL, b: HSL, epsilon: f32) -> bool {
        approx_eq(a.h, b.h, epsilon) && approx_eq(a.s, b.s, epsilon) && approx_eq(a.l, b.l, epsilon)
    }

    fn hsv_approx_eq(a: HSV, b: HSV, epsilon: f32) -> bool {
        approx_eq(a.h, b.h, epsilon) && approx_eq(a.s, b.s, epsilon) && approx_eq(a.v, b.v, epsilon)
    }

    #[allow(dead_code)]
    fn lab_approx_eq(a: Lab, b: Lab, epsilon: f32) -> bool {
        approx_eq(a.l, b.l, epsilon) && approx_eq(a.a, b.a, epsilon) && approx_eq(a.b, b.b, epsilon)
    }

    #[test]
    fn test_clamp() {
        assert_eq!(clamp(0.5, 0.0, 1.0), 0.5);
        assert_eq!(clamp(-0.5, 0.0, 1.0), 0.0);
        assert_eq!(clamp(1.5, 0.0, 1.0), 1.0);
        assert_eq!(clamp(0.3, 0.2, 0.8), 0.3);
    }

    #[test]
    fn test_sgn() {
        assert_eq!(sgn(5.0), 1.0);
        assert_eq!(sgn(-5.0), -1.0);
        assert_eq!(sgn(0.0), 0.0);
    }

    #[test]
    fn test_toe_toe_inv_roundtrip() {
        let values = [0.0, 0.25, 0.5, 0.75, 1.0];
        for &val in &values {
            let toe_val = toe(val);
            let recovered = toe_inv(toe_val);
            assert!(
                approx_eq(recovered, val, EPSILON),
                "toe/toe_inv roundtrip failed for {}: got {}",
                val,
                recovered
            );
        }
    }

    #[test]
    fn test_srgb_transfer_roundtrip() {
        let values = [0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 1.0];
        for &val in &values {
            let gamma = srgb_transfer_function(val);
            let linear = srgb_transfer_function_inv(gamma);
            assert!(
                approx_eq(linear, val, EPSILON),
                "sRGB transfer roundtrip failed for {}: got {}",
                val,
                linear
            );
        }
    }

    #[test]
    fn test_gamma_transfer_roundtrip() {
        let values = [0.0, 0.1, 0.5, 0.9, 1.0];
        let gammas = [1.0, 1.8, 2.2, 2.4];

        for &gamma in &gammas {
            for &val in &values {
                let corrected = gamma_transfer_function(val, gamma);
                let linear = gamma_transfer_function_inv(corrected, gamma);
                assert!(
                    approx_eq(linear, val, EPSILON),
                    "Gamma {} transfer roundtrip failed for {}: got {}",
                    gamma,
                    val,
                    linear
                );
            }
        }
    }

    #[test]
    fn test_linear_srgb_oklab_roundtrip() {
        let test_colors = [
            RGB {
                r: 1.0,
                g: 0.0,
                b: 0.0,
            }, // Red
            RGB {
                r: 0.0,
                g: 1.0,
                b: 0.0,
            }, // Green
            RGB {
                r: 0.0,
                g: 0.0,
                b: 1.0,
            }, // Blue
            RGB {
                r: 1.0,
                g: 1.0,
                b: 1.0,
            }, // White
            RGB {
                r: 0.5,
                g: 0.5,
                b: 0.5,
            }, // Gray
            RGB {
                r: 0.3,
                g: 0.7,
                b: 0.2,
            }, // Random
        ];

        for color in &test_colors {
            let lab = linear_srgb_to_oklab(*color);
            let recovered = oklab_to_linear_srgb(lab);
            assert!(
                rgb_approx_eq(*color, recovered, 1e-4),
                "RGB->Oklab->RGB roundtrip failed for {:?}: got {:?}",
                color,
                recovered
            );
        }
    }

    #[test]
    fn test_okhsl_srgb_black() {
        let black_hsl = HSL {
            h: 0.0,
            s: 0.0,
            l: 0.0,
        };
        let rgb = okhsl_to_srgb(black_hsl);
        assert!(
            rgb_approx_eq(
                rgb,
                RGB {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0
                },
                EPSILON
            ),
            "Black OKHSL should convert to black sRGB"
        );
    }

    #[test]
    fn test_okhsl_srgb_white() {
        let white_hsl = HSL {
            h: 0.0,
            s: 0.0,
            l: 1.0,
        };
        let rgb = okhsl_to_srgb(white_hsl);
        assert!(
            rgb_approx_eq(
                rgb,
                RGB {
                    r: 1.0,
                    g: 1.0,
                    b: 1.0
                },
                EPSILON
            ),
            "White OKHSL should convert to white sRGB"
        );
    }

    #[test]
    fn test_okhsl_srgb_roundtrip() {
        let test_hsl = [
            HSL {
                h: 0.0,
                s: 1.0,
                l: 0.5,
            }, // Red
            HSL {
                h: 0.333,
                s: 1.0,
                l: 0.5,
            }, // Green
            HSL {
                h: 0.667,
                s: 1.0,
                l: 0.5,
            }, // Blue
            HSL {
                h: 0.5,
                s: 0.5,
                l: 0.5,
            }, // Cyan-ish
            HSL {
                h: 0.1,
                s: 0.8,
                l: 0.6,
            }, // Orange-ish
        ];

        for hsl in &test_hsl {
            let rgb = okhsl_to_srgb(*hsl);
            let recovered = srgb_to_okhsl(rgb);
            assert!(
                hsl_approx_eq(*hsl, recovered, 1e-3),
                "OKHSL->sRGB->OKHSL roundtrip failed for {:?}: got {:?}",
                hsl,
                recovered
            );
        }
    }

    #[test]
    fn test_okhsv_srgb_roundtrip() {
        let test_hsv = [
            HSV {
                h: 0.0,
                s: 1.0,
                v: 1.0,
            }, // Red
            HSV {
                h: 0.333,
                s: 1.0,
                v: 1.0,
            }, // Green
            HSV {
                h: 0.667,
                s: 1.0,
                v: 1.0,
            }, // Blue
            HSV {
                h: 0.5,
                s: 0.5,
                v: 0.8,
            }, // Cyan-ish
            HSV {
                h: 0.1,
                s: 0.8,
                v: 0.6,
            }, // Orange-ish
        ];

        for hsv in &test_hsv {
            let rgb = okhsv_to_srgb(*hsv);
            let recovered = srgb_to_okhsv(rgb);
            assert!(
                hsv_approx_eq(*hsv, recovered, 1e-3),
                "OKHSV->sRGB->OKHSV roundtrip failed for {:?}: got {:?}",
                hsv,
                recovered
            );
        }
    }

    #[test]
    fn test_gamma_rgb_conversions() {
        let linear = RGB {
            r: 0.5,
            g: 0.3,
            b: 0.8,
        };
        let gamma = 2.2;

        let gamma_corrected = linear_rgb_to_gamma_rgb(linear, gamma);
        let recovered = gamma_rgb_to_linear_rgb(gamma_corrected, gamma);

        assert!(
            rgb_approx_eq(linear, recovered, EPSILON),
            "Linear->Gamma->Linear roundtrip failed"
        );
    }

    #[test]
    fn test_okhsl_gamma_rgb_roundtrip() {
        let hsl = HSL {
            h: 0.5,
            s: 0.7,
            l: 0.6,
        };
        let gamma = 2.2;

        let rgb = okhsl_to_gamma_rgb(hsl, gamma);
        let recovered = gamma_rgb_to_okhsl(rgb, gamma);

        assert!(
            hsl_approx_eq(hsl, recovered, 1e-3),
            "OKHSL->Gamma RGB->OKHSL roundtrip failed for gamma {}",
            gamma
        );
    }

    #[test]
    fn test_okhsv_gamma_rgb_roundtrip() {
        let hsv = HSV {
            h: 0.3,
            s: 0.8,
            v: 0.7,
        };
        let gamma = 2.4;

        let rgb = okhsv_to_gamma_rgb(hsv, gamma);
        let recovered = gamma_rgb_to_okhsv(rgb, gamma);

        assert!(
            hsv_approx_eq(hsv, recovered, 1e-3),
            "OKHSV->Gamma RGB->OKHSV roundtrip failed for gamma {}",
            gamma
        );
    }

    #[test]
    fn test_gamut_clip_preserve_chroma_in_gamut() {
        let in_gamut = RGB {
            r: 0.5,
            g: 0.3,
            b: 0.8,
        };
        let clipped = gamut_clip_preserve_chroma(in_gamut);
        assert!(
            rgb_approx_eq(in_gamut, clipped, EPSILON),
            "In-gamut color should not be clipped"
        );
    }

    #[test]
    fn test_gamut_clip_out_of_gamut() {
        let out_of_gamut = RGB {
            r: 1.5,
            g: -0.2,
            b: 0.5,
        };
        let clipped = gamut_clip_preserve_chroma(out_of_gamut);

        // Should be in valid range
        assert!(clipped.r >= 0.0 && clipped.r <= 1.0);
        assert!(clipped.g >= 0.0 && clipped.g <= 1.0);
        assert!(clipped.b >= 0.0 && clipped.b <= 1.0);
    }

    #[test]
    fn test_linear_to_srgb_conversion() {
        let linear = RGB {
            r: 0.5,
            g: 0.3,
            b: 0.8,
        };
        let srgb = linear_rgb_to_srgb(linear);
        let recovered = srgb_to_linear_rgb(srgb);

        assert!(
            rgb_approx_eq(linear, recovered, EPSILON),
            "linear_rgb_to_srgb/srgb_to_linear_rgb roundtrip failed"
        );
    }

    #[test]
    fn test_compute_max_saturation() {
        // Test for red hue (a=1, b=0)
        let s_red = compute_max_saturation(1.0, 0.0);
        assert!(s_red > 0.0, "Max saturation for red should be positive");

        // Test for green hue
        let s_green = compute_max_saturation(-0.5, 0.866);
        assert!(s_green > 0.0, "Max saturation for green should be positive");

        // Test for blue hue
        let s_blue = compute_max_saturation(-0.5, -0.866);
        assert!(s_blue > 0.0, "Max saturation for blue should be positive");
    }

    #[test]
    fn test_find_cusp() {
        // Test for red hue
        let cusp = find_cusp(1.0, 0.0);
        assert!(cusp.l > 0.0 && cusp.l < 1.0, "L_cusp should be in (0, 1)");
        assert!(cusp.c > 0.0, "C_cusp should be positive");

        // Test ST conversion
        let st = to_st(cusp);
        assert!(st.s > 0.0, "S should be positive");
        assert!(st.t > 0.0, "T should be positive");
    }

    #[test]
    fn test_get_st_mid() {
        let st_mid = get_st_mid(1.0, 0.0);
        assert!(st_mid.s > 0.0, "S_mid should be positive");
        assert!(st_mid.t > 0.0, "T_mid should be positive");
    }

    #[test]
    fn test_get_cs() {
        let cs = get_cs(0.5, 1.0, 0.0);
        assert!(cs.c_0 > 0.0, "C_0 should be positive");
        assert!(cs.c_mid > 0.0, "C_mid should be positive");
        assert!(cs.c_max > 0.0, "C_max should be positive");
        // C_mid should generally be <= C_max (max chroma in gamut)
        assert!(cs.c_mid <= cs.c_max, "C_mid should be <= C_max");
    }

    #[test]
    fn test_known_color_red() {
        // Pure red in sRGB should have hue near 0
        let red = RGB {
            r: 1.0,
            g: 0.0,
            b: 0.0,
        };
        let hsl = srgb_to_okhsl(red);
        assert!(hsl.h < 0.1 || hsl.h > 0.9, "Red hue should be near 0/1");
        assert!(hsl.s > 0.5, "Red should be saturated");
    }

    #[test]
    fn test_known_color_green() {
        // Pure green in sRGB
        let green = RGB {
            r: 0.0,
            g: 1.0,
            b: 0.0,
        };
        let hsl = srgb_to_okhsl(green);
        assert!(
            hsl.h > 0.2 && hsl.h < 0.5,
            "Green hue should be around 0.33"
        );
        assert!(hsl.s > 0.5, "Green should be saturated");
    }

    #[test]
    fn test_known_color_blue() {
        // Pure blue in sRGB
        let blue = RGB {
            r: 0.0,
            g: 0.0,
            b: 1.0,
        };
        let hsl = srgb_to_okhsl(blue);
        assert!(hsl.h > 0.5 && hsl.h < 0.8, "Blue hue should be around 0.67");
        assert!(hsl.s > 0.5, "Blue should be saturated");
    }

    #[test]
    fn test_gray_desaturation() {
        // Gray colors should have very low saturation
        let gray = RGB {
            r: 0.5,
            g: 0.5,
            b: 0.5,
        };
        let hsl = srgb_to_okhsl(gray);
        assert!(
            hsl.s < 0.01,
            "Gray should have near-zero saturation, got {}",
            hsl.s
        );
    }
}
