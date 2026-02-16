# okhsl-embedded
Bart Massey 2026

A `no_std` Rust implementation of OKHSL and OKHSV color
space conversions with f32 floats, derived from Björn
Ottosson's C++ `ok_color` code. See [Ottosson's blog
post](https://bottosson.github.io/posts/colorpicker/) about
color picking for more information.

## Features

- `no_std` compatible for embedded systems
- Uses f32 floating point throughout
- Includes OKHSL, OKHSV, and Oklab color spaces
- sRGB and custom gamma correction support
- Multiple gamut clipping strategies

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies.okhsl-embedded]
version = "0.1"
git = "https://github.com/BartMassey/okhsl-embedded"
```

## Example

```rust
use okhsl_embedded::{HSL, RGB, okhsl_to_srgb};

let hsl = HSL { h: 0.5, s: 0.8, l: 0.6 };
let rgb = okhsl_to_srgb(hsl);
```

## License

* Original: MIT License - Copyright (c) 2021 Björn Ottosson

  See `cpp-src/ok_color.h` for original implementation and full license text.

* This work: MIT License - Copyright (c) 2026 Bart Massey
