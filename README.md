# okhsl-embedded

A `no_std` Rust implementation of OKHSL and OKHSV color space
conversions with f32 floats, based on Björn Ottosson's ok_color
library.

## Features

- `no_std` compatible for embedded systems
- Uses f32 floating point throughout
- Includes OKHSL, OKHSV, and Oklab color spaces
- sRGB and custom gamma correction support
- Multiple gamut clipping strategies

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
okhsl-embedded = "0.1"
libm = "0.2"
```

## Example

```rust
use okhsl_embedded::{HSL, RGB, okhsl_to_srgb};

let hsl = HSL { h: 0.5, s: 0.8, l: 0.6 };
let rgb = okhsl_to_srgb(hsl);
```

## License

MIT License - Copyright (c) 2021 Björn Ottosson

See ok_color.h for original implementation and full license text.
