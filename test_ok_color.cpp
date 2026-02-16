// Unit tests for ok_color.h
// Mirrors the Rust tests to ensure compatibility

#include "ok_color.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>

using namespace ok_color;

const float EPSILON = 1e-5f;
int test_count = 0;
int test_passed = 0;

bool approx_eq(float a, float b, float epsilon = EPSILON) {
    return fabsf(a - b) < epsilon;
}

bool rgb_approx_eq(RGB a, RGB b, float epsilon = EPSILON) {
    return approx_eq(a.r, b.r, epsilon) && approx_eq(a.g, b.g, epsilon) && approx_eq(a.b, b.b, epsilon);
}

bool hsl_approx_eq(HSL a, HSL b, float epsilon = 1e-3f) {
    return approx_eq(a.h, b.h, epsilon) && approx_eq(a.s, b.s, epsilon) && approx_eq(a.l, b.l, epsilon);
}

bool hsv_approx_eq(HSV a, HSV b, float epsilon = 1e-3f) {
    return approx_eq(a.h, b.h, epsilon) && approx_eq(a.s, b.s, epsilon) && approx_eq(a.v, b.v, epsilon);
}

bool lab_approx_eq(Lab a, Lab b, float epsilon = EPSILON) {
    return approx_eq(a.L, b.L, epsilon) && approx_eq(a.a, b.a, epsilon) && approx_eq(a.b, b.b, epsilon);
}

#define TEST(name) \
    void test_##name(); \
    void run_##name() { \
        test_count++; \
        printf("Running test_%s... ", #name); \
        test_##name(); \
        test_passed++; \
        printf("PASSED\n"); \
    } \
    void test_##name()

#define ASSERT(cond, ...) \
    do { \
        if (!(cond)) { \
            printf("FAILED\n  Assertion failed: " #cond "\n  "); \
            printf(__VA_ARGS__); \
            printf("\n"); \
            exit(1); \
        } \
    } while(0)

// Test helper functions
TEST(clamp) {
    ASSERT(clamp(0.5f, 0.0f, 1.0f) == 0.5f, "clamp(0.5, 0, 1) should be 0.5");
    ASSERT(clamp(-0.5f, 0.0f, 1.0f) == 0.0f, "clamp(-0.5, 0, 1) should be 0.0");
    ASSERT(clamp(1.5f, 0.0f, 1.0f) == 1.0f, "clamp(1.5, 0, 1) should be 1.0");
    ASSERT(clamp(0.3f, 0.2f, 0.8f) == 0.3f, "clamp(0.3, 0.2, 0.8) should be 0.3");
}

TEST(sgn) {
    ASSERT(sgn(5.0f) == 1.0f, "sgn(5) should be 1");
    ASSERT(sgn(-5.0f) == -1.0f, "sgn(-5) should be -1");
    ASSERT(sgn(0.0f) == 0.0f, "sgn(0) should be 0");
}

TEST(toe_toe_inv_roundtrip) {
    float values[] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    for (float val : values) {
        float toe_val = toe(val);
        float recovered = toe_inv(toe_val);
        ASSERT(approx_eq(recovered, val, EPSILON),
            "toe/toe_inv roundtrip failed for %f: got %f", val, recovered);
    }
}

TEST(srgb_transfer_roundtrip) {
    float values[] = {0.0f, 0.001f, 0.01f, 0.1f, 0.5f, 0.9f, 1.0f};
    for (float val : values) {
        float gamma = srgb_transfer_function(val);
        float linear = srgb_transfer_function_inv(gamma);
        ASSERT(approx_eq(linear, val, EPSILON),
            "sRGB transfer roundtrip failed for %f: got %f", val, linear);
    }
}

TEST(linear_srgb_oklab_roundtrip) {
    RGB test_colors[] = {
        {1.0f, 0.0f, 0.0f},  // Red
        {0.0f, 1.0f, 0.0f},  // Green
        {0.0f, 0.0f, 1.0f},  // Blue
        {1.0f, 1.0f, 1.0f},  // White
        {0.5f, 0.5f, 0.5f},  // Gray
        {0.3f, 0.7f, 0.2f},  // Random
    };

    for (const auto& color : test_colors) {
        Lab lab = linear_srgb_to_oklab(color);
        RGB recovered = oklab_to_linear_srgb(lab);
        ASSERT(rgb_approx_eq(color, recovered, 1e-4f),
            "RGB->Oklab->RGB roundtrip failed for (%f,%f,%f): got (%f,%f,%f)",
            color.r, color.g, color.b, recovered.r, recovered.g, recovered.b);
    }
}

TEST(okhsl_srgb_black) {
    HSL black_hsl = {0.0f, 0.0f, 0.0f};
    RGB rgb = okhsl_to_srgb(black_hsl);
    ASSERT(rgb_approx_eq(rgb, {0.0f, 0.0f, 0.0f}, EPSILON),
        "Black OKHSL should convert to black sRGB");
}

TEST(okhsl_srgb_white) {
    HSL white_hsl = {0.0f, 0.0f, 1.0f};
    RGB rgb = okhsl_to_srgb(white_hsl);
    ASSERT(rgb_approx_eq(rgb, {1.0f, 1.0f, 1.0f}, EPSILON),
        "White OKHSL should convert to white sRGB");
}

TEST(okhsl_srgb_roundtrip) {
    HSL test_hsl[] = {
        {0.0f, 1.0f, 0.5f},    // Red
        {0.333f, 1.0f, 0.5f},  // Green
        {0.667f, 1.0f, 0.5f},  // Blue
        {0.5f, 0.5f, 0.5f},    // Cyan-ish
        {0.1f, 0.8f, 0.6f},    // Orange-ish
    };

    for (const auto& hsl : test_hsl) {
        RGB rgb = okhsl_to_srgb(hsl);
        HSL recovered = srgb_to_okhsl(rgb);
        ASSERT(hsl_approx_eq(hsl, recovered, 1e-3f),
            "OKHSL->sRGB->OKHSL roundtrip failed for (%f,%f,%f): got (%f,%f,%f)",
            hsl.h, hsl.s, hsl.l, recovered.h, recovered.s, recovered.l);
    }
}

TEST(okhsv_srgb_roundtrip) {
    HSV test_hsv[] = {
        {0.0f, 1.0f, 1.0f},    // Red
        {0.333f, 1.0f, 1.0f},  // Green
        {0.667f, 1.0f, 1.0f},  // Blue
        {0.5f, 0.5f, 0.8f},    // Cyan-ish
        {0.1f, 0.8f, 0.6f},    // Orange-ish
    };

    for (const auto& hsv : test_hsv) {
        RGB rgb = okhsv_to_srgb(hsv);
        HSV recovered = srgb_to_okhsv(rgb);
        ASSERT(hsv_approx_eq(hsv, recovered, 1e-3f),
            "OKHSV->sRGB->OKHSV roundtrip failed for (%f,%f,%f): got (%f,%f,%f)",
            hsv.h, hsv.s, hsv.v, recovered.h, recovered.s, recovered.v);
    }
}

TEST(gamut_clip_preserve_chroma_in_gamut) {
    RGB in_gamut = {0.5f, 0.3f, 0.8f};
    RGB clipped = gamut_clip_preserve_chroma(in_gamut);
    ASSERT(rgb_approx_eq(in_gamut, clipped, EPSILON),
        "In-gamut color should not be clipped");
}

TEST(gamut_clip_out_of_gamut) {
    RGB out_of_gamut = {1.5f, -0.2f, 0.5f};
    RGB clipped = gamut_clip_preserve_chroma(out_of_gamut);

    // Should be in valid range
    ASSERT(clipped.r >= 0.0f && clipped.r <= 1.0f, "Red should be in [0,1]");
    ASSERT(clipped.g >= 0.0f && clipped.g <= 1.0f, "Green should be in [0,1]");
    ASSERT(clipped.b >= 0.0f && clipped.b <= 1.0f, "Blue should be in [0,1]");
}

TEST(compute_max_saturation) {
    // Test for red hue (a=1, b=0)
    float s_red = compute_max_saturation(1.0f, 0.0f);
    ASSERT(s_red > 0.0f, "Max saturation for red should be positive");

    // Test for green hue
    float s_green = compute_max_saturation(-0.5f, 0.866f);
    ASSERT(s_green > 0.0f, "Max saturation for green should be positive");

    // Test for blue hue
    float s_blue = compute_max_saturation(-0.5f, -0.866f);
    ASSERT(s_blue > 0.0f, "Max saturation for blue should be positive");
}

TEST(find_cusp) {
    // Test for red hue
    LC cusp = find_cusp(1.0f, 0.0f);
    ASSERT(cusp.L > 0.0f && cusp.L < 1.0f, "L_cusp should be in (0, 1)");
    ASSERT(cusp.C > 0.0f, "C_cusp should be positive");

    // Test ST conversion
    ST st = to_ST(cusp);
    ASSERT(st.S > 0.0f, "S should be positive");
    ASSERT(st.T > 0.0f, "T should be positive");
}

TEST(get_ST_mid) {
    ST st_mid = get_ST_mid(1.0f, 0.0f);
    ASSERT(st_mid.S > 0.0f, "S_mid should be positive");
    ASSERT(st_mid.T > 0.0f, "T_mid should be positive");
}

TEST(get_Cs) {
    Cs cs = get_Cs(0.5f, 1.0f, 0.0f);
    ASSERT(cs.C_0 > 0.0f, "C_0 should be positive");
    ASSERT(cs.C_mid > 0.0f, "C_mid should be positive");
    ASSERT(cs.C_max > 0.0f, "C_max should be positive");
    ASSERT(cs.C_mid <= cs.C_max, "C_mid should be <= C_max");
}

TEST(known_color_red) {
    // Pure red in sRGB should have hue near 0
    RGB red = {1.0f, 0.0f, 0.0f};
    HSL hsl = srgb_to_okhsl(red);
    ASSERT(hsl.h < 0.1f || hsl.h > 0.9f, "Red hue should be near 0/1, got %f", hsl.h);
    ASSERT(hsl.s > 0.5f, "Red should be saturated, got %f", hsl.s);
}

TEST(known_color_green) {
    // Pure green in sRGB
    RGB green = {0.0f, 1.0f, 0.0f};
    HSL hsl = srgb_to_okhsl(green);
    ASSERT(hsl.h > 0.2f && hsl.h < 0.5f, "Green hue should be around 0.33, got %f", hsl.h);
    ASSERT(hsl.s > 0.5f, "Green should be saturated, got %f", hsl.s);
}

TEST(known_color_blue) {
    // Pure blue in sRGB
    RGB blue = {0.0f, 0.0f, 1.0f};
    HSL hsl = srgb_to_okhsl(blue);
    ASSERT(hsl.h > 0.5f && hsl.h < 0.8f, "Blue hue should be around 0.67, got %f", hsl.h);
    ASSERT(hsl.s > 0.5f, "Blue should be saturated, got %f", hsl.s);
}

TEST(gray_desaturation) {
    // Gray colors should have very low saturation
    RGB gray = {0.5f, 0.5f, 0.5f};
    HSL hsl = srgb_to_okhsl(gray);
    ASSERT(hsl.s < 0.01f, "Gray should have near-zero saturation, got %f", hsl.s);
}

int main() {
    printf("Running C++ ok_color tests...\n\n");

    run_clamp();
    run_sgn();
    run_toe_toe_inv_roundtrip();
    run_srgb_transfer_roundtrip();
    run_linear_srgb_oklab_roundtrip();
    run_okhsl_srgb_black();
    run_okhsl_srgb_white();
    run_okhsl_srgb_roundtrip();
    run_okhsv_srgb_roundtrip();
    run_gamut_clip_preserve_chroma_in_gamut();
    run_gamut_clip_out_of_gamut();
    run_compute_max_saturation();
    run_find_cusp();
    run_get_ST_mid();
    run_get_Cs();
    run_known_color_red();
    run_known_color_green();
    run_known_color_blue();
    run_gray_desaturation();

    printf("\n========================================\n");
    printf("Test Results: %d/%d tests passed\n", test_passed, test_count);
    printf("========================================\n");

    return (test_passed == test_count) ? 0 : 1;
}
