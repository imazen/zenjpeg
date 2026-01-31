// Profile jpegli encoding pipeline for comparison with zenjpeg Rust.
//
// Reads a PPM file, encodes N times using jpegli's C API.
// Designed for use with callgrind, cachegrind, and heaptrack.
//
// Build:
//   cd internal/jpegli-cpp/build
//   g++ -O2 -g -o profile_jpegli ../../../internal/profile_jpegli.cc \
//     -I.. -Llib -ljpegli-static -lhwy -lpthread -lm
//
// Usage:
//   valgrind --tool=callgrind ./profile_jpegli /tmp/test_profile.ppm 5
//   valgrind --tool=cachegrind ./profile_jpegli /tmp/test_profile.ppm 5
//   heaptrack ./profile_jpegli /tmp/test_profile.ppm 5

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "lib/jpegli/encode.h"

struct Image {
    int width;
    int height;
    std::vector<unsigned char> pixels;
};

static bool load_ppm(const char* path, Image* img) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", path);
        return false;
    }

    char magic[3];
    if (fscanf(f, "%2s", magic) != 1 || strcmp(magic, "P6") != 0) {
        fprintf(stderr, "Not a P6 PPM file\n");
        fclose(f);
        return false;
    }

    int w, h, maxval;
    // Skip comments
    int c;
    while ((c = fgetc(f)) != EOF) {
        if (c == '#') {
            while ((c = fgetc(f)) != EOF && c != '\n') {}
        } else if (c >= '0' && c <= '9') {
            ungetc(c, f);
            break;
        }
    }

    if (fscanf(f, "%d %d %d", &w, &h, &maxval) != 3) {
        fprintf(stderr, "Failed to parse PPM header\n");
        fclose(f);
        return false;
    }
    fgetc(f);  // consume single whitespace after maxval

    if (maxval != 255) {
        fprintf(stderr, "Unsupported maxval: %d\n", maxval);
        fclose(f);
        return false;
    }

    img->width = w;
    img->height = h;
    size_t npixels = (size_t)w * h * 3;
    img->pixels.resize(npixels);
    size_t nread = fread(img->pixels.data(), 1, npixels, f);
    fclose(f);

    if (nread != npixels) {
        fprintf(stderr, "Short read: %zu < %zu\n", nread, npixels);
        return false;
    }
    return true;
}

static size_t encode_jpegli(const Image& img, std::vector<unsigned char>* output) {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpegli_std_error(&jerr);
    jpegli_create_compress(&cinfo);

    unsigned char* outbuf = nullptr;
    unsigned long outsize = 0;
    jpegli_mem_dest(&cinfo, &outbuf, &outsize);

    cinfo.image_width = img.width;
    cinfo.image_height = img.height;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;

    jpegli_set_defaults(&cinfo);
    jpegli_set_quality(&cinfo, 85, TRUE);

    // 4:2:0 subsampling
    cinfo.comp_info[0].h_samp_factor = 2;
    cinfo.comp_info[0].v_samp_factor = 2;
    cinfo.comp_info[1].h_samp_factor = 1;
    cinfo.comp_info[1].v_samp_factor = 1;
    cinfo.comp_info[2].h_samp_factor = 1;
    cinfo.comp_info[2].v_samp_factor = 1;

    // Optimize Huffman tables (two-pass)
    cinfo.optimize_coding = TRUE;

    // Baseline sequential (not progressive)
    jpegli_set_progressive_level(&cinfo, 0);

    jpegli_start_compress(&cinfo, TRUE);

    // Write scanlines one at a time (like the Rust streaming encoder)
    int stride = img.width * 3;
    while (cinfo.next_scanline < cinfo.image_height) {
        JSAMPROW row = (JSAMPROW)&img.pixels[cinfo.next_scanline * stride];
        jpegli_write_scanlines(&cinfo, &row, 1);
    }

    jpegli_finish_compress(&cinfo);
    size_t result_size = outsize;

    // Copy output
    if (output) {
        output->assign(outbuf, outbuf + outsize);
    }

    free(outbuf);
    jpegli_destroy_compress(&cinfo);

    return result_size;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input.ppm> [iterations]\n", argv[0]);
        return 1;
    }

    const char* ppm_path = argv[1];
    int iterations = argc > 2 ? atoi(argv[2]) : 5;

    Image img;
    if (!load_ppm(ppm_path, &img)) return 1;

    fprintf(stderr, "Loaded %dx%d image (%zu bytes)\n",
            img.width, img.height, img.pixels.size());
    fprintf(stderr, "Encoding %d iterations, Q85, 4:2:0, optimize_huffman=true\n",
            iterations);

    std::vector<unsigned char> output;
    for (int i = 0; i < iterations; i++) {
        size_t sz = encode_jpegli(img, i == 0 ? &output : nullptr);
        if (i == 0) {
            fprintf(stderr, "Output size: %zu bytes\n", sz);
        }
    }

    fprintf(stderr, "Done.\n");
    return 0;
}
