//! Debug UltraHDR detection issue

use std::fs;
use zenjpeg::ultrahdr::UltraHdrExtras;

fn main() {
    let data = fs::read("tests/images/ultrahdr_sample.jpg").unwrap();

    // First, check ultrahdr_rs
    println!("=== ultrahdr_rs ===");
    let uhdr_decoder = ultrahdr_rs::Decoder::new(&data).unwrap();
    println!("is_ultrahdr: {}", uhdr_decoder.is_ultrahdr());
    println!("has metadata: {}", uhdr_decoder.metadata().is_some());
    println!("has gainmap: {}", uhdr_decoder.gainmap_jpeg().is_some());
    if let Some(gm) = uhdr_decoder.gainmap_jpeg() {
        println!("gainmap size: {} bytes", gm.len());
        // Find where gainmap starts in the file
        for i in 0..data.len() - gm.len() {
            if &data[i..i + gm.len()] == gm {
                println!("gainmap found at file offset: {} (0x{:X})", i, i);
                break;
            }
        }
    }

    // Now manually parse XMP and MPF
    println!("\n=== Manual parsing ===");

    const XMP_NS: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";
    const MPF_SIG: &[u8] = b"MPF\0";

    let mut pos = 2; // After SOI
    while pos < data.len() - 1 {
        if data[pos] != 0xFF {
            pos += 1;
            continue;
        }
        pos += 1;
        if pos >= data.len() {
            break;
        }

        let marker = data[pos];
        pos += 1;

        // Skip padding
        if marker == 0xFF || marker == 0x00 {
            continue;
        }

        // EOI
        if marker == 0xD9 {
            println!("Found EOI at pos {}", pos - 2);
            break;
        }

        // RST markers (no length)
        if marker >= 0xD0 && marker <= 0xD7 {
            continue;
        }

        // SOI has no length
        if marker == 0xD8 {
            continue;
        }

        // Read length
        if pos + 2 > data.len() {
            break;
        }
        let length = ((data[pos] as usize) << 8) | (data[pos + 1] as usize);
        if length < 2 {
            break;
        }
        pos += 2;
        let data_len = length - 2;

        if pos + data_len > data.len() {
            break;
        }

        let seg_data = &data[pos..pos + data_len];

        match marker {
            0xE1 => {
                // APP1
                if seg_data.starts_with(XMP_NS) {
                    let xmp =
                        std::str::from_utf8(&seg_data[XMP_NS.len()..]).unwrap_or("<invalid>");
                    println!("Found XMP at pos {} ({} bytes)", pos - 4, seg_data.len());
                    println!(
                        "  contains hdrgm:Version: {}",
                        xmp.contains("hdrgm:Version")
                    );
                    println!(
                        "  contains hdrgm:GainMapMax: {}",
                        xmp.contains("hdrgm:GainMapMax")
                    );
                }
            }
            0xE2 => {
                // APP2
                if seg_data.starts_with(MPF_SIG) {
                    println!("Found MPF at pos {} ({} bytes)", pos - 4, seg_data.len());
                    // Parse MPF
                    let is_le = &seg_data[4..6] == b"II";
                    println!("  Endianness: {}", if is_le { "LE" } else { "BE" });

                    let read_u16 = |p: usize| -> u16 {
                        if is_le {
                            u16::from_le_bytes([seg_data[p], seg_data[p + 1]])
                        } else {
                            u16::from_be_bytes([seg_data[p], seg_data[p + 1]])
                        }
                    };
                    let read_u32 = |p: usize| -> u32 {
                        if is_le {
                            u32::from_le_bytes([
                                seg_data[p],
                                seg_data[p + 1],
                                seg_data[p + 2],
                                seg_data[p + 3],
                            ])
                        } else {
                            u32::from_be_bytes([
                                seg_data[p],
                                seg_data[p + 1],
                                seg_data[p + 2],
                                seg_data[p + 3],
                            ])
                        }
                    };

                    let ifd_offset = read_u32(8);
                    let ifd_pos = 4 + ifd_offset as usize;
                    println!("  IFD offset: {} (pos in data: {})", ifd_offset, ifd_pos);

                    let num_entries = read_u16(ifd_pos);
                    println!("  IFD entries: {}", num_entries);

                    for i in 0..num_entries as usize {
                        let entry_pos = ifd_pos + 2 + i * 12;
                        let tag = read_u16(entry_pos);
                        let typ = read_u16(entry_pos + 2);
                        let count = read_u32(entry_pos + 4);
                        let value_or_offset = read_u32(entry_pos + 8);
                        println!(
                            "    Entry {}: tag=0x{:04X}, type={}, count={}, value/offset={}",
                            i, tag, typ, count, value_or_offset
                        );

                        if tag == 0xB002 {
                            // MP Entry
                            let mp_offset = value_or_offset as usize + 4; // relative to after "MPF\0"
                            let num_images = count / 16;
                            println!(
                                "    -> MP Entry: {} images at offset {}",
                                num_images, mp_offset
                            );

                            for j in 0..num_images as usize {
                                let ep = mp_offset + j * 16;
                                let attr = read_u32(ep);
                                let size = read_u32(ep + 4);
                                let offset = read_u32(ep + 8);
                                let type_code = attr & 0x00FFFFFF;
                                println!(
                                    "       Image {}: type=0x{:06X}, size={}, offset={}",
                                    j, type_code, size, offset
                                );

                                // Check what's at that offset
                                if j > 0 && offset > 0 {
                                    // mpf_header_pos = position in FILE of the byte right after "MPF\0"
                                    // pos currently points to the first byte of segment data
                                    // MPF\0 is at the start of segment data
                                    // So mpf_header_pos = pos + 4 (skip past "MPF\0")
                                    let mpf_header_pos = pos + 4;
                                    let abs_offset = mpf_header_pos + offset as usize;
                                    println!(
                                        "         Absolute offset: {} (mpf_header_pos={})",
                                        abs_offset, mpf_header_pos
                                    );

                                    if abs_offset + 2 <= data.len() {
                                        let first_bytes = &data[abs_offset
                                            ..abs_offset.min(data.len()).min(abs_offset + 10)];
                                        println!("         First bytes: {:02X?}", first_bytes);
                                        if data[abs_offset] == 0xFF && data[abs_offset + 1] == 0xD8
                                        {
                                            println!("         -> Valid JPEG SOI!");
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }

        pos += data_len;
    }

    // Decode the standard way and check extras
    println!("\n=== zenjpeg standard decode extras ===");
    use zenjpeg::decoder::Decoder;
    let decoded = Decoder::new().decode(&data).expect("decode");
    if let Some(extras) = decoded.extras() {
        println!("has_xmp: {}", extras.xmp().is_some());
        println!("is_ultrahdr: {}", extras.is_ultrahdr());
        if let Some(xmp) = extras.xmp() {
            println!("xmp contains hdrgm:Version: {}", xmp.contains("hdrgm:Version"));
        }
        if let Some(mpf) = extras.mpf() {
            println!("mpf: {} images", mpf.images.len());
            for (i, img) in mpf.images.iter().enumerate() {
                println!(
                    "  Image {}: type={:?}, offset={}, size={}",
                    i, img.image_type, img.offset, img.size
                );
            }
        } else {
            println!("mpf: None");
        }
        println!("secondary_images count: {}", extras.secondary_images().len());
    } else {
        println!("No extras!");
    }

    // Now test extract_gainmap_early behavior
    println!("\n=== zenjpeg UltraHdrReader ===");
    use zenjpeg::ultrahdr::{UltraHdrMode, UltraHdrReaderConfig};

    let config = UltraHdrReaderConfig::new()
        .mode(UltraHdrMode::SdrAndGainMap)
        .preserve_metadata(true);

    match Decoder::new().ultrahdr_reader(&data, config) {
        Ok(reader) => {
            println!("UltraHdrReader created successfully");
            println!("  is_ultrahdr: {}", reader.is_ultrahdr());
            println!("  has metadata: {}", reader.metadata().is_some());
        }
        Err(e) => {
            println!("UltraHdrReader creation failed: {:?}", e);
        }
    }
}
